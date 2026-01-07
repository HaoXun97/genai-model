import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import chromadb

# We'll process multiple PDFs under this directory
from pathlib import Path
import hashlib
import json
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

json_dir = Path("Law-json")
persist_dir = "chroma_db"
manifest_path = Path(persist_dir) / "manifest.json"


# helper: file sha256
def file_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# helper: manifest load/save
def load_manifest(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {}


def save_manifest(manifest, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                 encoding='utf-8')


manifest = load_manifest(manifest_path)


# Splitter (keeps reasonably sized chunks; used for sub-chunking long articles)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", "。", "？", "！", ";", ";", " ", ""]
)

HEADER_RE = re.compile(
    r"(?m)(?:^|\n\s*\n)[ \t]*(第\s*(?:[0-9０-９]+(?:[-‐‑–—][0-9０-９]+)*)\s*條|第\s*(?:[一二三四五六七八九十百千萬零]+)\s*(?:編|篇|章|節))(?=\s|$)")

# Use BAAI/bge-m3 for embeddings (adjust if you use another model)
embed_model_name = "BAAI/bge-m3"
embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Loaded embedding model: {embed_model_name} on device={device}")

# Initialize persistent Chroma in ./chroma_db
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(
    "legislation_whitepaper_embeddings")

# Process each PDF file in the directory, build per-file chunks_with_meta,
# and upsert in batches
batch_size = 32  # adjust based on your GPU memory

json_paths = sorted(json_dir.glob("*.json"))
print(f"Found {len(json_paths)} json files in {json_dir}")

for json_path in json_paths:
    json_name = json_path.name
    json_hash = file_sha256(json_path)
    prev = manifest.get(str(json_path))
    if prev and prev.get("hash") == json_hash:
        print(f"Skipping {json_name} (unchanged)")
        continue

    # if previous entry exists, delete old ids to avoid stale chunks
    if prev and prev.get("ids"):
        try:
            collection.delete(ids=prev.get("ids"))
            print(f"Deleted {len(prev.get('ids'))} old chunks for {json_name}")
        except Exception:
            print(f"Warning: unable to delete old ids for {json_name};"
                  f"they may remain in DB")

    # load JSON and build segments according to schema
    try:
        text = json_path.read_text(encoding='utf-8-sig')
    except Exception:
        text = json_path.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON {json_path}: {e}")
        data = {}
    document_segments = []

    # If JSON contains multiple laws, iterate each Law entry
    laws = data.get('Laws') if isinstance(data, dict) else None
    if isinstance(laws, list) and laws:
        for law_idx, law in enumerate(laws):
            law_name = law.get('LawName')
            law_url = law.get('LawURL')
            law_level = law.get('LawLevel')
            law_category = law.get('LawCategory')

            la = law.get('LawArticles')
            if isinstance(la, list):
                for art in la:
                    if isinstance(art, dict):
                        art_no = (art.get('ArticleNo') or '').strip()
                        content = art.get('ArticleContent') or ''
                        art_type_raw = (art.get('ArticleType') or '').strip()
                    else:
                        art_no = None
                        content = str(art)
                        art_type_raw = ''
                    # normalize ArticleType if provided
                    art_type = None
                    if art_type_raw:
                        at = art_type_raw.upper()
                        if any(ch in art_type_raw for ch in [
                                '編', '篇', '章', '節']) or at.startswith('C'):
                            art_type = 'C'
                        elif '條' in art_type_raw or at.startswith('A'):
                            art_type = 'A'
                        else:
                            art_type = at
                    # determine header from ArticleNo or ArticleContent
                    header = art_no if art_no else None
                    if not header:
                        m = HEADER_RE.search(content)
                        if m:
                            header = (m.group(1) or m.group(0)).strip()
                        else:
                            # fallback: relaxed match ignoring spaces
                            fm = re.search(r"第\s*([0-9０-９]+(?:[-‐‑–—][0-9０-９]+)*|[一二三四五六七八九十百千萬零]+)\s*(條|編|篇|章|節)", content)
                            if fm:
                                header = fm.group(0).strip()
                    # strip spaces for robust detection
                    header_stripped = (re.sub(r"\s+|\u3000", "", header)
                                       if header else None)
                    # infer article type if still missing
                    if not art_type:
                        if header_stripped and '條' in header_stripped:
                            art_type = 'A'
                        elif header_stripped and re.search(r'(編|篇|章|節)',
                                                           header_stripped):
                            art_type = 'C'
                        else:
                            art_type = None
                    # set header_type for internal counters
                    header_type = ('article' if (header_stripped
                                                 and header_stripped.endswith(
                                                     '條')) else None)
                    seg_entry = {
                        'header': header,
                        'header_type': header_type,
                        'article_type': art_type,
                        'text': content,
                        'start_page': 0,
                        'end_page': 0,
                        'law_name': law_name,
                        'law_url': law_url,
                        'law_level': law_level,
                        'law_category': law_category
                    }
                    document_segments.append(seg_entry)
            elif la is not None:
                # LawArticles is a big string — try to split by header regex
                text = str(la)
                matches = list(HEADER_RE.finditer(text))
                if not matches:
                    document_segments.append(
                        {'header': None,
                         'header_type': None,
                         'article_type': None,
                         'text': text,
                         'law_name': law_name,
                         'law_url': law_url,
                         'law_level': law_level,
                         'law_category': law_category})
                else:
                    for i, m in enumerate(matches):
                        start = m.start()
                        end = (matches[i+1].start()
                               if i + 1 < len(matches) else len(text))
                        header_text = (m.group(1) or m.group(0)).strip()
                        header_text_stripped = re.sub(
                            r"\s+|\u3000", "", header_text)
                        htype = ('article' if header_text_stripped.endswith(
                            '條') else 'unknown')
                        # infer article_type: 'A' for 條, 'C' for 編/章/節
                        if header_text_stripped.endswith('條'):
                            art_type = 'A'
                        elif re.search(r'(編|篇|章|節)', header_text_stripped):
                            art_type = 'C'
                        else:
                            art_type = None
                        segment_text = text[start:end].strip()
                        document_segments.append(
                            {'header': header_text,
                             'header_type': htype,
                             'article_type': art_type,
                             'text': segment_text,
                             'law_name': law_name,
                             'law_url': law_url,
                             'law_level': law_level,
                             'law_category': law_category})
            else:
                # Law has no LawArticles — fallback to other fields per law
                content_candidates = []
                for key in ['LawForeword', 'LawHistories',
                            'LawEffectiveNote', 'LawName']:
                    val = law.get(key)
                    if val:
                        content_candidates.append(str(val))
                if not content_candidates:
                    content_candidates.append(
                        json.dumps(law, ensure_ascii=False))
                full_text = "\n\n".join(content_candidates)
                document_segments.append({
                    'header': None,
                    'header_type': None,
                    'article_type': None,
                    'text': full_text,
                    'law_name': law_name,
                    'law_url': law_url,
                    'law_level': law_level,
                    'law_category': law_category})
    else:
        # fallback: collect other textual fields from top-level
        content_candidates = []
        for key in ['LawForeword', 'ArticleContent',
                    'LawHistories', 'LawEffectiveNote', 'LawName']:
            val = data.get(key)
            if val:
                content_candidates.append(str(val))
        if not content_candidates:
            content_candidates.append(json.dumps(data, ensure_ascii=False))
        full_text = "\n\n".join(content_candidates)
        document_segments.append({
            'header': None,
            'header_type': None,
            'article_type': None,
            'text': full_text,
            'law_name': None,
            'law_url': None,
            'law_level': None,
            'law_category': None
        })

    # split document_segments into subchunks and prepare for embedding
    chunks_with_meta = []
    counters = {"part": -1, "chapter": -1, "section": -1, "article": -1}

    for seg_idx, seg in enumerate(document_segments):
        header = seg.get('header')
        htype = seg.get('header_type')
        seg_text = seg.get('text')
        start_page = seg.get('start_page')
        end_page = seg.get('end_page')

        # update counters for articles
        if htype == 'article':
            counters['article'] += 1

        # Only index segments that are articles (ArticleType 'A')
        article_type = seg.get('article_type')
        if article_type != 'A':
            # skip non-article segments (e.g., chapters/parts)
            continue

        sub_chunks = text_splitter.split_text(seg_text)
        for sub_idx, chunk in enumerate(sub_chunks):
            meta = {
                "file": json_name,
                "header": header,
                "article_type": seg.get('article_type'),
                "law_name": seg.get('law_name'),
                "law_url": seg.get('law_url'),
                "law_level": seg.get('law_level'),
                "law_category": seg.get('law_category')
            }
            chunks_with_meta.append((chunk, meta))

    print(f"Indexing {json_name}: {len(chunks_with_meta)} chunks; "
          f"processing in batches of {batch_size}")

    file_ids = []
    for start_idx in range(0, len(chunks_with_meta), batch_size):
        batch = chunks_with_meta[start_idx:start_idx + batch_size]
        docs = [t for t, m in batch]
        metadatas = [m for t, m in batch]

        # Generate embeddings for the batch (normalize for cosine)
        emb_batch = embed_model.encode(
            docs, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True)

        # Prepare stable ids including file and start index
        ids = [f"{json_name}__chunk_{start_idx + i}" for i in range(len(docs))]

        # Upsert to allow re-running without duplicates
        collection.upsert(
            embeddings=emb_batch.tolist(),
            documents=docs,
            ids=ids,
            metadatas=metadatas
        )

        file_ids.extend(ids)

        # Clean up to free memory
        del emb_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # update manifest entry for this file
    manifest[str(json_path)] = {
        "hash": json_hash,
        "n_chunks": len(file_ids),
        "ids": file_ids,
        "indexed_at": datetime.utcnow().isoformat() + "Z"
    }
    save_manifest(manifest, manifest_path)
    print(f"Finished indexing {json_name}")

# Example query
query = "行政機關依第一百零二條給予相對人陳述意見之機會時，應以書面記載下列事項通知相對人，必要時並公告之︰"
query_embedding = embed_model.encode(
    [query], convert_to_numpy=True, show_progress_bar=False,
    normalize_embeddings=True)[0]

# Query the vector store and show sources
n_results_to_return = min(collection.count(), 5)
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=n_results_to_return,
    include=['documents', 'metadatas', 'distances']
)

print("\n" + "="*50 + "\n")

docs_list = results.get('documents', [[]])[0]
metas_list = results.get('metadatas', [[]])[0]
filtered = [(doc, meta) for doc, meta in zip(docs_list, metas_list) if meta.get('article_type') == 'A']
for i, (doc, meta) in enumerate(filtered, start=1):
    print(f"Result {i} (file={meta.get('file')}, "
          f"law={meta.get('law_name')}, "
          f"category={meta.get('law_category')}, "
          f"url={meta.get('law_url')}, "
          f"header={meta.get('header')}):")
    print(doc.strip())
    print("\n" + "="*50 + "\n")

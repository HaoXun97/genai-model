import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import chromadb

# Define the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# We'll process multiple PDFs under this directory
from pathlib import Path
import hashlib
import json
from datetime import datetime

pdf_dir = Path("Legislation")
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
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

manifest = load_manifest(manifest_path)

import re

# Splitter (keeps reasonably sized chunks; used for sub-chunking long articles)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", "。", "？", "！", ";", ";", " ", ""]
)

# Header regex (captures headers)
# Heuristic: require the header to start a paragraph (preceded by start or blank line) to avoid matching inline references
# Only allow Arabic or fullwidth digits (and hyphenated groups like 49-1 or 55‑2) for '條'; for 編/篇/章/節 require Chinese numerals
# Accept hyphen-like characters (-, ‐, ‑, –, —) between numeric groups and allow multiple groups, e.g., 49-1-2
# Ensure the entire header (either '第...條' or '第...編/章/...') is captured in group 1 so .group(1) is never None
HEADER_RE = re.compile(r"(?m)(?:^|\n\s*\n)[ \t]*(第\s*(?:[0-9０-９]+(?:[-‐‑–—][0-9０-９]+)*)\s*條|第\s*(?:[一二三四五六七八九十百千萬零]+)\s*(?:編|篇|章|節))(?=\s|$)")

# Build chunks per page and attach richer hierarchical metadata (file, page, header, header_type, part/chapter/section, subchunk_index)
chunks_with_meta = []

def split_page_into_header_segments(page_text: str):
    """Return a list of tuples (header_text, header_type, segment_text).
    If no header is found, returns [(None, None, full_page_text)]."""
    matches = list(HEADER_RE.finditer(page_text))
    if not matches:
        return [(None, None, page_text)]

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(page_text)
        header_text = (m.group(1) or m.group(0)).strip()
        # determine header type by suffix
        if header_text.endswith('編') or header_text.endswith('篇'):
            header_type = 'part'
        elif header_text.endswith('章'):
            header_type = 'chapter'
        elif header_text.endswith('節'):
            header_type = 'section'
        elif header_text.endswith('條'):
            header_type = 'article'
        else:
            header_type = 'unknown'
        segment_text = page_text[start:end].strip()
        segments.append((header_text, header_type, segment_text))
    return segments

# Legacy single-file preprocessing removed — we perform per-file processing inside the main loop over PDF files below.

# Use BAAI/bge-m3 for embeddings (adjust if you use another model)
embed_model_name = "BAAI/bge-m3"
embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Loaded embedding model: {embed_model_name} on device={device}")

# Initialize persistent Chroma in ./chroma_db
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("legislation_whitepaper_embeddings")

# Process each PDF file in the directory, build per-file chunks_with_meta, and upsert in batches
batch_size = 32  # adjust based on your GPU memory

pdf_paths = sorted(pdf_dir.glob("*.pdf"))
print(f"Found {len(pdf_paths)} pdf files in {pdf_dir}")

for pdf_path in pdf_paths:
    pdf_name = pdf_path.name
    pdf_hash = file_sha256(pdf_path)
    prev = manifest.get(str(pdf_path))
    if prev and prev.get("hash") == pdf_hash:
        print(f"Skipping {pdf_name} (unchanged)")
        continue

    # if previous entry exists, delete old ids to avoid stale chunks
    if prev and prev.get("ids"):
        try:
            collection.delete(ids=prev.get("ids"))
            print(f"Deleted {len(prev.get('ids'))} old chunks for {pdf_name}")
        except Exception:
            print(f"Warning: unable to delete old ids for {pdf_name}; they may remain in DB")

    # load this PDF and build segments
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # reuse earlier logic to build document_segments per file (cross-page merge included)
    document_segments = []
    for page_idx, page in enumerate(pages, start=1):
        page_text = page.page_content
        segments = split_page_into_header_segments(page_text)

        for header, htype, seg_text in segments:
            norm_header = header.strip() if header else None
            seg_entry = {
                "header": norm_header,
                "header_type": htype,
                "text": seg_text,
                "start_page": page_idx,
                "end_page": page_idx
            }

            if not document_segments:
                document_segments.append(seg_entry)
                continue

            last = document_segments[-1]
            if norm_header is None or (last['header'] is not None and norm_header == last['header']):
                # smart concatenate to avoid truncation at page breaks (preserve words/punctuation)
                def is_cjk(ch: str):
                    return bool(ch) and ("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or "\u20000" <= ch <= "\u2a6df")

                def smart_join(a: str, b: str) -> str:
                    a = a.rstrip('\n\r ')
                    b = b.lstrip('\n\r ')
                    if not a:
                        return b
                    if not b:
                        return a
                    # remove trailing hyphen-like characters
                    if a.endswith(('-', '‐', '‑', '–', '—')):
                        a = a[:-1]
                    # if both end/start are ASCII alnum, put a space
                    if a[-1].isalnum() and b[0].isalnum():
                        return a + ' ' + b
                    # otherwise, for CJK don't add space
                    if is_cjk(a[-1]) or is_cjk(b[0]):
                        return a + b
                    # default join with a single space
                    return a + ' ' + b

                last['text'] = smart_join(last['text'], seg_text)
                last['end_page'] = page_idx
            else:
                document_segments.append(seg_entry)

    # split document_segments into subchunks and prepare for embedding
    chunks_with_meta = []
    context = {"part": None, "chapter": None, "section": None}
    counters = {"part": -1, "chapter": -1, "section": -1, "article": -1}

    for seg_idx, seg in enumerate(document_segments):
        header = seg.get('header')
        htype = seg.get('header_type')
        seg_text = seg.get('text')
        start_page = seg.get('start_page')
        end_page = seg.get('end_page')

        # update hierarchy context and counters
        if htype == 'part':
            counters['part'] += 1
            context['part'] = header
        elif htype == 'chapter':
            counters['chapter'] += 1
            context['chapter'] = header
        elif htype == 'section':
            counters['section'] += 1
            context['section'] = header
        elif htype == 'article':
            counters['article'] += 1

        sub_chunks = text_splitter.split_text(seg_text)
        for sub_idx, chunk in enumerate(sub_chunks):
            meta = {
                "file": pdf_name,
                "start_page": start_page,
                "end_page": end_page,
                "header": header,
                "header_type": htype,
                "part": context.get('part'),
                "chapter": context.get('chapter'),
                "section": context.get('section'),
                "subchunk_index": sub_idx,
                "article_index": counters.get('article') if htype == 'article' else None
            }
            chunks_with_meta.append((chunk, meta))

    print(f"Indexing {pdf_name}: {len(chunks_with_meta)} chunks; processing in batches of {batch_size}")

    file_ids = []
    for start_idx in range(0, len(chunks_with_meta), batch_size):
        batch = chunks_with_meta[start_idx:start_idx + batch_size]
        docs = [t for t, m in batch]
        metadatas = [m for t, m in batch]

        # Generate embeddings for the batch (normalize for cosine)
        emb_batch = embed_model.encode(docs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

        # Prepare stable ids including file and start index
        ids = [f"{pdf_name}__chunk_{start_idx + i}" for i in range(len(docs))]

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
    manifest[str(pdf_path)] = {
        "hash": pdf_hash,
        "n_chunks": len(file_ids),
        "ids": file_ids,
        "indexed_at": datetime.utcnow().isoformat() + "Z"
    }
    save_manifest(manifest, manifest_path)
    print(f"Finished indexing {pdf_name}")

# Example query
query = "有下列各款情形之一者，得以再審之訴對於確定終局判決聲明不服。"
# query = "最高行政法院駁回上訴或廢棄原判決自為裁判"
query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]

# Query the vector store and show sources
n_results_to_return = min(collection.count(), 5)
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=n_results_to_return,
    include=['documents', 'metadatas', 'distances']
)

for i, (doc, meta) in enumerate(zip(results.get('documents', [[]])[0], results.get('metadatas', [[]])[0])):
    print(f"Result {i+1} (file={meta.get('file')}, start_page={meta.get('start_page')}, end_page={meta.get('end_page')}, header={meta.get('header')}, header_type={meta.get('header_type')}, part={meta.get('part')}, chapter={meta.get('chapter')}, section={meta.get('section')}, subchunk_index={meta.get('subchunk_index')}):")
    print(doc.strip())
    print("\n" + "="*50 + "\n")

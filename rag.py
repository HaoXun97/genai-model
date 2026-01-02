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

# Load the PDF and extract text
pdf_file_path = "Legislation/中華民國刑法.pdf"
loader = PyPDFLoader(pdf_file_path)
pages = loader.load()

import re

# Splitter (keeps reasonably sized chunks; used for sub-chunking long articles)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=128,
    length_function=len,
    separators=["\n\n", "\n", "。", "？", "！", ";", ";", " ", ""]
)

# Header regex (captures headers like 第1編/第1章/第1節/第1條 etc.)
HEADER_RE = re.compile(r"(第\s*[0-9０-９一二三四五六七八九十百千萬零]+\s*(?:編|篇|章|節|條))")

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
        header_text = m.group(1).strip()
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

# First, build a sequence of document-level segments and merge across pages when headers continue
document_segments = []  # each item: {header, header_type, text, start_page, end_page}

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
        # Merge if continuation: either header repeated (same header) or current header is None (continuation text)
        if norm_header is None:
            # continuation of previous segment
            last['text'] = last['text'] + '\n' + seg_text
            last['end_page'] = page_idx
        elif last['header'] is not None and norm_header == last['header']:
            last['text'] = last['text'] + '\n' + seg_text
            last['end_page'] = page_idx
        else:
            # new segment starting on this page
            document_segments.append(seg_entry)

# Now split document_segments into sub-chunks (retain merged page ranges) and emit metadata
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
            "file": os.path.basename(pdf_file_path),
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

# Use BAAI/bge-m3 for embeddings (adjust if you use another model)
embed_model_name = "BAAI/bge-m3"
embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Loaded embedding model: {embed_model_name} on device={device}")

# Process chunks in batches to reduce memory usage and upsert to persistent Chroma
batch_size = 32  # adjust based on your GPU memory
print(f"Total chunks: {len(chunks_with_meta)}; processing in batches of {batch_size}")

# Initialize persistent Chroma in ./chroma_db
persist_dir = "chroma_db"
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection("legislation_whitepaper_embeddings")

for start_idx in range(0, len(chunks_with_meta), batch_size):
    batch = chunks_with_meta[start_idx:start_idx + batch_size]
    docs = [t for t, m in batch]
    metadatas = [m for t, m in batch]

    # Generate embeddings for the batch (normalize for cosine)
    emb_batch = embed_model.encode(docs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

    # Prepare stable ids including file and start index
    ids = [f"{os.path.basename(pdf_file_path)}_chunk_{start_idx + i}" for i in range(len(docs))]

    # Upsert to allow re-running without duplicates
    collection.upsert(
        embeddings=emb_batch.tolist(),
        documents=docs,
        ids=ids,
        metadatas=metadatas
    )

    # Clean up to free memory
    del emb_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Example query
query = "省法規與國家法律牴觸者無效"
query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]

# Query the vector store and show sources
n_results_to_return = min(collection.count(), 5)
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=n_results_to_return,
    include=['documents', 'metadatas', 'distances']
)

for i, (doc, meta) in enumerate(zip(results.get('documents', [[]])[0], results.get('metadatas', [[]])[0])):
    print(f"Result {i+1} (file={meta.get('file')}, page={meta.get('page')}, header={meta.get('header')}, header_type={meta.get('header_type')}, part={meta.get('part')}, chapter={meta.get('chapter')}, section={meta.get('section')}, subchunk_index={meta.get('subchunk_index')}):")
    print(doc.strip())
    print("\n" + "="*50 + "\n")

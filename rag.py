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
pdf_file_path = "Legislation/中華民國憲法.pdf"
loader = PyPDFLoader(pdf_file_path)
pages = loader.load()

# Splitter (keeps reasonably large chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Build chunks per page and attach metadata (file, page, chunk_index)
chunks_with_meta = []
for page_idx, page in enumerate(pages, start=1):
    page_text = page.page_content
    page_chunks = text_splitter.split_text(page_text)
    for chunk_idx, chunk in enumerate(page_chunks):
        meta = {
            "file": os.path.basename(pdf_file_path),
            "page": page_idx,
            "chunk_index": chunk_idx
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
query = "有議決法律案、預算案及國家其他重要事項之權的是哪個院？"
query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]

# Query the vector store and show sources
n_results_to_return = min(collection.count(), 3)
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=n_results_to_return,
    include=['documents', 'metadatas', 'distances']
)

for i, (doc, meta) in enumerate(zip(results.get('documents', [[]])[0], results.get('metadatas', [[]])[0])):
    print(f"Result {i+1} (file={meta.get('file')}, page={meta.get('page')}, chunk_index={meta.get('chunk_index')}):")
    print(doc.strip())
    print("\n" + "="*50 + "\n")

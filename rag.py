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

# Combine all pages into a single text
text = " ".join([page.page_content for page in pages])

# Split the text into larger chunks with more overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 2000
    chunk_overlap=100,  # 200
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(text)

# Use BAAI/bge-m3 for embeddings (high-quality embeddings; may require more memory)
# Note: ensure you have access to the model and sufficient resources to load it.
embed_model_name = "BAAI/bge-m3"
embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Loaded embedding model: {embed_model_name} on device={device}")

# Process chunks in batches to reduce memory usage and batch-add to Chroma
batch_size = 32  # adjust based on your GPU memory
print(f"Total chunks: {len(chunks)}; processing in batches of {batch_size}")

# Initialize Chroma
client = chromadb.Client()
collection = client.get_or_create_collection("legislation_whitepaper_embeddings")

for start_idx in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start_idx:start_idx + batch_size]

    # Generate embeddings for the batch using sentence-transformers (L2-normalized for cosine search)
    emb_batch = embed_model.encode(batch_chunks, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

    # Prepare ids
    ids = [f"chunk_{j}" for j in range(start_idx, start_idx + len(batch_chunks))]

    # Batch add to Chroma (convert numpy array to list of lists)
    collection.add(
        embeddings=emb_batch.tolist(),
        documents=batch_chunks,
        ids=ids
    )

    # Clean up to free memory
    del emb_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Query embedding generation (use sentence-transformers; L2-normalized for cosine search)
query = "有議決法律案、預算案及國家其他重要事項之權的是哪個院？"
query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]

# Query the vector store
n_results_to_return = min(collection.count(), 3)
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=n_results_to_return
)

# Display the retrieved document content (text chunks)
for i, document in enumerate(results['documents'][0]):
    print(f"Result {i+1}:")
    print(document.strip())
    print("\n" + "="*50 + "\n")

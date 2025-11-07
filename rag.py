import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
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
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(text)

# --- NEW: Create the quantization configuration ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --- UPDATED: Load model using the new quantization_config ---
model = AutoModel.from_pretrained(
    "Llama-3.2-11B-Vision-Instruct",
    quantization_config=quantization_config,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision-Instruct")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Tokenize the chunks with padding and truncation
inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move the document input tensors to the GPU
inputs = {key: val.to(device) for key, val in inputs.items()}

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

# Initialize Chroma
client = chromadb.Client()
collection = client.get_or_create_collection("legislation_whitepaper_embeddings")

# Store embeddings and chunks in Chroma
for i, embedding in enumerate(embeddings):
    # Move embeddings back to CPU for numpy/ChromaDB
    collection.add(
        embeddings=[embedding.cpu().detach().numpy().tolist()],
        documents=[chunks[i]],
        ids=[f"chunk_{i}"]
    )

# Query embedding generation
query = "有議決法律案、預算案及國家其他重要事項之權的是哪個院？"
query_inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)

# Move the query input tensors to the GPU
query_inputs = {key: val.to(device) for key, val in query_inputs.items()}

query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1)

# Query the vector store
n_results_to_return = min(collection.count(), 3)
results = collection.query(
    # Move query embedding back to CPU for numpy/ChromaDB
    query_embeddings=query_embedding.cpu().detach().numpy().tolist(),
    n_results=n_results_to_return
)

# Display the retrieved document content (text chunks)
for i, document in enumerate(results['documents'][0]):
    print(f"Result {i+1}:")
    print(document.strip())
    print("\n" + "="*50 + "\n")

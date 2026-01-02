import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import textwrap
import chromadb
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

# Configuration
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "legislation_whitepaper_embeddings"
EMBED_MODEL_NAME = "BAAI/bge-m3"  # same as your indexing
TOP_K = 3  # number of retrieved passages to include in prompt
MAX_CHARS_PER_DOC = 1200  # truncate each retrieved doc to avoid too long prompts

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model
print(f"Loading embedding model {EMBED_MODEL_NAME} on device={device}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
print("Embedding model loaded.")

# Open Chroma DB
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

if collection.count() == 0:
    print("⚠️  注意：Chroma 集合目前為空。請先執行 `python rag.py` 來建立索引，或確保 `chroma_db` 資料夾存在且已包含向量。")

# Helper: retrieve top-k docs
def retrieve(query: str, top_k: int = TOP_K):
    emb = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]
    n_results = min(collection.count(), top_k) if collection.count() > 0 else 0
    if n_results == 0:
        return []
    results = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs_list, metas_list, dists_list):
        docs.append({
            "doc": doc,
            "meta": meta,
            "distance": dist
        })
    return docs

# Build a prompt that includes retrieved context and the user question
def build_prompt(query: str, retrieved_docs: list):
    if not retrieved_docs:
        ctx = "(未找到相關法條或段落)"
    else:
        parts = []
        for i, r in enumerate(retrieved_docs, start=1):
            meta = r.get("meta", {})
            header = meta.get("header") or "(無標題)"
            src = f"{meta.get('file', 'unknown')} (pages {meta.get('start_page')} - {meta.get('end_page')})  標題: {header}"
            # truncate doc text
            text = r.get("doc", "").strip()
            if len(text) > MAX_CHARS_PER_DOC:
                text = text[:MAX_CHARS_PER_DOC] + "..."
            part = f"【來源 {i}】 {src}\n{text}"
            parts.append(part)
        ctx = "\n\n---\n\n".join(parts)

    prompt = textwrap.dedent(f"""
    Human: 以下是從法規資料庫檢索到的相關內容，請參考並且根據這些內容回答下面的問題；回答後請簡短列出你引用的來源 (檔名、起訖頁、標題)。

    {ctx}

    問題: {query}

    悟空:
    """)
    return prompt

# Load LLM for inference
print("\nLoading fine-tuned model for inference...")
MODEL_PATH = "./bart_finetuned"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded. 你可以開始輸入問題，輸入 /exit 結束。\n")

# Interactive loop

def interactive_loop():
    while True:
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nBye!")
            return
        if user_input.strip().lower() == "/exit":
            print("Bye!")
            break
        if not user_input.strip():
            continue

        # safety flag
        skip = False

        # Retrieve
        retrieved = retrieve(user_input, TOP_K)

        # Build prompt
        prompt = build_prompt(user_input, retrieved)

        # Tokenize with safety: ensure prompt is string, enable truncation and fallback on errors
        if not isinstance(prompt, str):
            prompt = str(prompt)
        try:
            max_len = getattr(tokenizer, "model_max_length", None) or getattr(tokenizer, "max_length", None) or 4096
            truncation_max = max(32, max_len - 64)
            inputs = tokenizer(
                text=prompt,
                images=None,
                return_tensors="pt",
                truncation=True,
                max_length=truncation_max,
            ).to(model.device)
        except TypeError as e:
            print(f"⚠️ Tokenizer TypeError: {e}. 只使用簡化 prompt 並重試。")
            safe_prompt = f"Human: {user_input}\n\n悟空:"
            try:
                inputs = tokenizer(
                    text=safe_prompt,
                    images=None,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(256, getattr(tokenizer, "model_max_length", 1024)-1),
                ).to(model.device)
            except Exception as e2:
                print(f"⚠️ 無法 tokenized，即刻跳過：{e2}")
                skip = True
        except Exception as e:
            print(f"⚠️ Tokenizer 錯誤：{e}。跳過此輸入。")
            continue

        if skip:
            continue

        # Generate — use max_new_tokens to avoid ValueError when input length already equals max_length
        max_new_tokens = 256  # adjust as needed
        try:
            max_pos = getattr(model.config, "max_position_embeddings", None)
            input_len = inputs["input_ids"].shape[1]
            if max_pos is not None:
                max_new_tokens = min(max_new_tokens, max_pos - input_len - 1)
                if max_new_tokens < 1:
                    max_new_tokens = 32
        except Exception:
            pass

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            print(f"⚠️ 生成時發生錯誤：{e}。嘗試用簡短提示重試。")
            try:
                safe_prompt = f"Human: {user_input}\n悟空:"
                inputs = tokenizer(text=safe_prompt, images=None, return_tensors="pt", truncation=True, max_length=256).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except Exception as e2:
                print(f"⚠️ 仍無法產生回應：{e2}。跳過此輸入。")
                skip = True

        if skip:
            continue

        # Decode safely
        try:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️ 解碼回應時出錯：{e}。跳過此輸入。")
            continue

        response_part = response.split(prompt)
        if len(response_part) > 1:
            answer = response_part[1].strip()
        else:
            answer = response.split("悟空:")[-1].strip()

        print(f"\n悟空: {answer}\n")

        # Print retrieved sources for transparency
        if retrieved:
            print("🔎 檢索到的來源：")
            for i, r in enumerate(retrieved, start=1):
                meta = r.get("meta", {})
                print(f" {i}. {meta.get('file', 'unknown')} | pages {meta.get('start_page')} | 條文: {meta.get('header')}")
            print("")

if __name__ == "__main__":
    interactive_loop()

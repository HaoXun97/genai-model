# Hello World Demo for Llama 3.2 11B Vision-Instruct
# ---------------------------------------------------

# 1. Install required libraries if not already installed
#!pip install transformers accelerate safetensors torch pillow --upgrade

# 2. Import dependencies
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# 3. Set model path (local download)
model_path = "Llama-3.2-11B-Vision-Instruct"

# 4. Load processor (handles text + vision input) and model
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",       # Automatically uses GPU if available
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# 5. Create a simple Hello World prompt
prompt = "Hello Llama! Can you introduce yourself in one sentence?"

# 6. Preprocess input
inputs = processor(text=prompt, images=None, return_tensors="pt").to(model.device)

# 7. Generate output
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

# 8. Decode result
response = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
print("Model Response:\n", response)

import os
import sys
import torch
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoProcessor, MllamaForConditionalGeneration, TextIteratorStreamer
from threading import Thread

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = os.environ.get("LLAMA_VISION_PATH", "Llama-3.2-11B-Vision-Instruct")
USE_BF16 = torch.cuda.is_available()
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------
# Load model & processor
# ---------------------------
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16 if USE_BF16 else torch.float32,
)
model.eval()

# ---------------------------
# Conversation state
# ---------------------------
history: List[Dict[str, Any]] = []
SYSTEM_PROMPT = "You are a helpful, concise assistant."

def build_inputs(
    history: List[Dict[str, Any]],
    user_text: Optional[str] = None,
    user_images: Optional[List[Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[List[Any]]]:
    """
    Build chat-formatted prompt + image bundle using processor.apply_chat_template.
    """
    conversation = []
    # optional system instruction
    conversation.append({"role": "system", "content": SYSTEM_PROMPT})

    # Find all images from conversation history
    all_images = []
    
    # prior turns
    for turn in history:
        if turn["role"] == "user":
            if turn.get("images"):
                conversation.append({
                    "role": "user",
                    "content": [{"type": "text", "text": turn["content"]}] +
                               [{"type": "image"} for _ in turn["images"]],
                })
                # Collect images for the processor
                all_images.extend(turn["images"])
            else:
                conversation.append({"role": "user", "content": turn["content"]})
        else:
            conversation.append({"role": "assistant", "content": turn["content"]})

    # current user turn
    if user_text is not None:
        if user_images:
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}] +
                           [{"type": "image"} for _ in user_images],
            })
            all_images.extend(user_images)
        else:
            conversation.append({"role": "user", "content": user_text})

    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # For Llama 3.2 Vision, we need to pass ALL images that appear in the conversation
    # even if they were from previous turns
    final_images = all_images if all_images else None
    
    # Processor packs text+images to tensors
    inputs = processor(
        text=prompt,
        images=final_images,
        return_tensors="pt",
    ).to(model.device)

    return inputs, final_images

def generate_stream(inputs: Dict[str, torch.Tensor]) -> str:
    """
    Stream tokens to stdout during generation. Returns the finalized assistant text.
    """
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
    
    # Extract generation arguments - don't pass processor inputs directly
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "streamer": streamer,
    }
    
    # Add vision inputs if present
    if "pixel_values" in inputs:
        generation_kwargs["pixel_values"] = inputs["pixel_values"]
    if "aspect_ratio_ids" in inputs:
        generation_kwargs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"]
    if "aspect_ratio_mask" in inputs:
        generation_kwargs["aspect_ratio_mask"] = inputs["aspect_ratio_mask"]

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # stream to terminal
    sys.stdout.write("Assistant: ")
    sys.stdout.flush()
    chunks = []
    for token_text in streamer:
        sys.stdout.write(token_text)
        sys.stdout.flush()
        chunks.append(token_text)
    sys.stdout.write("\n")
    return "".join(chunks).strip()

def main():
    print("=== Llama 3.2 11B Vision-Instruct — Interactive Chat ===")
    print("Tips:")
    print("  • Type your question and press Enter.")
    print("  • Type '/img path1 [path2 ...]' to load images, then ask a question.")
    print("  • Type '/quit' to exit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"/quit", "quit", "exit"}:
            break

        # Handle image loading
        if user.startswith("/img"):
            # Lazily import PIL
            try:
                from PIL import Image
            except Exception as e:
                print(f"Error: Pillow not available ({e}). Try: pip install pillow")
                continue

            paths = user.split()[1:]
            if not paths:
                print("Usage: /img /path/to/img1 [/path/to/img2 ...]")
                continue
                
            imgs = []
            for p in paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    print(f"Loaded: {p}")
                except Exception as e:
                    print(f"Could not open {p}: {e}")
            if not imgs:
                print("No valid images loaded.")
                continue
                
            # Ask for a text question next
            q = input("Image(s) loaded. Ask your first question about them:\nYou: ").strip()
            if not q:
                print("No question provided.")
                continue
            
            # Build + stream with images
            try:
                inputs, _ = build_inputs(history, user_text=q, user_images=imgs)
                answer = generate_stream(inputs)
                history.append({"role": "user", "content": q, "images": imgs})
                history.append({"role": "assistant", "content": answer})
            except Exception as e:
                print(f"Error during generation: {e}")
            continue

        # Normal turn (text-only, but may reference previous images)
        try:
            inputs, _ = build_inputs(history, user_text=user, user_images=None)
            answer = generate_stream(inputs)
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Try loading images first with '/img /path/to/image.jpg'")

if __name__ == "__main__":
    main()

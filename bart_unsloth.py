import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    gc.collect()
    torch.cuda.empty_cache()

def prepare_dataset(file_path, tokenizer, max_length=512):
    """Prepare dataset with proper tokenization for training (text-only on multimodal models)."""
    dataset = load_dataset('json', data_files=file_path)['train']

    # Detect if tokenizer is a multimodal processor (e.g., Mllama / Llama 3.2 Vision)
    is_multimodal = hasattr(tokenizer, "image_processor")

    def tokenize_function(examples):
        if 'prompt' not in examples or 'completion' not in examples:
            raise KeyError("The dataset must have 'prompt' and 'completion' fields.")

        # CORRECTED: Combined multi-line f-string into one line with \n
        # prompts = [f"Human: {q}\nBart:" for q in examples['prompt']]
        prompts = [f"Human: {q}\n悟空:" for q in examples['prompt']]
        responses = examples['completion']

        # Join prompt + response into a single text. We will mask the prompt part in labels.
        texts = [p + " " + r for p, r in zip(prompts, responses)]

        common_kwargs = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Important: for multimodal processors, pass text=... and images=None explicitly
        if is_multimodal:
            enc = tokenizer(text=texts, images=None, **common_kwargs)
            # Tokenize prompts alone to compute mask length
            prompt_enc = tokenizer(text=prompts, images=None, **common_kwargs)
        else:
            enc = tokenizer(texts, **common_kwargs)
            prompt_enc = tokenizer(prompts, **common_kwargs)

        # Convert lists to tensors AFTER the HF datasets map pipeline (let set_format handle torch)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = [ids.copy() for ids in input_ids]

        # Mask the prompt tokens (avoid computing loss on prompt part)
        for i in range(len(texts)):
            prompt_len = len([tok for tok in prompt_enc["input_ids"][i] if tok != tokenizer.pad_token_id])
            # Replace prompt portion with -100
            labels[i][:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format(type='torch')

    print("Dataset structure (keys):", tokenized_dataset[0].keys())
    print("Dataset length:", len(tokenized_dataset))

    return tokenized_dataset

def load_unsloth_model(model_name, max_seq_length=512, load_in_4bit=True, from_checkpoint=None):
    """Load model using Unsloth for optimized training"""
    
    # This function now correctly loads the base model AND adapters when a checkpoint is provided.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=from_checkpoint if from_checkpoint else model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    
    # CHANGE: Only add new LoRA adapters if we are NOT loading from a checkpoint.
    if from_checkpoint is None:
        print("Adding new LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    else:
        print("Loading existing LoRA adapters from checkpoint...")

    # Force the entire model to the correct dtype
    model.to(torch.bfloat16)
    
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = "<|end_of_text|>"
    
    return model, tokenizer

def fine_tune(model, dataset, tokenizer, output_dir, num_epochs, 
              batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4):
    """Fine-tune the model with custom training loop"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=min(100, total_steps // 10), 
        num_training_steps=total_steps
    )

    print(f"Training for {num_epochs} epochs with {len(dataloader)} batches per epoch")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            total_loss += loss.item() * gradient_accumulation_steps
            
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
        
        clear_gpu_memory()

    return model

def main():
    # Configuration
    base_model_name = "Llama-3.2-11B-Vision-Instruct"
    max_seq_length = 512
    load_in_4bit = True # Changed to True to be consistent with inference settings
    
    print("="*50)
    print("STAGE 1: Informal Language Fine-tuning")
    print("="*50)
    
    informal_model_path = "./informal_finetuned"
    
    # Check if Stage 1 is already complete
    if not os.path.exists(os.path.join(informal_model_path, "adapter_model.safetensors")):
        print("Starting Stage 1 training...")
        
        # Load model and tokenizer
        model, tokenizer = load_unsloth_model(
            base_model_name, 
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit
        )
        
        # Prepare dataset
        informal_dataset = prepare_dataset('informal.jsonl', tokenizer, max_length=max_seq_length)
        
        # Fine-tune
        model = fine_tune(
            model, 
            informal_dataset, 
            tokenizer,
            informal_model_path, 
            num_epochs=3,
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4
        )
        
        # Save final model
        model.save_pretrained(informal_model_path)
        tokenizer.save_pretrained(informal_model_path)
        print(f"Stage 1 complete! Model saved to {informal_model_path}")
        
        # Clear memory before Stage 2
        del model
        clear_gpu_memory()
    else:
        print("Stage 1 already complete. Skipping to Stage 2...")
    
    print("\n" + "="*50)
    print("STAGE 2: Bart Simpson-Specific Fine-tuning")
    print("="*50)
    
    # Load the informal fine-tuned model for Stage 2
    model, tokenizer = load_unsloth_model(
        base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        from_checkpoint=informal_model_path
    )
    
    # Prepare Bart dataset
    bart_dataset = prepare_dataset('bart.jsonl', tokenizer, max_length=max_seq_length)
    
    # Fine-tune on Bart Simpson data
    bart_model_path = "./bart_finetuned"
    model = fine_tune(
        model, 
        bart_dataset, 
        tokenizer,
        bart_model_path, 
        num_epochs=5,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4
    )
    
    # Save final model
    model.save_pretrained(bart_model_path)
    tokenizer.save_pretrained(bart_model_path)
    print(f"\nStage 2 complete! Final model saved to {bart_model_path}")
    
    print("\n" + "="*50)
    print("Fine-tuning completed successfully!")
    print("="*50)

# ============================================
# INFERENCE CODE
# ============================================

def run_inference():
    """Run inference with the fine-tuned Bart model"""
    print("\n" + "="*50)
    print("Loading model for inference...")
    print("="*50)
    
    model_path = "./bart_finetuned"
    
    # Load model in inference mode
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=torch.bfloat16, # Use bfloat16 for consistency
        load_in_4bit=True,
    )
    
    # Enable inference mode for 2x faster generation
    FastLanguageModel.for_inference(model)
    
    print("Model loaded! Ask a question. Type '/exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == '/exit':
            break
        
        # Prepare input using the correct format from training
        # prompt = f"Human: {user_input}\nBart:"
        prompt = f"Human: {user_input}\n悟空:"

        # CORRECTED: Explicitly pass 'text' and 'images' arguments
        inputs = tokenizer(
            text=prompt, 
            images=None, 
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_part = response.split(prompt)
        if len(response_part) > 1:
            response = response_part[1].strip()
        else:
            # response = response.split("Bart:")[-1].strip()
            response = response.split("悟空:")[-1].strip()

        # print(f"Bart: {response}\n")
        print(f"悟空: {response}\n")
        
if __name__ == "__main__":
    main()
    # run_inference()

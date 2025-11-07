import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import bitsandbytes as bnb

def prepare_dataset(file_path, tokenizer, max_length=256):
    """Loads a JSONL file and tokenizes it for causal language modeling."""
    dataset = load_dataset("json", data_files=file_path)["train"]

    def tokenize_function(examples):
        if "prompt" not in examples or "completion" not in examples:
            raise KeyError("Dataset must have 'prompt' and 'completion' fields.")

        # Combine prompt and completion for Causal LM fine-tuning
        full_text = [
            # f"Human: {p}\nBart:{c}{tokenizer.eos_token}"
            f"Human: {p}\n悟空:{c}{tokenizer.eos_token}"
            for p, c in zip(examples["prompt"], examples["completion"])
        ]
        model_inputs = tokenizer(
            full_text, truncation=True, max_length=max_length, padding="max_length"
        )

        # Create labels and mask the prompt portion
        labels = torch.tensor(model_inputs["input_ids"])
        # prompts_only = [f"Human: {p}\nBart:" for p in examples["prompt"]]
        prompts_only = [f"Human: {p}\n悟空:" for p in examples["prompt"]]
        prompt_toks = tokenizer(prompts_only, add_special_tokens=False)
        prompt_lengths = [len(p) for p in prompt_toks["input_ids"]]

        for i, length in enumerate(prompt_lengths):
            labels[i, :length] = -100  # Mask prompt tokens

        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format(type="torch")
    return tokenized_dataset


def load_model(model_name, peft_config, peft_model_path=None):
    """Loads the base model with 4-bit quantization and applies PEFT adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )

    if peft_model_path and os.path.exists(os.path.join(peft_model_path, "adapter_config.json")):
        print(f"Loading PEFT adapters from {peft_model_path}")
        model = PeftModel.from_pretrained(base_model, peft_model_path)
    else:
        print("Creating new PEFT model")
        model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()
    return model


def fine_tune(model, dataset, output_dir, num_epochs, batch_size=1, learning_rate=1e-4):
    """Performs the training loop for the PEFT model."""
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * num_epochs
    )

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Data is on CPU, must be moved to the model's device (GPU)
            batch = {k: v.to("cuda") for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        model.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}"))

    return model


def main():
    """Main function to run the two-stage fine-tuning process."""
    model_name = "Llama-3.2-11B-Vision-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    # Stage 1: Informal language fine-tuning
    informal_model_path = "./informal_finetuned"
    if not os.path.exists(os.path.join(informal_model_path, "adapter_model.safetensors")):
        print("--- Stage 1: Informal language fine-tuning ---")
        informal_dataset = prepare_dataset("informal.jsonl", tokenizer)
        model = load_model(model_name, peft_config)
        model = fine_tune(model, informal_dataset, informal_model_path, num_epochs=3)
    else:
        print("Skipping Stage 1: Informal fine-tuned model already exists")

    # Stage 2: Bart Simpson-specific fine-tuning
    print("\n--- Stage 2: Bart Simpson-specific fine-tuning ---")
    bart_dataset = prepare_dataset("bart.jsonl", tokenizer)
    bart_model_path = "./bart_finetuned"
    model = load_model(model_name, peft_config, informal_model_path)
    model = fine_tune(model, bart_dataset, bart_model_path, num_epochs=5)

    model.save_pretrained(bart_model_path)
    tokenizer.save_pretrained(bart_model_path)
    print("\nFine-tuning completed. Final model saved to:", bart_model_path)


if __name__ == "__main__":
    main()

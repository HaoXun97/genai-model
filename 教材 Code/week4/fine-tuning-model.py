import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os

def load_model_and_tokenizer(model_path):
    """Loads the 4-bit quantized model and tokenizer from a PEFT checkpoint."""
    # Step 1: Load the PEFT config to get the base model name
    config = PeftConfig.from_pretrained(model_path)
    base_model_name = config.base_model_name_or_path

    # Step 2: Setup 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Step 3: Load the base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Step 4: Load the PEFT model by combining the base model and adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()  # Set the model to evaluation mode

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generates a response from the model based on a user prompt."""
    # Format the prompt to match the training data
    # full_prompt = f"Human: {prompt}\nBart:"
    full_prompt = f"Human: {prompt}\n悟空:"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

    # Generate the response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract only the new, generated tokens
    response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return response.strip()

def main():
    """Main function to load the model and start a chat session."""
    model_path = "./bart_finetuned"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found at {model_path}")
        return

    print("Loading fine-tuned model...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print("\nAy, caramba! Model loaded. Ask me anything, man. Type '/exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == '/exit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        # print(f"Bart: {response}")
        print(f"悟空: {response}")

if __name__ == "__main__":
    main()

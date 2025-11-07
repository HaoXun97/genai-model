
# ============================================
# INFERENCE CODE
# ============================================

from unsloth import FastLanguageModel
import torch

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
    run_inference()

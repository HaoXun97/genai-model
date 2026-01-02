from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Llama 3.1 model and tokenizer with bfloat16 and gradient checkpointing
model_name = "/home/jj/Llama-3.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with bfloat16 and gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Enable bfloat16 precision
    device_map="auto"  # Automatically map model to available devices
)

# Enable gradient checkpointing to save memory during training/inference
model.gradient_checkpointing_enable()
# Function to get a connection to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('feedback.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the SQLite database with a feedback table
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            generated_text TEXT NOT NULL,
            fluency INTEGER NOT NULL,
            coherence INTEGER NOT NULL,
            relevance INTEGER NOT NULL,
            diversity INTEGER NOT NULL
        )
    ''')
    conn.close()

# Call the function to initialize the database
init_db()

@app.route('/', methods=['GET'])
def home():
    return "AI Text Generation API is running"


@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json
        prompt = data.get('prompt', None)

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        temperature = data.get('temperature', 0.7)
        max_length = data.get('max_length', 100)

        # Encode the input prompt and move it to the model's device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        # Create the attention mask and move it to the model's device
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)

        # Generate the text using the model
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Ensure attention mask is on the same device
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id  # Handle pad_token_id warning
            )

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({'generated_text': generated_text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Route to handle feedback submission
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        # Ensure the request contains JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO feedback (prompt, generated_text, fluency, coherence, relevance, diversity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['prompt'], 
            data['generated_text'], 
            data['fluency'], 
            data['coherence'], 
            data['relevance'], 
            data['diversity']
        ))
        conn.commit()
        conn.close()

        return jsonify({'message': 'Feedback submitted successfully'})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

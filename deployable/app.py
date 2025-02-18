from flask import Flask, request, jsonify
from flask_cors import CORS
from mamba import load_model, TextDataset, generate_text
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and dataset once at startup
try:
    with open('data.txt', 'r') as file:
        text_data = file.read()

    loaded_model, saved_vocab, vocab_size = load_model(256, 2, 'mamba_helpsteer2.pth')
    dataset = TextDataset(text_data, 50, vocab_size)
    dataset.chars = saved_vocab
    dataset.char_to_idx = {ch: i for i, ch in enumerate(saved_vocab)}
    dataset.idx_to_char = {i: ch for i, ch in enumerate(saved_vocab)}
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        
        generated = generate_text(
            loaded_model, 
            prompt, 
            max_length, 
            dataset, 
            hidden_size=256, 
            num_layers=2
        )
        return jsonify({'response': generated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
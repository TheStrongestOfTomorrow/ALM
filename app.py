import os
import sys
import subprocess
import torch
import tiktoken
from flask import Flask, send_from_directory, request, jsonify
from model import ALM, get_tiny_config
from search_agent import SearchAgent
from adaptive_learning import AdaptiveTrainer

app = Flask(__name__, static_folder='web')

# Initialize model for Web Mode
config = get_tiny_config()
model = ALM(config)
enc = tiktoken.get_encoding("gpt2")
search_agent = SearchAgent()
trainer = AdaptiveTrainer(model)

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'chat')

    # Adaptive Learning from every interaction
    trainer.learn_step(prompt, enc)

    if mode == 'search':
        context = search_agent.query(prompt)
        # In a real RAG, we'd feed context to the model
        input_ids = torch.tensor(enc.encode(context + " " + prompt)).unsqueeze(0)
    else:
        input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)

    output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
    response = enc.decode(output_ids[0].tolist()[input_ids.size(1):])

    return jsonify({"response": response})

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

def start_web_mode():
    print("\nStarting ALM-1 Neural Web Server...")
    print("Point your browser to http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)

def start_terminal_mode():
    subprocess.run([sys.executable, "cli.py"])

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("   ALM-1: Adaptive Neural Engine        ")
    print("========================================")
    print("1. Web Mode (Real Neural Interface)")
    print("2. Terminal Mode (Rich Neural TUI)")
    print("3. Exit")
    print("========================================")

    choice = input("\nChoose a mode (1-3): ")

    if choice == '1':
        start_web_mode()
    elif choice == '2':
        start_terminal_mode()
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()

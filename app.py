"""
ALM Web Application
Flask-based web interface for the Adaptive Learning Model.
Supports conversation history, streaming, search agent, and adaptive learning.
"""

import os
import sys
import json
import uuid
import time
import torch
import tiktoken
from flask import Flask, send_from_directory, request, jsonify, Response, stream_with_context
from model import ALM, get_alm_tiny_config, ASSISTANT_TOKEN, END_TOKEN
from search_agent import SearchAgent
from adaptive_learning import AdaptiveTrainer

app = Flask(__name__, static_folder='web')

# Global state
model = None
enc = None
search_agent = None
trainer = None
conversations = {}  # In-memory conversation store
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_model():
    """Initialize or load the ALM model."""
    global model, enc, search_agent, trainer

    enc = tiktoken.get_encoding("gpt2")
    config = get_alm_tiny_config()

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'alm_tiny_best.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading trained checkpoint from {checkpoint_path}")
        model = ALM.load_checkpoint(checkpoint_path, device=device)
    else:
        print("No checkpoint found, initializing fresh model")
        model = ALM(config).to(device)

    model.eval()
    search_agent = SearchAgent()
    trainer = AdaptiveTrainer(model)
    print(f"ALM ready on {device}")


def format_conversation_prompt(messages, max_context=200):
    """Format conversation history into a prompt for the model."""
    # Build the conversation context
    parts = []
    total_tokens = 0

    for msg in messages[-10:]:  # Last 10 messages for context
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'user':
            parts.append(content)
        elif role == 'assistant':
            parts.append(f"{ASSISTANT_TOKEN}{content}")

        total_tokens += len(enc.encode(parts[-1]))
        if total_tokens > max_context:
            break

    # Join with the expected format
    prompt = "\n".join(parts)
    if not prompt.endswith(ASSISTANT_TOKEN):
        prompt += f"\n{ASSISTANT_TOKEN}"

    return prompt


@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "model": "ALM-Tiny (7.8M params)",
        "architecture": "Adaptive MoE Transformer",
        "device": device,
        "conversations": len(conversations)
    })


@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all conversations."""
    conv_list = []
    for conv_id, conv in conversations.items():
        conv_list.append({
            "id": conv_id,
            "title": conv.get("title", "New Chat"),
            "messages": len(conv.get("messages", [])),
            "created": conv.get("created", 0),
            "updated": conv.get("updated", 0)
        })
    return jsonify(conv_list)


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation."""
    data = request.json or {}
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = {
        "id": conv_id,
        "title": data.get("title", "New Chat"),
        "messages": [],
        "system_prompt": data.get("system_prompt", ""),
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 0.9),
        "max_tokens": data.get("max_tokens", 150),
        "created": time.time(),
        "updated": time.time()
    }
    return jsonify(conversations[conv_id])


@app.route('/api/conversations/<conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Get a specific conversation."""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify(conversations[conv_id])


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """Delete a conversation."""
    if conv_id in conversations:
        del conversations[conv_id]
    return jsonify({"status": "deleted"})


@app.route('/api/conversations/<conv_id>/settings', methods=['PUT'])
def update_settings(conv_id):
    """Update conversation settings."""
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404

    data = request.json or {}
    conv = conversations[conv_id]

    for key in ['title', 'system_prompt', 'temperature', 'top_p', 'max_tokens']:
        if key in data:
            conv[key] = data[key]

    conv['updated'] = time.time()
    return jsonify(conv)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Standard chat endpoint (non-streaming)."""
    global model, enc

    data = request.json or {}
    prompt = data.get('prompt', '')
    conv_id = data.get('conversation_id')
    mode = data.get('mode', 'chat')
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 40)
    max_tokens = data.get('max_tokens', 150)
    repetition_penalty = data.get('repetition_penalty', 1.3)

    if not prompt.strip():
        return jsonify({"error": "Empty prompt"}), 400

    # Build conversation context
    messages = []
    if conv_id and conv_id in conversations:
        conv = conversations[conv_id]
        messages = conv.get('messages', [])
        temperature = conv.get('temperature', temperature)
        top_p = conv.get('top_p', top_p)
        max_tokens = conv.get('max_tokens', max_tokens)

    # Add user message
    user_msg = {"role": "user", "content": prompt, "timestamp": time.time()}
    messages.append(user_msg)

    # Build prompt from conversation
    if mode == 'search':
        context = search_agent.query(prompt)
        full_prompt = f"Context: {context}\n\n{prompt}\n{ASSISTANT_TOKEN}"
    else:
        full_prompt = format_conversation_prompt(messages)

    # Tokenize
    input_tokens = enc.encode(full_prompt)
    # Truncate to block_size - max_tokens
    max_input = model.config.block_size - max_tokens
    if len(input_tokens) > max_input:
        input_tokens = input_tokens[-max_input:]

    input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

    response_tokens = output_ids[0].tolist()[input_ids.size(1):]
    response_text = enc.decode(response_tokens).strip()

    # Clean up response - remove any special tokens
    response_text = response_text.replace(ASSISTANT_TOKEN, "").replace(END_TOKEN, "").strip()

    # Adaptive learning step (lightweight)
    try:
        trainer.learn_step(prompt + " " + response_text, enc)
    except Exception:
        pass

    # Save assistant message
    assistant_msg = {"role": "assistant", "content": response_text, "timestamp": time.time()}
    messages.append(assistant_msg)

    # Update conversation
    if conv_id and conv_id in conversations:
        conv = conversations[conv_id]
        conv['messages'] = messages
        conv['updated'] = time.time()
        # Auto-title from first message
        if len(messages) <= 2:
            conv['title'] = prompt[:50] + ("..." if len(prompt) > 50 else "")

    return jsonify({
        "response": response_text,
        "conversation_id": conv_id,
        "mode": mode
    })


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    global model, enc

    data = request.json or {}
    prompt = data.get('prompt', '')
    conv_id = data.get('conversation_id')
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    max_tokens = data.get('max_tokens', 150)

    if not prompt.strip():
        return jsonify({"error": "Empty prompt"}), 400

    # Build context
    messages = []
    if conv_id and conv_id in conversations:
        messages = conversations[conv_id].get('messages', [])

    messages.append({"role": "user", "content": prompt, "timestamp": time.time()})
    full_prompt = format_conversation_prompt(messages)

    input_tokens = enc.encode(full_prompt)
    max_input = model.config.block_size - max_tokens
    if len(input_tokens) > max_input:
        input_tokens = input_tokens[-max_input:]

    input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)

    def generate():
        """Stream tokens one at a time."""
        current_ids = input_ids
        collected_tokens = []

        for i in range(max_tokens):
            with torch.no_grad():
                idx_cond = current_ids[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Repetition penalty
                for tid in current_ids[0].tolist()[-50:]:
                    if 0 <= tid < logits.size(-1):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= 1.3
                        else:
                            logits[0, tid] *= 1.3

                # Top-k
                v, _ = torch.topk(logits, min(40, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                token_text = enc.decode([token_id])

                # Check for end
                if token_text and token_text.strip():
                    collected_tokens.append(token_id)
                    yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"

                current_ids = torch.cat((current_ids, next_token), dim=1)

        # Final response
        full_response = enc.decode(collected_tokens).replace(ASSISTANT_TOKEN, "").strip()
        yield f"data: {json.dumps({'token': '', 'done': True, 'response': full_response})}\n\n"

        # Save to conversation
        if conv_id and conv_id in conversations:
            conv = conversations[conv_id]
            conv['messages'].append({"role": "assistant", "content": full_response, "timestamp": time.time()})
            conv['updated'] = time.time()

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


def start_web_mode():
    """Start the web server."""
    init_model()
    print("\n" + "=" * 50)
    print("  ALM-1 Neural Interface Online")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)


def start_terminal_mode():
    """Start the terminal interface."""
    subprocess.run([sys.executable, "cli.py"])


def main():
    import subprocess
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("   ALM-1: Adaptive Neural Engine        ")
    print("========================================")
    print("1. Web Mode (Real Neural Interface)")
    print("2. Terminal Mode (Rich Neural TUI)")
    print("3. Train Model")
    print("4. Exit")
    print("========================================")

    choice = input("\nChoose a mode (1-4): ")

    if choice == '1':
        start_web_mode()
    elif choice == '2':
        start_terminal_mode()
    elif choice == '3':
        from train import train
        train()
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()

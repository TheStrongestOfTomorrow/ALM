# ALM: Adaptive Learning Model

ALM (Adaptive Learning Model) is a next-generation transformer-based architecture designed for high-performance reasoning, coding, and natural language understanding — built from scratch with a Mixture of Experts (MoE) approach.

## ALM-Tiny: The First Working Model

ALM-Tiny is the first trained model in the ALM family, featuring **7.8 million parameters** and a real Mixture of Experts architecture:

| Feature | Value |
| --- | --- |
| Parameters | 7.8M |
| Architecture | Adaptive MoE Transformer |
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Dim | 128 |
| MoE Experts | 2 per layer |
| Context Window | 256 tokens |
| Training Data | Conversational (code, Q&A, knowledge) |
| License | MIT |

## ALM-1: Flagship Specification (Future)

| Feature | Value |
| --- | --- |
| Parameters | 135.35B |
| Architecture | Adaptive Transformer |
| Training Focus | English & Code |
| License | MIT |

## Quick Start

```bash
pip install torch flask tiktoken rich
python app.py
```

### Modes Available:
1. **Web Mode:** Launches a local Flask server with a professional chat interface featuring conversations sidebar, settings panel, search (RAG) mode, and adaptive learning.
2. **Terminal Mode:** Launches a TUI with rich formatting and conversation support.
3. **Train Mode:** Trains or continues training the model on conversational data.

## Training

Train ALM from scratch or continue training:

```bash
python train.py
```

The training pipeline:
- Uses conversational data with question-answer pairs
- Trains with AdamW optimizer and cosine LR scheduling
- Saves checkpoints to `checkpoints/`
- Supports resuming from the best checkpoint

## Architecture

### Mixture of Experts (MoE)
ALM uses a **Soft MoE** approach where a gating network computes weighted combinations of expert outputs. Each expert is a full MLP that specializes in different types of patterns. The gating network learns to route different inputs to the most relevant experts.

### Adaptive Learning
Unlike static models, ALM implements **online learning** — each user interaction provides signal that fine-tunes the model's weights in real-time. This means ALM progressively improves with use.

### Search Agent (RAG)
The built-in Search Agent uses **TF-IDF retrieval** to find relevant knowledge from a document store, grounding responses in factual context without requiring external embedding models.

## Web Interface

The web interface includes:
- **Conversation sidebar** with search and management
- **Settings panel** with temperature, top-p, top-k, repetition penalty, max tokens
- **System prompt** customization
- **Search/RAG mode** toggle
- **Streaming** and **adaptive learning** toggles
- **Export chat** functionality
- **Mobile responsive** design
- **Dark theme** with professional UI

## GitHub Pages

A static version of the ALM interface is hosted via GitHub Pages. The interface works in demo mode when accessed statically, and provides full AI responses when run locally via `python app.py`.

---

*Part of the ALM (Adaptive-Learning-Model) series.*

<p align="center">
  <img src="https://img.shields.io/badge/ALM-Adaptive_Learning_Model-7c3aed?style=for-the-badge&labelColor=0f0f1a" alt="ALM Badge"/>
  <br/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Runs_in_Browser-Pyodide-F7DF1E?style=flat-square"/>
</p>

<h1 align="center">ALM — Adaptive Learning Model Series</h1>

<p align="center">
  <strong>A family of AI models built from scratch — not wrappers, not APIs — real trained neural networks.</strong>
</p>

<p align="center">
  ALM is a series of transformer-based language models with a focus on <strong>code generation</strong>, <strong>adaptive learning</strong>, and <strong>transparency</strong>. Every model is trained from the ground up with novel architectures, deployable in-browser via Pyodide, and fully open source under the MIT license.
</p>

---

## The Models

### ALM-1 — The Foundation

| Detail | Value |
|--------|-------|
| Parameters | 7.8M |
| Architecture | Adaptive MoE Transformer |
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Dim | 128 |
| MoE Experts | 2 per layer |
| Context Window | 256 tokens |
| Focus | General conversation, Q&A, knowledge |
| License | MIT |

ALM-1 is the first model in the ALM family — a compact 7.8M parameter transformer that proves the concept of Mixture of Experts (MoE) at a tiny scale. It features a soft MoE approach where a gating network computes weighted combinations of expert outputs, allowing the model to route different inputs to specialized sub-networks. ALM-1 also introduces **adaptive learning**, where each user interaction provides signal that fine-tunes the model's weights in real-time, making it progressively better with use.

**Key features:**
- Mixture of Experts (MoE) with learned gating
- Adaptive online learning from user interactions
- Search Agent (RAG) with TF-IDF retrieval
- Web interface with Flask server
- Terminal TUI with Rich formatting
- GitHub Pages deployment via Pyodide

---

### ALM-1-Coder — The Code Specialist

| Detail | Value |
|--------|-------|
| Parameters (Small) | ~20M |
| Parameters (Medium) | ~150M |
| Parameters (Full) | ~1.3B |
| Architecture | Mixture of Agents (MoA) Transformer |
| Agents | 10 named specialists |
| Active Params/Token | ~600M (Full config, MoA sparsity) |
| Optimizations | RoPE, SwiGLU, GQA |
| Context Window | 512–2048 tokens |
| Focus | Code generation, debugging, architecture |
| License | MIT |

ALM-1-Coder is a revolutionary coding model that replaces anonymous MoE experts with **named, role-defined agents** — each with a personality, a job description, and dedicated parameters. Instead of routing tokens to generic "Expert 3", ALM-1-Coder routes them to **SYNTAX** (Senior Developer), **DEBUGGER** (QA Engineer), **ARCHITECT** (Systems Architect), and 7 other specialists who collaborate inside a single transformer.

**Key features:**
- **Mixture of Agents (MoA)** — 10 named agents, not anonymous experts
- **Dynamic Agent Composition** — ENGLISH agent blends existing agents into temporary new ones for novel tasks
- **View Thoughts Mode** — See agents argue and collaborate in real-time
- **RoPE** — Rotary Position Embeddings for better code context
- **SwiGLU** — Modern gated activation for richer representations
- **GQA** — Grouped-Query Attention for efficient inference
- **Role-Conditioned Training** — Agent tokens guide the model during training
- **In-Browser Inference** — Runs entirely in the browser via Pyodide + NumPy
- **Model Selector** — Switch between ALM-1-Coder and ALM-1 from the UI

---

## The 10 Agents

| # | Agent | Business Role | Personality | What It Does |
|---|-------|---------------|-------------|--------------|
| 1 | **ENGLISH** | CEO | Authoritative, decisive, slightly annoyed | Comprehends prompts, routes to agents, composes temporary agents |
| 2 | **SYNTAX** | Senior Developer | Pragmatic, ships fast, hates over-engineering | Writes clean, idiomatic code |
| 3 | **LOGIC** | Algorithms Engineer | Perfectionist, pedantic, loves algorithms | Optimizes algorithms, reasons about correctness |
| 4 | **DEBUGGER** | QA Engineer | Suspicious, thorough, loves finding mistakes | Finds bugs, inserts error handling, defensive checks |
| 5 | **ARCHITECT** | Systems Architect | Abstract thinker, "let us zoom out", never codes | Designs module boundaries, interfaces, data flow |
| 6 | **HTML** | Frontend Developer | Dramatic, passionate, defensive about CSS | Builds web UIs, HTML/CSS/JS |
| 7 | **TRANSLATOR** | Localization Engineer | Chill, neutral, no opinions, just converts | Converts between programming languages |
| 8 | **REASONING** | Chief Strategy Officer | Over-analytical, cautious, "well actually..." | Chain-of-thought reasoning, step-by-step analysis |
| 9 | **THOUGHT** | Project Manager | Diplomatic, peacemaker, exhausted | Coordinates agent-to-agent communication |
| 10 | **SEARCH** | Research Analyst | Helpful, always has a reference, "according to docs..." | Looks up documentation and references |

### Dynamic Agent Composition

When no single agent fits a task, the ENGLISH agent creates a **temporary new agent** by blending existing ones:

```
Prompt: "Optimize this recursive function and add error handling"

ENGLISH composes: 40% LOGIC + 40% DEBUGGER + 20% SYNTAX
→ Temporary agent handles the novel combination
```

This means the model isn't limited to 10 fixed agents — it can theoretically create unlimited agent combinations.

---

## View Thoughts

The **View Thoughts** feature is ALM-1-Coder's transparency superpower. Toggle it on and you can watch the agents collaborate (and argue) in real-time:

- Which agents were activated for each token
- The confidence weight each agent contributed
- A step-by-step trace of how agents collaborated across layers

It's both **entertaining** (watch SYNTAX and ARCHITECT disagree about abstraction levels) and **useful** (understand why the model made specific choices, debug unexpected outputs).

---

## Quick Start

### Try in Your Browser

The easiest way to use ALM — just visit the GitHub Pages deployment. Both models run entirely in your browser via Pyodide. No installation, no server, no API keys.

1. Go to the **GitHub Pages** link
2. Select your model: **ALM-1-Coder** (code specialist) or **ALM-1** (general)
3. Start chatting!

Toggle **View Thoughts** to watch the agents collaborate in real-time.

### Run Locally

#### ALM-1 (Original)

```bash
git clone https://github.com/TheStrongestOfTomorrow/ALM.git
cd ALM/ALM-1
pip install torch flask tiktoken rich
python app.py
```

This launches a local Flask server with a professional chat interface including conversations sidebar, settings panel, search (RAG) mode, and adaptive learning.

#### ALM-1-Coder (Code Specialist)

```bash
git clone https://github.com/TheStrongestOfTomorrow/ALM.git
cd ALM/ALM-1-Coder
pip install torch tiktoken
```

Open `web/index.html` in your browser for the chat UI with View Thoughts and Model Selector.

---

## Architecture Deep Dive

### ALM-1: Mixture of Experts (MoE)

ALM-1 uses a **Soft MoE** approach where a gating network computes weighted combinations of expert outputs. Each expert is a full MLP that specializes in different types of patterns. The gating network learns to route different inputs to the most relevant experts.

### ALM-1-Coder: Mixture of Agents (MoA)

For each token representation **x**:

1. The agent gate computes scores: `g(x) = softmax(W_g · x)`
2. Top-k agents are selected
3. Each selected agent produces output through its expert FFNs
4. Agent role embeddings inject identity signals
5. The final output is the weighted sum across active agents

This allows the model to dynamically compose expertise — e.g., SYNTAX and ARCHITECT may dominate on scaffolding, while LOGIC and DEBUGGER activate on performance-critical loops.

### Shared Optimizations (ALM-1-Coder)

| Component | Description |
|-----------|-------------|
| **RoPE** | Rotary Position Embeddings encode relative positional information directly into attention, enabling better length generalization |
| **SwiGLU** | Gated linear activation (SiLU + GLU) for richer feature representation in feed-forward layers |
| **GQA** | Grouped-Query Attention shares key/value heads across query groups, reducing KV-cache overhead |

---

## Project Structure

```
ALM/
├── README.md                    ← You are here
├── ALM-1/                       ← Original ALM model
│   ├── model.py                 ← 7.8M MoE transformer
│   ├── app.py                   ← Flask web server
│   ├── cli.py                   ← Terminal TUI
│   ├── search_agent.py          ← RAG search agent
│   ├── adaptive_learning.py     ← Online learning
│   └── web_original/            ← Static web interface + weights
│
└── ALM-1-Coder/                 ← Coding specialist model
    ├── model.py                 ← MoA transformer with 10 agents
    └── web/
        ├── index.html           ← Chat UI with View Thoughts + Model Selector
        └── inference.py         ← NumPy inference engine for Pyodide
```

---

## Roadmap

- [x] ALM-1: 7.8M MoE transformer with adaptive learning
- [x] ALM-1-Coder: MoA architecture with 10 named agents
- [x] View Thoughts: Real-time agent collaboration visualization
- [x] Model Selector: Switch between Coder and Normal modes
- [x] In-browser inference via Pyodide
- [x] GitHub Actions training workflow
- [ ] ALM-1-Coder Medium/Full training (needs GPU)
- [ ] 4-bit quantization for mobile deployment
- [ ] ALM-1.5: Updated agent roster
- [ ] ALM-2: Next generation architecture

---

## License

MIT — Use it, modify it, build on it. Just give credit.

---

<p align="center">
  <strong>ALM — Built from scratch, not wrapped from APIs.</strong>
</p>

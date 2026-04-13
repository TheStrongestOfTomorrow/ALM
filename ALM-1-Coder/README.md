# ALM-1-Coder - Mixture of Agents Coding Model

ALM-1-Coder is a Mixture of Agents (MoA) language model purpose-built for code generation. Rather than relying on a single monolithic network, it routes tokens through **10 specialized agent sub-networks**, each an expert in a distinct aspect of programming. A learned gating mechanism selects which agents contribute to each token, producing ensemble-quality output at a fraction of the cost.

---

## The 10 Agents & Their Roles

| # | Agent | Role |
|---|-------|------|
| 1 | **Planner** | Decomposes high-level prompts into sub-tasks and outlines solution structure |
| 2 | **Architect** | Designs module boundaries, interfaces, and data-flow patterns |
| 3 | **Writer** | Generates clean, idiomatic code following language conventions |
| 4 | **Reviewer** | Audits generated code for correctness, style, and anti-patterns |
| 5 | **Debugger** | Inserts defensive checks, error handling, and diagnostic logging |
| 6 | **Optimizer** | Applies performance-aware transformations (vectorization, caching, lazy eval) |
| 7 | **Tester** | Produces unit tests, edge-case inputs, and property-based checks |
| 8 | **Documenter** | Generates docstrings, inline comments, and usage examples |
| 9 | **Refactorer** | Simplifies control flow, removes duplication, enforces DRY/SOLID |
| 10 | **Integrator** | Manages imports, dependency wiring, and cross-module compatibility |

---

## Training

### GitHub Actions (recommended)

1. Go to **Actions → Train & Deploy ALM-1-Coder** in your repository.
2. Click **Run workflow**.
3. Set `train_model` to **true** and click **Run workflow**.
4. The workflow will train for 100 epochs, then upload checkpoints and numpy weights as artifacts.

### Google Colab

Open the notebook in Colab, mount your drive, and run the training cells. Ensure you have `torch` and `tiktoken` installed:

```bash
pip install torch tiktoken
```

Then run:

```bash
python train.py --epochs 100
```

---

## Use in Browser (Pyodide)

ALM-1-Coder ships a fully client-side web interface powered by **Pyodide** — no server required.

1. Open the deployed GitHub Pages URL (or serve `web/index.html` locally).
2. The model weights (numpy format) are fetched and loaded in-browser via Pyodide.
3. Type your prompt and press **Generate** — inference runs entirely in your browser.

---

## View Thoughts

The **View Thoughts** feature exposes the internal reasoning of the model. When enabled, the gating activations for each token are visualized alongside the output, showing:

- Which agents were activated for each token.
- The confidence weight each agent contributed.
- A step-by-step trace of how the Planner, Writer, and Reviewer collaborated.

This makes ALM-1-Coder uniquely interpretable — you can *see* the model think.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **RoPE** (Rotary Position Embeddings) | Encodes relative positional information directly into attention, enabling better length generalization |
| **SwiGLU** | Gated linear activation (SiLU + GLU) for richer feature representation in feed-forward layers |
| **GQA** (Grouped-Query Attention) | Shares key/value heads across query groups, reducing KV-cache overhead without sacrificing quality |
| **MoA** (Mixture of Agents) | Routes each token through a subset of the 10 agent FFNs via a learned top-k gate, blending expert outputs |

### How MoA Works

For each token representation **x**:

1. The gating network computes scores `g(x) = softmax(W_g · x)`.
2. The top-k agents are selected.
3. Each selected agent `i` produces an output `FFN_i(x)`.
4. The final output is the weighted sum: `Σ (g_i · FFN_i(x))`.

This allows the model to dynamically compose expertise — e.g., the Planner and Writer may dominate on scaffolding, while the Optimizer and Debugger activate on performance-critical loops.

---

## Configuration Options

ALM-1-Coder ships in three sizes to balance speed and quality:

| Config | Parameters | Layers | Heads | Agent FFN Dim | Use Case |
|--------|-----------|--------|-------|---------------|----------|
| **small** | ~14 M | 6 | 6 | 256 | Rapid prototyping, in-browser demo |
| **medium** | ~85 M | 12 | 12 | 512 | Development & fine-tuning |
| **full** | ~350 M | 24 | 16 | 1024 | Production-quality code generation |

Set the desired configuration in `config.py` before training, or pass `--config small|medium|full` to `train.py`.

---

## License

MIT

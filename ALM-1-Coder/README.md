# ALM-1-Coder — Mixture of Agents Coding Model

ALM-1-Coder is a Mixture of Agents (MoA) language model purpose-built for code generation. Rather than relying on a single monolithic network, it routes tokens through **10 named, role-defined agent sub-networks**, each with a personality, a job title, and dedicated parameters. A learned gating mechanism selects which agents contribute to each token, producing ensemble-quality output at a fraction of the cost.

---

## The 10 Agents & Their Roles

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
| 9 | **THOUGHT** | Project Manager | Diplomatic, peacemaker, exhausted | Coordinates agent-to-agent dialogue |
| 10 | **SEARCH** | Research Analyst | Helpful, always has a reference, "according to docs..." | Looks up documentation and references |

### Dynamic Agent Composition

When no single agent fits a task, the ENGLISH agent creates a **temporary new agent** by blending existing ones:

```
Prompt: "Optimize this recursive function and add error handling"

ENGLISH composes: 40% LOGIC + 40% DEBUGGER + 20% SYNTAX
→ Temporary agent handles the novel combination
```

This means the model is not limited to 10 fixed agents — it can theoretically create unlimited agent combinations.

---

## View Thoughts

The **View Thoughts** feature exposes the internal reasoning of the model. When enabled, you can watch the agents collaborate (and argue) in real-time:

- Which agents were activated for each token
- The confidence weight each agent contributed
- Agent dialogue: watch SYNTAX and ARCHITECT disagree about abstraction levels, see DEBUGGER catch edge cases LOGIC missed
- A step-by-step trace of how agents collaborated across layers

This makes ALM-1-Coder uniquely interpretable — you can *see* the model think, and it is both entertaining and useful.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **RoPE** (Rotary Position Embeddings) | Encodes relative positional information directly into attention, enabling better length generalization |
| **SwiGLU** | Gated linear activation (SiLU + GLU) for richer feature representation in feed-forward layers |
| **GQA** (Grouped-Query Attention) | Shares key/value heads across query groups, reducing KV-cache overhead without sacrificing quality |
| **MoA** (Mixture of Agents) | Routes each token through a subset of the 10 agent FFNs via a learned top-k gate, blending expert outputs |
| **Agent Role Embeddings** | Each agent has a learned identity vector that modulates expert outputs |
| **AgentTalk** | Agents communicate with each other between layers, debating and refining the output |

### How MoA Works

For each token representation **x**:

1. The agent gate computes scores: `g(x) = softmax(W_g · x)`
2. Top-k agents are selected (default k=3)
3. Each selected agent produces output through its expert FFNs
4. Agent role embeddings inject identity signals: `agent_output = expert_output * (1 + 0.1 * role_emb)`
5. The final output is the weighted sum across active agents

This allows the model to dynamically compose expertise — e.g., SYNTAX and ARCHITECT may dominate on scaffolding, while LOGIC and DEBUGGER activate on performance-critical loops.

---

## Use in Browser (Pyodide + IndexedDB)

ALM-1-Coder ships a fully client-side web interface powered by **Pyodide** — no server required. Model weights and the Pyodide runtime are **cached in IndexedDB** so you only download them once:

1. Open the deployed GitHub Pages URL (or serve `web/index.html` locally)
2. On first visit, model weights and packages are downloaded and stored in IndexedDB
3. On subsequent visits, everything loads instantly from IndexedDB — no re-downloading
4. Type your prompt and press **Generate** — inference runs entirely in your browser

---

## Configuration Options

ALM-1-Coder ships in three sizes to balance speed and quality:

| Config | Parameters | Layers | Heads | Embedding Dim | Agent FFN Dim | Use Case |
|--------|-----------|--------|-------|---------------|---------------|----------|
| **small** | ~10M | 4 | 8 | 128 | 352 | Rapid prototyping, in-browser demo |
| **medium** | ~150M | 16 | 12 | 512 | 1408 | Development & fine-tuning |
| **full** | ~1.3B | 24 | 16 | 2048 | 5632 | Production-quality code generation |

---

## License

MIT

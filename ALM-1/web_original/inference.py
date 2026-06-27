"""
ALM Inference Engine - Pure NumPy Implementation
Runs in Pyodide (browser) without PyTorch dependency.
"""

import numpy as np
import json

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias

def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class ALMInference:
    """Pure NumPy ALM inference for browser deployment."""

    def __init__(self, weights_path='alm_weights.npz', config_path='alm_config.json'):
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.vocab_size = config['vocab_size']
        self.block_size = config['block_size']
        self.n_experts = config['n_experts']
        self.head_dim = self.n_embd // self.n_head

        # Load weights
        print("Loading ALM weights...")
        data = np.load(weights_path)
        self.w = {key: data[key] for key in data.files}
        print(f"Loaded {len(self.w)} tensors")

    def forward(self, idx):
        """Full forward pass. idx: [1, T] token ids. Returns logits [1, T, vocab_size]."""
        T = idx.shape[1]

        # Token + position embeddings
        tok_emb = self.w['transformer.wte.weight'][idx]  # [1, T, n_embd]
        pos = np.arange(T)
        pos_emb = self.w['transformer.wpe.weight'][pos]  # [T, n_embd]
        x = tok_emb + pos_emb  # [1, T, n_embd]

        # Transformer blocks
        for i in range(self.n_layer):
            x = self._block(x, i)

        # Final layer norm
        x = layer_norm(x, self.w['transformer.ln_f.weight'], self.w['transformer.ln_f.bias'])

        # LM head (weight tied with wte)
        logits = x @ self.w['transformer.wte.weight'].T  # [1, T, vocab_size]
        return logits

    def _block(self, x, i):
        """Transformer block i."""
        # Self-attention
        x = x + self._attention(layer_norm(x,
            self.w[f'transformer.h.{i}.ln_1.weight'],
            self.w[f'transformer.h.{i}.ln_1.bias']), i)
        # MoE FFN
        x = x + self._moe(layer_norm(x,
            self.w[f'transformer.h.{i}.ln_2.weight'],
            self.w[f'transformer.h.{i}.ln_2.bias']), i)
        return x

    def _attention(self, x, i):
        """Multi-head causal self-attention."""
        B, T, C = x.shape

        # QKV projection
        c_attn_w = self.w[f'transformer.h.{i}.attn.c_attn.weight']
        c_attn_b = self.w[f'transformer.h.{i}.attn.c_attn.bias']
        qkv = x @ c_attn_w.T + c_attn_b  # [1, T, 3*n_embd]

        q, k, v = np.split(qkv, 3, axis=-1)

        # Reshape for multi-head
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Causal mask
        mask = np.tril(np.ones((T, T))).reshape(1, 1, T, T)
        att = np.where(mask == 1, att, -1e9)
        att = softmax(att, axis=-1)

        # Attention output
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        c_proj_w = self.w[f'transformer.h.{i}.attn.c_proj.weight']
        c_proj_b = self.w[f'transformer.h.{i}.attn.c_proj.bias']
        return out @ c_proj_w.T + c_proj_b

    def _moe(self, x, i):
        """Mixture of Experts layer."""
        B, T, C = x.shape

        # Gate
        gate_w = self.w[f'transformer.h.{i}.mlp.gate.weight']
        gate_b = self.w[f'transformer.h.{i}.mlp.gate.bias']
        gate_logits = x @ gate_w.T + gate_b  # [1, T, n_experts]
        gate_weights = softmax(gate_logits, axis=-1)  # [1, T, n_experts]

        # Expert outputs
        expert_outs = []
        for e in range(self.n_experts):
            expert_out = self._mlp(x, i, e)  # [1, T, C]
            expert_outs.append(expert_out)

        # Stack: [1, T, C, n_experts]
        all_expert_outs = np.stack(expert_outs, axis=-1)

        # Weighted sum: gate_weights [1, T, n_experts] -> [1, T, 1, n_experts]
        out = (all_expert_outs * gate_weights[..., np.newaxis, :]).sum(axis=-1)
        return out

    def _mlp(self, x, i, e):
        """Single expert MLP."""
        c_fc_w = self.w[f'transformer.h.{i}.mlp.experts.{e}.c_fc.weight']
        c_fc_b = self.w[f'transformer.h.{i}.mlp.experts.{e}.c_fc.bias']
        c_proj_w = self.w[f'transformer.h.{i}.mlp.experts.{e}.c_proj.weight']
        c_proj_b = self.w[f'transformer.h.{i}.mlp.experts.{e}.c_proj.bias']

        h = gelu(x @ c_fc_w.T + c_fc_b)
        return h @ c_proj_w.T + c_proj_b

    def generate(self, tokens, max_new_tokens=80, temperature=0.7, top_k=40,
                 repetition_penalty=1.3):
        """Generate tokens autoregressively."""
        idx = np.array(tokens).reshape(1, -1)

        for step in range(max_new_tokens):
            # Crop to block_size
            if idx.shape[1] > self.block_size:
                idx_cond = idx[:, -self.block_size:]
            else:
                idx_cond = idx

            logits = self.forward(idx_cond)[:, -1, :]  # [1, vocab_size]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tid in idx[0].tolist()[-50:]:
                    if 0 <= tid < logits.shape[-1]:
                        if logits[0, tid] > 0:
                            logits[0, tid] /= repetition_penalty
                        else:
                            logits[0, tid] *= repetition_penalty

            # Top-k
            if top_k > 0:
                top_k_val = min(top_k, logits.shape[-1])
                top_vals = np.partition(logits[0], -top_k_val)[-top_k_val:]
                threshold = np.min(top_vals)
                logits[0][logits[0] < threshold] = -1e9

            # Sample
            probs = softmax(logits, axis=-1)[0]
            next_token = np.random.choice(len(probs), p=probs)
            idx = np.concatenate([idx, [[next_token]]], axis=1)

        return idx[0].tolist()


# Test when run directly (not in Pyodide)
if __name__ == "__main__":
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model = ALMInference()

    prompts = ["Hello!", "What is Python?", "What can you do?"]
    for p in prompts:
        tokens = enc.encode(p)
        output = model.generate(tokens, max_new_tokens=60)
        print(f"Q: {p}")
        print(f"A: {enc.decode(output[len(tokens):])[:200]}")
        print()

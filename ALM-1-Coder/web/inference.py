"""
ALM-1-Coder Pure NumPy Inference Engine
========================================
Mixture of Agents (MoA) architecture inference for Pyodide / browser.
Zero PyTorch dependency — only NumPy is required.

Weight layout (PyTorch naming convention):
  wte.weight                                          (vocab_size, n_embd)
  h.{i}.ln_1.weight / .bias                           (n_embd,)
  h.{i}.attn.wq.weight                                (n_head * head_dim, n_embd)
  h.{i}.attn.wk.weight                                (n_kv_heads * head_dim, n_embd)
  h.{i}.attn.wv.weight                                (n_kv_heads * head_dim, n_embd)
  h.{i}.attn.wo.weight                                (n_embd, n_head * head_dim)
  h.{i}.ln_2.weight / .bias                           (n_embd,)
  h.{i}.moa.agent_gate.weight                         (n_agents, n_embd)
  h.{i}.moa.expert_gate.weight                        (n_experts, n_embd)
  h.{i}.moa.agent_role_embeddings.weight              (n_agents, n_embd)
  h.{i}.moa.experts.{e}.w1.weight                     (ffn_dim, n_embd)
  h.{i}.moa.experts.{e}.w2.weight                     (n_embd, ffn_dim)
  h.{i}.moa.experts.{e}.w3.weight                     (ffn_dim, n_embd)
  ln_f.weight / .bias                                 (n_embd,)
"""

import json
import numpy as np


# ---------------------------------------------------------------------------
# Agent names for View Thoughts
# ---------------------------------------------------------------------------

AGENT_NAMES = [
    "ENGLISH", "SYNTAX", "LOGIC", "DEBUGGER", "ARCHITECT",
    "HTML", "TRANSLATOR", "REASONING", "THOUGHT", "SEARCH",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically-stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """Standard Layer Normalization over the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return weight * x_norm + bias


def silu(x):
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * sigmoid(x)


def sigmoid(x):
    """Numerically-stable sigmoid."""
    pos = x >= 0
    z = np.zeros_like(x, dtype=np.float64)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_neg = np.exp(x[~pos])
    z[~pos] = exp_neg / (1.0 + exp_neg)
    return z


def rotate_half(x):
    """Split the last dimension in half, swap, and negate the first half.

    For x = [x1, x2] along the last axis, returns [-x2, x1].
    This is the "rotate_half" primitive used in RoPE.
    """
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary(q, k, cos, sin):
    """Apply Rotary Position Embeddings to query and key tensors.

    Args:
        q: shape (..., n_head, head_dim)
        k: shape (..., n_kv_heads, head_dim)
        cos: shape (1, 1, seq_len, head_dim)  — broadcastable
        sin: shape (1, 1, seq_len, head_dim)

    Returns:
        q_embed, k_embed with rotary embeddings applied.
    """
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def linear(x, weight):
    """Linear projection without bias: x @ weight^T.

    Args:
        x:      (..., in_features)
        weight: (out_features, in_features)

    Returns:
        (..., out_features)
    """
    return x @ weight.T


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class ALMCoderInference:
    """Pure-NumPy inference engine for ALM-1-Coder (MoA architecture).

    Designed to run inside Pyodide (browser) with no PyTorch dependency.
    Weights are loaded from a single .npz file and config from a JSON file.
    """

    # Default config values — overridden by whatever is in config_path
    _DEFAULTS = dict(
        vocab_size=32000,
        n_layer=12,
        n_head=32,
        n_embd=4096,
        block_size=2048,
        n_agents=4,
        n_experts_per_agent=4,
        top_k_agents=2,
        n_kv_heads=8,
        ffn_dim=11008,
        dropout=0.0,
        rope_theta=10000.0,
    )

    def __init__(self, weights_path="alm_coder_weights.npz",
                 config_path="alm_coder_config.json"):
        # Load configuration
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            cfg = {}

        # Merge with defaults
        for k, v in self._DEFAULTS.items():
            setattr(self, k, cfg.get(k, v))

        # Derived dimensions
        self.head_dim = self.n_embd // self.n_head
        self.n_experts = self.n_experts_per_agent * self.n_agents  # total experts

        # Load weights (lazy .npz — access keys on demand)
        try:
            self.w = dict(np.load(weights_path, allow_pickle=False))
        except (FileNotFoundError, OSError, EOFError, ValueError, Exception):
            # Allow instantiation without weights for testing
            self.w = {}

        # Pre-compute RoPE cos/sin cache
        self._build_rope_cache()

    # -----------------------------------------------------------------------
    # RoPE cache
    # -----------------------------------------------------------------------

    def _build_rope_cache(self):
        """Pre-compute cos and sin tables for Rotary Position Embeddings.

        Uses the standard formula:
            inv_freq = 1 / (theta^(2i / d))   for i in [0, d/2)
            For each position t:
                freq = t * inv_freq
                cos(t, 2i)   = cos(freq_i)
                cos(t, 2i+1) = cos(freq_i)   (duplicated)
                sin(t, 2i)   = sin(freq_i)
                sin(t, 2i+1) = sin(freq_i)   (duplicated)
        """
        theta = self.rope_theta
        head_dim = self.head_dim
        max_len = self.block_size

        # inv_freq = 1 / (theta^(2i / d))  for i in [0, d/2)
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        # positions
        t = np.arange(max_len, dtype=np.float32)
        # outer product: (max_len, head_dim/2)
        freqs = np.outer(t, inv_freq)
        # Duplicate to full head_dim: (max_len, head_dim)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self._cos_cached = np.cos(emb)[np.newaxis, np.newaxis, :, :]  # (1,1,block_size,head_dim)
        self._sin_cached = np.sin(emb)[np.newaxis, np.newaxis, :, :]

    def _rope_slice(self, seq_len, offset=0):
        """Return cos/sin for positions [offset, offset+seq_len)."""
        cos = self._cos_cached[:, :, offset:offset + seq_len, :]
        sin = self._sin_cached[:, :, offset:offset + seq_len, :]
        return cos, sin

    # -----------------------------------------------------------------------
    # Weight accessors
    # -----------------------------------------------------------------------

    def _wt(self, key):
        """Retrieve a weight tensor by dotted key name."""
        return self.w[key]

    # -----------------------------------------------------------------------
    # Single-layer forward passes
    # -----------------------------------------------------------------------

    def _attention(self, x, layer_idx, offset=0):
        """Grouped-Query Attention with RoPE and causal masking.

        Args:
            x:         (seq_len, n_embd)
            layer_idx:  integer layer index
            offset:     position offset (for KV-cache, future use)

        Returns:
            (seq_len, n_embd)
        """
        seq_len, _ = x.shape
        head_dim = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = n_head // n_kv  # repetitions for GQA

        i = layer_idx

        # Project Q, K, V  — all (seq_len, proj_dim)
        wq = self._wt(f"h.{i}.attn.wq.weight")
        wk = self._wt(f"h.{i}.attn.wk.weight")
        wv = self._wt(f"h.{i}.attn.wv.weight")

        q = linear(x, wq)  # (seq_len, n_head * head_dim)
        k = linear(x, wk)  # (seq_len, n_kv * head_dim)
        v = linear(x, wv)  # (seq_len, n_kv * head_dim)

        # Reshape to (seq_len, n_heads, head_dim)
        q = q.reshape(seq_len, n_head, head_dim)
        k = k.reshape(seq_len, n_kv, head_dim)
        v = v.reshape(seq_len, n_kv, head_dim)

        # Apply RoPE to Q and K
        cos, sin = self._rope_slice(seq_len, offset)
        # Transpose for broadcasting: (1, n_heads, seq_len, head_dim)
        q_t = q.transpose(1, 0, 2)[np.newaxis, :, :, :]
        k_t = k.transpose(1, 0, 2)[np.newaxis, :, :, :]
        v_t = v.transpose(1, 0, 2)[np.newaxis, :, :, :]

        q_t, k_t = apply_rotary(q_t, k_t, cos, sin)

        # GQA: repeat K, V heads to match Q heads
        if n_rep > 1:
            # k_t: (1, n_kv, seq_len, head_dim) -> (1, n_head, seq_len, head_dim)
            k_t = np.repeat(k_t, n_rep, axis=1)
            v_t = np.repeat(v_t, n_rep, axis=1)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(head_dim)
        # (1, n_head, seq_len, head_dim) @ (1, n_head, head_dim, seq_len)
        attn_scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale  # (1, n_head, seq_len, seq_len)

        # Causal mask: positions j > i cannot attend
        causal_mask = np.triu(
            np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1
        )
        attn_scores = attn_scores + causal_mask[np.newaxis, np.newaxis, :, :]

        attn_weights = softmax(attn_scores, axis=-1)  # (1, n_head, seq_len, seq_len)

        # Weighted sum
        attn_out = attn_weights @ v_t  # (1, n_head, seq_len, head_dim)

        # Merge heads: (seq_len, n_head * head_dim)
        attn_out = attn_out[0].transpose(1, 0, 2).reshape(seq_len, n_head * head_dim)

        # Output projection
        wo = self._wt(f"h.{i}.attn.wo.weight")
        out = linear(attn_out, wo)  # (seq_len, n_embd)
        return out

    def _swiglu_ffn(self, x, w1, w2, w3):
        """SwiGLU feed-forward network: w2(silu(w1(x)) * w3(x))."""
        gate = silu(linear(x, w1))   # (..., ffn_dim)
        up = linear(x, w3)           # (..., ffn_dim)
        return linear(gate * up, w2)  # (..., n_embd)

    def _moa_layer(self, x, layer_idx, thought_log=None):
        """Mixture of Agents (MoA) layer.

        1. Route input to top-k agents via agent_gate (softmax -> top-k).
        2. Compute expert_gate weights for all experts (softmax).
        3. Run each expert SwiGLU FFN on the input.
        4. Combine expert outputs weighted by expert_gate.
        5. For each top-k agent, modulate the expert output by the agent's
           role embedding and accumulate a weighted sum:
               agent_output = expert_output * (1.0 + 0.1 * agent_role_embedding)
               moa_out += agent_weight * agent_output

        Args:
            x:          (seq_len, n_embd)
            layer_idx:  integer layer index
            thought_log: optional list to append routing diagnostics

        Returns:
            (seq_len, n_embd)
        """
        i = layer_idx
        seq_len, n_embd = x.shape

        # --- Agent routing ---
        agent_gate_w = self._wt(f"h.{i}.moa.agent_gate.weight")  # (n_agents, n_embd)
        agent_logits = linear(x, agent_gate_w)  # (seq_len, n_agents)
        agent_probs = softmax(agent_logits, axis=-1)  # (seq_len, n_agents)

        # Top-k agent selection: keep only top_k_agents, zero out the rest
        top_k = self.top_k_agents
        if top_k < self.n_agents:
            sorted_idx = np.argsort(-agent_probs, axis=-1)
            mask = np.zeros_like(agent_probs, dtype=bool)
            for pos in range(seq_len):
                mask[pos, sorted_idx[pos, :top_k]] = True
            agent_probs = agent_probs * mask
            # Re-normalize
            agent_probs = agent_probs / (np.sum(agent_probs, axis=-1, keepdims=True) + 1e-9)

        # --- Expert gating ---
        expert_gate_w = self._wt(f"h.{i}.moa.expert_gate.weight")  # (n_experts, n_embd)
        expert_logits = linear(x, expert_gate_w)  # (seq_len, n_experts)
        expert_weights = softmax(expert_logits, axis=-1)  # (seq_len, n_experts)

        # --- Expert FFN outputs ---
        expert_outputs = []
        for e in range(self.n_experts):
            w1 = self._wt(f"h.{i}.moa.experts.{e}.w1.weight")
            w2 = self._wt(f"h.{i}.moa.experts.{e}.w2.weight")
            w3 = self._wt(f"h.{i}.moa.experts.{e}.w3.weight")
            out_e = self._swiglu_ffn(x, w1, w2, w3)  # (seq_len, n_embd)
            expert_outputs.append(out_e)
        expert_stack = np.stack(expert_outputs, axis=1)  # (seq_len, n_experts, n_embd)

        # Weighted combination of experts: (seq_len, n_embd)
        expert_combined = np.sum(
            expert_stack * expert_weights[:, :, np.newaxis], axis=1
        )

        # --- Agent role modulation ---
        # For each top-k agent:
        #   agent_output = expert_combined * (1.0 + 0.1 * agent_role_embedding[agent_idx])
        #   moa_out += agent_weight * agent_output
        role_emb = self._wt(f"h.{i}.moa.agent_role_embeddings.weight")  # (n_agents, n_embd)

        moa_out = np.zeros_like(x)  # (seq_len, n_embd)

        # Identify the top-k agents per position
        # For efficiency, we iterate over the (small) set of top-k agents
        if top_k < self.n_agents:
            sorted_idx = np.argsort(-agent_probs, axis=-1)  # (seq_len, n_agents)
            # We process each position; for short sequences this is fine
            for pos in range(seq_len):
                for k_idx in range(top_k):
                    agent_idx = int(sorted_idx[pos, k_idx])
                    agent_weight = agent_probs[pos, agent_idx]  # scalar
                    agent_emb = role_emb[agent_idx]  # (n_embd,)
                    # agent_output = expert_output * (1.0 + 0.1 * agent_emb)
                    agent_output = expert_combined[pos] * (1.0 + 0.1 * agent_emb)
                    moa_out[pos] += agent_weight * agent_output
        else:
            # All agents are active — vectorized
            for agent_idx in range(self.n_agents):
                agent_weight = agent_probs[:, agent_idx]  # (seq_len,)
                agent_emb = role_emb[agent_idx]  # (n_embd,)
                # (seq_len, n_embd) * (1.0 + 0.1 * (n_embd,)) -> (seq_len, n_embd)
                agent_output = expert_combined * (1.0 + 0.1 * agent_emb)
                # (seq_len,) -> (seq_len, 1) for broadcasting
                moa_out += agent_weight[:, np.newaxis] * agent_output

        # --- Thought logging ---
        if thought_log is not None:
            # Log the last token position's routing info
            last_agent_probs = agent_probs[-1]  # (n_agents,)
            last_expert_weights = expert_weights[-1]  # (n_experts,)
            active_agents = sorted(
                range(self.n_agents),
                key=lambda a: -last_agent_probs[a]
            )[:top_k]
            thought_log.append({
                "layer": i,
                "agent_probs": last_agent_probs.tolist(),
                "active_agents": active_agents,
                "expert_weights": last_expert_weights.tolist(),
                "top_agent": int(active_agents[0]),
                "top_agent_weight": float(last_agent_probs[active_agents[0]]),
            })

        return moa_out

    def _transformer_block(self, x, layer_idx, thought_log=None):
        """Single transformer block: LN -> Attn -> residual -> LN -> MoA -> residual."""
        i = layer_idx

        # Pre-norm attention
        ln1_w = self._wt(f"h.{i}.ln_1.weight")
        ln1_b = self._wt(f"h.{i}.ln_1.bias")
        x_norm = layer_norm(x, ln1_w, ln1_b)
        attn_out = self._attention(x_norm, i)
        x = x + attn_out

        # Pre-norm MoA (FFN replacement)
        ln2_w = self._wt(f"h.{i}.ln_2.weight")
        ln2_b = self._wt(f"h.{i}.ln_2.bias")
        x_norm = layer_norm(x, ln2_w, ln2_b)
        moa_out = self._moa_layer(x_norm, i, thought_log)
        x = x + moa_out

        return x

    # -----------------------------------------------------------------------
    # Full forward pass
    # -----------------------------------------------------------------------

    def forward(self, idx):
        """Forward pass through the full model.

        Args:
            idx: token indices, shape (seq_len,) or (1, seq_len)

        Returns:
            logits: shape (seq_len, vocab_size)
        """
        if idx.ndim == 1:
            idx = idx[np.newaxis, :]  # (1, seq_len)
        seq_len = idx.shape[1]

        # Token embeddings (no positional embedding — RoPE handles positions)
        wte = self._wt("wte.weight")  # (vocab_size, n_embd)
        x = wte[idx]  # (1, seq_len, n_embd)
        x = x[0]  # (seq_len, n_embd) — squeeze batch dim for simplicity

        # Transformer blocks
        for i in range(self.n_layer):
            x = self._transformer_block(x, i)

        # Final layer norm
        ln_f_w = self._wt("ln_f.weight")
        ln_f_b = self._wt("ln_f.bias")
        x = layer_norm(x, ln_f_w, ln_f_b)

        # Project to vocabulary (tie weights with embedding table)
        logits = x @ wte.T  # (seq_len, vocab_size)

        return logits

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(logits, tokens, penalty):
        """Apply repetition penalty to already-seen tokens.

        Divides logits of seen tokens by penalty if positive,
        multiplies if negative.
        """
        if penalty == 1.0:
            return logits
        seen = set(tokens)
        for tok in seen:
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty
        return logits

    @staticmethod
    def _top_k_filter(logits, k):
        """Zero out logits not in the top-k."""
        if k <= 0 or k >= logits.shape[0]:
            return logits
        indices_to_remove = logits < np.sort(logits)[-k]
        logits[indices_to_remove] = -np.inf
        return logits

    @staticmethod
    def _top_p_filter(logits, p):
        """Nucleus (top-p) filtering."""
        sorted_indices = np.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(softmax(sorted_logits))
        # Remove tokens with cumulative probability above the threshold
        # (keep at least one token)
        sorted_indices_to_remove = cumulative_probs > p
        # Shift right so the first token above threshold is kept
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -np.inf
        return logits

    def _sample_token(self, logits, generated_tokens, temperature, top_k,
                      top_p, repetition_penalty):
        """Sample a single token from logits with various filtering strategies."""
        logits = logits.copy()

        # Repetition penalty
        logits = self._apply_repetition_penalty(logits, generated_tokens,
                                                repetition_penalty)

        # Temperature
        if temperature > 0.0:
            logits = logits / temperature
        else:
            # Greedy
            return int(np.argmax(logits))

        # Top-k filtering
        logits = self._top_k_filter(logits, top_k)

        # Top-p (nucleus) filtering
        logits = self._top_p_filter(logits, top_p)

        # Sample from the (renormalized) distribution
        probs = softmax(logits)
        token = int(np.random.choice(len(probs), p=probs))
        return token

    def generate(self, tokens, max_new_tokens=80, temperature=0.7,
                 top_k=40, top_p=0.9, repetition_penalty=1.3,
                 return_thoughts=False):
        """Autoregressive generation with MoA thought tracking.

        Args:
            tokens: list or 1-D array of initial token ids
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering parameter
            top_p: nucleus (top-p) filtering parameter
            repetition_penalty: penalty factor for repeating tokens
            return_thoughts: if True, also return per-step thought logs

        Returns:
            If return_thoughts is False:
                list of generated token ids (including the prompt)
            If return_thoughts is True:
                tuple (generated_tokens, thought_logs)
                where thought_logs is a list of dicts, one per generated step,
                each containing "step", "token", and "layers" keys.
        """
        tokens = list(tokens)
        all_thoughts = [] if return_thoughts else None

        for step in range(max_new_tokens):
            # Truncate to block_size if needed
            context = tokens[-self.block_size:]
            idx = np.array(context, dtype=np.int64)

            # Forward pass — collect per-layer thought logs if requested
            step_thoughts = [] if return_thoughts else None
            if idx.ndim == 1:
                idx = idx[np.newaxis, :]
            seq_len = idx.shape[1]

            # Token embedding
            wte = self._wt("wte.weight")
            x = wte[idx]  # (1, seq_len, n_embd)
            x = x[0]      # (seq_len, n_embd)

            # Transformer blocks
            for i in range(self.n_layer):
                x = self._transformer_block(x, i, step_thoughts)

            # Final layer norm
            ln_f_w = self._wt("ln_f.weight")
            ln_f_b = self._wt("ln_f.bias")
            x = layer_norm(x, ln_f_w, ln_f_b)

            # Logits for the last position only
            logits = (x[-1] @ wte.T)  # (vocab_size,)

            # Sample
            next_token = self._sample_token(
                logits, tokens, temperature, top_k, top_p, repetition_penalty
            )
            tokens.append(next_token)

            # Record thought log for this step
            if return_thoughts:
                all_thoughts.append({
                    "step": step,
                    "token": next_token,
                    "layers": step_thoughts,
                })

        if return_thoughts:
            return tokens, all_thoughts
        return tokens

    # -----------------------------------------------------------------------
    # View Thoughts — human-readable summary
    # -----------------------------------------------------------------------

    def view_thoughts(self, thought_logs, agent_names=None):
        """Produce a human-readable summary of MoA routing during generation.

        Args:
            thought_logs: list of dicts returned by generate(return_thoughts=True)
            agent_names: optional list of agent name strings

        Returns:
            A formatted string summarising each step.
        """
        if agent_names is None:
            agent_names = AGENT_NAMES[:self.n_agents]
            # Pad if more agents than names
            while len(agent_names) < self.n_agents:
                agent_names.append(f"Agent-{len(agent_names)}")

        lines = []
        lines.append("=" * 70)
        lines.append("  ALM-1-Coder  ·  Mixture of Agents Thought Trace")
        lines.append("=" * 70)

        for entry in thought_logs:
            step = entry["step"]
            tok = entry["token"]
            layers = entry["layers"]
            lines.append(f"\n--- Step {step}  |  token={tok} ---")
            for l_entry in layers:
                li = l_entry["layer"]
                active = l_entry["active_agents"]
                top_a = l_entry["top_agent"]
                top_w = l_entry["top_agent_weight"]
                a_probs = l_entry["agent_probs"]
                e_weights = l_entry["expert_weights"]
                agent_strs = []
                for a in active:
                    name = agent_names[a] if a < len(agent_names) else f"Agent-{a}"
                    agent_strs.append(f"{name}({a_probs[a]:.3f})")
                lines.append(
                    f"  Layer {li:2d}: "
                    f"top-agent={agent_names[top_a] if top_a < len(agent_names) else top_a} "
                    f"({top_w:.3f})  "
                    f"active=[{', '.join(agent_strs)}]  "
                    f"experts={e_weights}"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: load model
# ---------------------------------------------------------------------------

def load_model(weights_path="alm_coder_weights.npz",
               config_path="alm_coder_config.json"):
    """Instantiate and return an ALMCoderInference object."""
    return ALMCoderInference(weights_path=weights_path,
                             config_path=config_path)

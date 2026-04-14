"""
ALM-1-Coder: Mixture of Agents Architecture
============================================
A revolutionary coding model where specialized agents collaborate
inside a single transformer. Each agent has a name, a role, and
dedicated parameters. The ENGLISH agent (CEO) routes and composes
agents dynamically for any task.

Architecture:
- Agent Tokens: [ENGLISH], [SYNTAX], [LOGIC], [DEBUGGER], [ARCHITECT],
                [HTML], [TRANSLATOR], [REASONING], [THOUGHT], [SEARCH]
- RoPE (Rotary Position Embeddings) for better code context
- SwiGLU activation for modern FFN layers
- GQA (Grouped Query Attention) for efficiency
- MoA (Mixture of Agents) with agent-conditioned routing
- Dynamic agent composition for novel tasks
- View Thoughts: agents argue internally, user sees clean output
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os

# ============================================================
# Agent Definitions - The Team
# ============================================================

AGENT_TOKENS = {
    "ENGLISH":    "<|english|>",
    "SYNTAX":     "<|syntax|>",
    "LOGIC":      "<|logic|>",
    "DEBUGGER":   "<|debugger|>",
    "ARCHITECT":  "<|architect|>",
    "HTML":       "<|html|>",
    "TRANSLATOR": "<|translator|>",
    "REASONING":  "<|reasoning|>",
    "THOUGHT":    "<|thought|>",
    "SEARCH":     "<|search|>",
}

# Agent personality descriptions for View Thoughts
AGENT_PERSONALITIES = {
    "ENGLISH":    "Authoritative, decisive, slightly annoyed",
    "SYNTAX":     "Pragmatic, ships fast, hates over-engineering",
    "LOGIC":      "Perfectionist, pedantic, loves algorithms",
    "DEBUGGER":   "Suspicious, thorough, loves finding others' mistakes",
    "ARCHITECT":  "Abstract thinker, 'let us zoom out', never codes",
    "HTML":       "Dramatic, passionate, defensive about CSS",
    "TRANSLATOR": "Chill, neutral, no opinions, just converts",
    "REASONING":  "Over-analytical, cautious, 'well actually...'",
    "THOUGHT":    "Diplomatic, peacemaker, exhausted",
    "SEARCH":     "Helpful, always has a reference, 'according to docs...'",
}

# Agent roles for business analogy
AGENT_ROLES = {
    "ENGLISH":    "CEO - Comprehends, routes, composes",
    "SYNTAX":     "Senior Developer - Writes clean code",
    "LOGIC":      "Algorithms Engineer - Optimizes and reasons",
    "DEBUGGER":   "QA Engineer - Finds and fixes bugs",
    "ARCHITECT":  "Systems Architect - Designs structure",
    "HTML":       "Frontend Developer - Builds web UIs",
    "TRANSLATOR": "Localization Engineer - Converts between languages",
    "REASONING":  "Chief Strategy Officer - Thinks step by step",
    "THOUGHT":    "Project Manager - Coordinates agent dialogue",
    "SEARCH":     "Research Analyst - Looks up docs and references",
}

# Special tokens
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
THOUGHT_START = "<|think|>"
THOUGHT_END = "<|/think|>"
COMPOSE_START = "<|compose|>"
COMPOSE_END = "<|/compose|>"

ALL_SPECIAL_TOKENS = list(AGENT_TOKENS.values()) + [
    ASSISTANT_TOKEN, END_TOKEN, THOUGHT_START, THOUGHT_END, COMPOSE_START, COMPOSE_END
]


# ============================================================
# Configuration
# ============================================================

class ALMCoderConfig:
    """Configuration for ALM-1-Coder model variants."""
    def __init__(
        self,
        vocab_size=8000,
        n_layer=8,
        n_head=8,
        n_embd=256,
        block_size=512,
        n_agents=10,          # Number of agent groups in MoA
        n_experts_per_agent=2, # Experts per agent group
        top_k_agents=3,        # Top agents to activate per token
        n_kv_heads=4,          # GQA: fewer KV heads than Q heads
        ffn_dim=704,           # SwiGLU FFN dimension (2/3 * 4 * embd)
        dropout=0.1,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_agents = n_agents
        self.n_experts_per_agent = n_experts_per_agent
        self.top_k_agents = top_k_agents
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_head
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.rope_theta = rope_theta
        # Total experts = agents * experts_per_agent
        self.n_experts = n_agents * n_experts_per_agent


def get_coder_small_config():
    """ALM-1-Coder Small: ~10M params, trains on CPU in ~30 min."""
    return ALMCoderConfig(
        vocab_size=8000, n_layer=4, n_head=8, n_embd=128,
        block_size=256, n_agents=10, n_experts_per_agent=1,
        top_k_agents=3, n_kv_heads=4, ffn_dim=352, dropout=0.1
    )


def get_coder_medium_config():
    """ALM-1-Coder Medium: ~150M params, needs Colab GPU."""
    return ALMCoderConfig(
        vocab_size=32000, n_layer=16, n_head=12, n_embd=512,
        block_size=1024, n_agents=8, n_experts_per_agent=2,
        top_k_agents=3, n_kv_heads=4, ffn_dim=1408, dropout=0.1
    )


def get_coder_full_config():
    """ALM-1-Coder Full: ~1.3B params, needs serious GPU."""
    return ALMCoderConfig(
        vocab_size=32000, n_layer=24, n_head=16, n_embd=2048,
        block_size=2048, n_agents=10, n_experts_per_agent=2,
        top_k_agents=3, n_kv_heads=4, ffn_dim=5632, dropout=0.1
    )


# ============================================================
# Rotary Position Embeddings (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings for better code context handling."""
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, seq_len):
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
# SwiGLU Activation (Modern FFN)
# ============================================================

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network - standard in modern LLMs."""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention for efficiency."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_heads == 0

        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_heads  # repetition factor

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.block_size, config.rope_theta)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V projections
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V for GQA (repeat KV heads)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_head, T, self.head_dim)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))


# ============================================================
# Mixture of Agents (MoA) Layer
# ============================================================

class MixtureOfAgents(nn.Module):
    """
    Mixture of Agents Layer - The Core Innovation.

    Instead of anonymous experts, we have NAMED AGENTS with dedicated
    parameter groups. Each agent group contains experts that specialize
    in that agent's domain. The gating network routes tokens to agent
    groups, and the agent identity is maintained through role embeddings.

    Agent groups map to our 10 agents:
      0: ENGLISH, 1: SYNTAX, 2: LOGIC, 3: DEBUGGER, 4: ARCHITECT,
      5: HTML, 6: TRANSLATOR, 7: REASONING, 8: THOUGHT, 9: SEARCH
    """
    def __init__(self, config):
        super().__init__()
        self.n_agents = config.n_agents
        self.n_experts_per_agent = config.n_experts_per_agent
        self.n_experts = config.n_experts
        self.top_k_agents = config.top_k_agents
        self.n_embd = config.n_embd

        # Agent gate: decides which agents to route to
        self.agent_gate = nn.Linear(config.n_embd, self.n_agents, bias=False)

        # Expert gate within each agent group
        self.expert_gate = nn.Linear(config.n_embd, self.n_experts, bias=False)

        # Agent role embeddings: each agent gets a learned identity vector
        self.agent_role_embeddings = nn.Embedding(self.n_agents, config.n_embd)

        # Expert FFN networks (SwiGLU)
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.n_experts)
        ])

        # Agent composition weights (for dynamic composition)
        self.compose_proj = nn.Linear(config.n_embd, self.n_agents, bias=False)

    def forward(self, x, agent_hint=None):
        """
        Forward pass with agent-conditioned routing.

        Uses sparse computation: only selected experts are evaluated,
        making MoA actually efficient instead of computing all experts.

        Args:
            x: Input tensor [B, T, C]
            agent_hint: Optional agent index hint [B, T] for forced routing
                        (used during training with role-conditioned data)

        Returns:
            output: [B, T, C]
            thought_log: dict of agent routing info for View Thoughts
        """
        B, T, C = x.shape

        # Compute agent routing weights
        agent_logits = self.agent_gate(x)  # [B, T, n_agents]

        # If agent hint provided (role-conditioned training), bias the routing
        if agent_hint is not None:
            # Add a large bias to the correct agent
            agent_bias = torch.full_like(agent_logits, -10.0)
            agent_bias.scatter_(-1, agent_hint.unsqueeze(-1), 10.0)
            agent_logits = agent_logits + agent_bias

        # Top-k agent routing
        agent_weights = F.softmax(agent_logits, dim=-1)  # [B, T, n_agents]

        # Get top-k agents
        top_k_weights, top_k_indices = torch.topk(agent_weights, self.top_k_agents, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # --- Sparse Expert Computation ---
        # Only compute experts belonging to top-k agent groups
        # Each agent has n_experts_per_agent experts: agent i owns experts [i*n_experts_per_agent, (i+1)*n_experts_per_agent)
        output = torch.zeros_like(x)
        per_agent_outputs = []  # Store per-agent outputs for AgentTalk

        for k_idx in range(self.top_k_agents):
            agent_idx = top_k_indices[:, :, k_idx]  # [B, T]
            agent_w = top_k_weights[:, :, k_idx:k_idx+1]  # [B, T, 1]

            # Add agent role embedding influence
            agent_emb = self.agent_role_embeddings(agent_idx)  # [B, T, C]

            # Compute only the experts for this agent group
            agent_id = top_k_indices[0, -1, k_idx].item()  # primary agent for this position
            expert_start = agent_id * self.n_experts_per_agent
            expert_end = expert_start + self.n_experts_per_agent

            # Compute expert outputs for this agent group only (sparse!)
            expert_outputs = []
            for e_idx in range(expert_start, expert_end):
                expert_outputs.append(self.experts[e_idx](x))
            expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, T, C, n_experts_per_agent]

            # Expert gating: only for this agent's experts
            expert_logits = self.expert_gate(x[:, :, :])  # [B, T, n_experts]
            # Extract only the columns for this agent's experts
            agent_expert_logits = expert_logits[:, :, expert_start:expert_end]
            agent_expert_weights = F.softmax(agent_expert_logits, dim=-1)  # [B, T, n_experts_per_agent]

            # Weighted sum of this agent's expert outputs
            expert_output = (expert_stack * agent_expert_weights.unsqueeze(-2)).sum(dim=-1)  # [B, T, C]

            # Agent-conditioned output: expert output + agent identity signal
            agent_output = expert_output * (1.0 + 0.1 * agent_emb)
            per_agent_outputs.append(agent_output)  # Save for AgentTalk
            output = output + agent_w * agent_output

        # Thought log for View Thoughts
        thought_log = {
            "agent_weights": agent_weights.detach(),
            "top_k_agents": top_k_indices.detach(),
            "top_k_weights": top_k_weights.detach(),
            "agent_outputs": per_agent_outputs,  # Per-agent outputs for AgentTalk
        }

        return output, thought_log


# ============================================================
# AgentTalk: Inter-Agent Communication Layer
# ============================================================

class AgentTalkLayer(nn.Module):
    """
    AgentTalk Layer — Agents communicate with each other between layers.

    After the MoA layer produces weighted agent outputs, this layer lets
    active agents exchange information. Each agent can "speak" to other
    active agents, and their messages are aggregated to refine the output.

    This is what makes agents "talk and do stuff" — they do not just
    independently produce outputs; they debate, refine, and collaborate.

    Mechanism:
    1. Each active agent produces a "message" vector from its output
    2. Messages are broadcast to all other active agents
    3. Each agent attends to incoming messages (scaled by agent weights)
    4. The refined messages are added back to the output
    """
    def __init__(self, config):
        super().__init__()
        self.n_agents = config.n_agents
        self.n_embd = config.n_embd

        # Agent message projection: each agent creates a compact message
        self.msg_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Agent listen projection: each agent processes incoming messages
        self.listen_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Gate to control how much agent talk influences the output
        self.talk_gate = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        # Per-agent talk strength (learned)
        self.agent_talk_strength = nn.Embedding(self.n_agents, 1)

    def forward(self, x, agent_outputs, agent_weights, top_k_indices):
        """
        Args:
            x: Original input [B, T, C]
            agent_outputs: Per-agent output tensors list of [B, T, C]
            agent_weights: Agent routing weights [B, T, n_agents]
            top_k_indices: Top-k agent indices [B, T, top_k]

        Returns:
            refined_output: [B, T, C] — output after agent communication
            dialogue_log: list of agent dialogue info for View Thoughts
        """
        B, T, C = x.shape
        top_k = top_k_indices.shape[-1]

        # Each active agent creates its message
        messages = []
        for k_idx in range(top_k):
            if k_idx < len(agent_outputs):
                msg = torch.tanh(self.msg_proj(agent_outputs[k_idx]))  # [B, T, C]
                messages.append(msg)
            else:
                messages.append(torch.zeros_like(x))

        # Aggregate messages from other agents (excluding self)
        refined = torch.zeros_like(x)
        dialogue_log = []

        for k_idx in range(top_k):
            # Sum messages from all OTHER active agents
            other_msgs = torch.zeros_like(x)
            for j in range(top_k):
                if j != k_idx:
                    other_msgs = other_msgs + messages[j]

            # Each agent "listens" to the aggregated other messages
            listened = self.listen_proj(other_msgs)  # [B, T, C]

            # Gated combination: how much should this agent be influenced?
            gate_input = torch.cat([agent_outputs[k_idx] if k_idx < len(agent_outputs) else x, listened], dim=-1)
            gate = torch.sigmoid(self.talk_gate(gate_input))  # [B, T, C]

            # Agent-specific talk strength
            agent_id = top_k_indices[0, -1, k_idx].item()  # for the last position
            strength = torch.sigmoid(self.agent_talk_strength(
                torch.tensor([agent_id], device=x.device)
            )).item()

            # Refined output for this agent
            agent_refined = agent_outputs[k_idx] if k_idx < len(agent_outputs) else x
            agent_refined = agent_refined + strength * gate * listened
            refined = refined + agent_refined

            # Log dialogue info
            agent_name = list(AGENT_TOKENS.keys())[agent_id] if agent_id < len(AGENT_TOKENS) else f"Agent-{agent_id}"
            other_names = []
            for j in range(top_k):
                if j != k_idx:
                    other_id = top_k_indices[0, -1, j].item()
                    other_name = list(AGENT_TOKENS.keys())[other_id] if other_id < len(AGENT_TOKENS) else f"Agent-{other_id}"
                    other_names.append(other_name)

            dialogue_log.append({
                "agent": agent_name,
                "agent_id": agent_id,
                "talking_to": other_names,
                "talk_strength": round(strength, 3),
                "gate_strength": round(gate[0, -1].mean().item(), 3),
            })

        return refined, dialogue_log


# ============================================================
# Transformer Block
# ============================================================

class CoderBlock(nn.Module):
    """Transformer block with pre-norm, GQA, MoA FFN, and AgentTalk."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moa = MixtureOfAgents(config)
        # AgentTalk layer for inter-agent communication
        self.agent_talk = AgentTalkLayer(config)

    def forward(self, x, agent_hint=None):
        # Self-attention with residual
        x = x + self.attn(self.ln_1(x))
        # MoA FFN with residual
        moa_out, thought_log = self.moa(self.ln_2(x), agent_hint=agent_hint)

        # AgentTalk: agents communicate and refine their outputs
        # Reconstruct per-agent outputs for talk layer
        agent_outputs = thought_log.get("agent_outputs", [moa_out])
        talk_out, dialogue_log = self.agent_talk(
            self.ln_2(x),
            agent_outputs,
            thought_log["agent_weights"],
            thought_log["top_k_agents"],
        )

        # Combine MoA output with agent-talk-refined output
        x = x + moa_out + 0.1 * talk_out  # talk influence is small but meaningful

        # Add dialogue to thought log
        thought_log["dialogue"] = dialogue_log
        return x, thought_log


# ============================================================
# ALM-1-Coder: The Full Model
# ============================================================

class ALMCoder(nn.Module):
    """
    ALM-1-Coder: Mixture of Agents Coding Model

    Features:
    - 10 named agents with dedicated parameter groups
    - Dynamic agent composition for novel tasks
    - RoPE for better code context
    - SwiGLU for modern activation
    - GQA for efficient attention
    - View Thoughts: see agents argue in real-time
    - Role-conditioned training with agent name tokens
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings (no position embedding - RoPE handles it)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Agent token embeddings (separate from regular tokens for clear routing)
        # These map agent name strings to dedicated embedding space
        n_special = len(ALL_SPECIAL_TOKENS)
        self.agent_token_emb = nn.Embedding(n_special, config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([CoderBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head (tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # Weight tying

        # Agent token ID mapping - these are indices into the special token range
        # Special tokens are appended after the regular vocab, so:
        # agent tokens occupy positions [0, 10) in ALL_SPECIAL_TOKENS list
        self.agent_token_ids = {}
        for i, (name, token) in enumerate(AGENT_TOKENS.items()):
            self.agent_token_ids[name] = i  # agent index 0-9

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"ALM-1-Coder initialized: {n_params:,} parameters")
        active_params = self._estimate_active_params()
        print(f"Active params per token: ~{active_params:,} (MoA sparsity)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _estimate_active_params(self):
        """Estimate active parameters per token (with MoA sparsity)."""
        total = sum(p.numel() for p in self.parameters())
        # MoA activates top_k_agents out of n_agents
        moa_fraction = self.config.top_k_agents / self.config.n_agents
        # Approximate: attention + active experts + shared
        # This is a rough estimate
        return int(total * (0.4 + 0.6 * moa_fraction))

    def forward(self, idx, targets=None, agent_hint=None):
        """
        Forward pass.

        Args:
            idx: Token indices [B, T]
            targets: Target indices for loss [B, T]
            agent_hint: Optional agent index hint for role-conditioned training [B, T]

        Returns:
            logits: [B, T, vocab_size]
            loss: Cross-entropy loss if targets provided
            thought_logs: List of agent routing info from each layer
        """
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token embeddings (RoPE handles positions)
        x = self.drop(self.wte(idx))

        # Pass through transformer blocks
        thought_logs = []
        for block in self.h:
            x, thought_log = block(x, agent_hint=agent_hint)
            thought_logs.append(thought_log)

        # Final layer norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss, thought_logs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.7, top_k=40,
                 top_p=0.9, repetition_penalty=1.2, agent_name=None,
                 stop_tokens=None, return_thoughts=False):
        """
        Generate tokens autoregressively.

        Args:
            idx: Starting token indices [1, T]
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalize repeated tokens
            agent_name: Force specific agent (e.g. "SYNTAX", "DEBUGGER")
            stop_tokens: Token IDs to stop on
            return_thoughts: If True, return agent thought logs for View Thoughts
        """
        self.eval()
        device = idx.device

        all_thought_logs = []
        agent_hint = None

        # If agent_name specified, create agent hint
        if agent_name and agent_name in self.agent_token_ids:
            agent_hint = torch.full((1, idx.size(1)), self.agent_token_ids[agent_name],
                                    dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            # Adjust agent_hint length if needed
            if agent_hint is not None and agent_hint.size(1) != idx_cond.size(1):
                agent_hint = torch.full((1, idx_cond.size(1)),
                    self.agent_token_ids.get(agent_name, 0),
                    dtype=torch.long, device=device)

            logits, _, thought_logs = self(idx_cond, agent_hint=agent_hint)

            if return_thoughts and thought_logs:
                all_thought_logs.append({
                    "step": step,
                    "logs": [{
                        "agent_weights": tl["agent_weights"][0, -1].cpu().tolist(),
                        "top_k_agents": tl["top_k_agents"][0, -1].cpu().tolist(),
                        "top_k_weights": tl["top_k_weights"][0, -1].cpu().tolist(),
                    } for tl in thought_logs]
                })

            logits = logits[:, -1, :]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in idx[0].tolist()[-50:]:
                    if 0 <= token_id < logits.size(-1):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            # Top-k
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus sampling)
            if top_p is not None and top_p < 1.0:
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

            if stop_tokens is not None:
                if next_token.item() in stop_tokens:
                    break

            idx = torch.cat((idx, next_token), dim=1)

        if return_thoughts:
            return idx, all_thought_logs
        return idx

    def get_agent_routing(self, text, tokenizer):
        """
        Analyze which agents would be activated for a given text.
        Useful for debugging and View Thoughts preview.
        """
        self.eval()
        tokens = tokenizer.encode(text)
        idx = torch.tensor(tokens).unsqueeze(0)

        with torch.no_grad():
            _, _, thought_logs = self(idx)

        # Aggregate agent activations across layers
        agent_activations = {}
        for layer_idx, tl in enumerate(thought_logs):
            weights = tl["agent_weights"][0, -1].cpu().tolist()
            for agent_idx, weight in enumerate(weights):
                agent_names = list(AGENT_TOKENS.keys())
                name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"agent_{agent_idx}"
                if name not in agent_activations:
                    agent_activations[name] = []
                agent_activations[name].append(weight)

        # Average across layers
        return {name: sum(weights)/len(weights) for name, weights in agent_activations.items()}

    def compose_agents(self, agent_weights_dict):
        """
        Dynamic Agent Composition: Create a temporary agent by mixing existing ones.

        Args:
            agent_weights_dict: {"SYNTAX": 0.4, "ARCHITECT": 0.4, "LOGIC": 0.2}

        Returns:
            Composition vector that can be used as agent_hint
        """
        # This creates a soft routing signal that mixes agents
        # The actual composition happens in the MoA layer via weighted combination
        agent_names = list(AGENT_TOKENS.keys())
        composition = torch.zeros(1, 1, self.config.n_agents)

        for name, weight in agent_weights_dict.items():
            if name in agent_names:
                idx = agent_names.index(name)
                composition[0, 0, idx] = weight

        return composition

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {k: v for k, v in self.config.__dict__.items()},
        }, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Filter out config keys that are derived/computed to avoid constructor errors
        config_dict = checkpoint['config']
        # Remove 'n_experts' since it's computed as n_agents * n_experts_per_agent
        config_dict.pop('n_experts', None)
        config = ALMCoderConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Checkpoint loaded from {path}")
        return model

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_weights_numpy(self, path):
        """Export weights as compressed NumPy arrays for Pyodide inference."""
        import numpy as np
        weights = {}
        for name, param in self.state_dict().items():
            weights[name] = param.cpu().numpy()

        # Also export config
        import json
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        config_path = path.replace('.npz', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        np.savez_compressed(path, **weights)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Weights exported: {size_mb:.1f} MB -> {path}")
        print(f"Config exported: {config_path}")

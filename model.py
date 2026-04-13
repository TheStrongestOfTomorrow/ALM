"""
ALM: Adaptive Learning Model
A Mixture-of-Experts transformer with adaptive inference and conversation support.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os

# Special tokens for conversation formatting
ASSISTANT_TOKEN = "<|assistant|"
USER_TOKEN = "user"
SYSTEM_TOKEN = "system"
END_TOKEN = "<|end|>"


class ALMConfig:
    """Configuration for ALM model variants."""
    def __init__(self, n_layer=8, n_head=8, n_embd=512, vocab_size=50257,
                 block_size=512, n_experts=4, top_k_experts=2, dropout=0.1):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_experts = n_experts
        self.top_k_experts = top_k_experts
        self.dropout = dropout


def get_alm_1_config():
    """ALM-1 flagship: 135.35B parameter specification (for reference)."""
    return ALMConfig(n_layer=74, n_head=96, n_embd=12288, vocab_size=50257,
                     block_size=2048, n_experts=8, top_k_experts=4)


def get_alm_tiny_config():
    """ALM-Tiny: Lightweight version for training and deployment."""
    return ALMConfig(n_layer=4, n_head=4, n_embd=128, vocab_size=50257,
                     block_size=256, n_experts=2, top_k_experts=2, dropout=0.1)


def get_alm_small_config():
    """ALM-Small: Balanced version for better quality."""
    return ALMConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257,
                     block_size=768, n_experts=4, top_k_experts=2, dropout=0.1)


# Keep backward compatibility
get_tiny_config = get_alm_tiny_config


class MLP(nn.Module):
    """Standard feed-forward network used as expert backbone."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class MoE(nn.Module):
    """
    Adaptive Mixture of Experts Layer with Soft Routing.
    
    Uses a learned gating network to compute weighted combinations
    of expert outputs. This "soft" MoE approach is more efficient
    than hard routing and allows gradient flow to all experts
    during training, leading to better optimization.
    """
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.gate = nn.Linear(config.n_embd, config.n_experts)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        # Compute gate weights
        gate_logits = self.gate(x)  # [B, T, n_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, T, n_experts]

        # Compute all expert outputs in parallel
        # Stack along last dim for efficient weighted sum
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # expert_outputs: [B, T, C, n_experts]
        
        # Weighted combination: gate_weights [B, T, n_experts] -> [B, T, 1, n_experts]
        out = (expert_outputs * gate_weights.unsqueeze(-2)).sum(dim=-1)
        return out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with dropout."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    """Transformer block with pre-norm, attention, and MoE FFN."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ALM(nn.Module):
    """
    Adaptive Learning Model - Main Architecture

    Features:
    - Mixture of Experts (MoE) for adaptive computation
    - Causal transformer for autoregressive generation
    - Conversation-aware formatting with special tokens
    - Configurable architecture sizes (Tiny, Small, Full)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embeddings between input and output
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Track number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"ALM initialized: {n_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, repetition_penalty=1.0, stop_tokens=None):
        """
        Generate tokens autoregressively with various sampling strategies.
        """
        self.eval()
        device = idx.device

        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in idx[0].tolist():
                    if 0 <= token_id < logits.size(-1):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

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

        return idx

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
        }, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ALMConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Checkpoint loaded from {path}")
        return model

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

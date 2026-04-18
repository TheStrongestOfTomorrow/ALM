#!/usr/bin/env python3
"""
ALM-1-Coder Training Script
============================
Mixture of Agents coding model training with role-conditioned agent hints.

Features:
- tiktoken (GPT-2) tokenization with special token injection
- Role-conditioned training: agent_hint derived from training data
- Cosine LR schedule with warmup
- Gradient clipping, weight decay, AdamW
- Checkpoint save/resume with best-loss tracking
- NumPy weight export for Pyodide inference
- Post-training generation test with coding prompts
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

import tiktoken

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    ALMCoder,
    ALMCoderConfig,
    get_coder_small_config,
    ALL_SPECIAL_TOKENS,
    AGENT_TOKENS,
    ASSISTANT_TOKEN,
)
from data.train import get_training_data

# ============================================================
# Tokenizer & Special-Token Setup
# ============================================================

SPECIAL_TOKEN_START = 7990  # tiktoken IDs >= this get truncated

# Special token string -> integer ID  (7990, 7991, …)
SPECIAL_TOKEN_IDS: dict[str, int] = {}
for _i, _tok in enumerate(ALL_SPECIAL_TOKENS):
    SPECIAL_TOKEN_IDS[_tok] = SPECIAL_TOKEN_START + _i

# Effective vocabulary size: must accommodate all special tokens
VOCAB_SIZE = SPECIAL_TOKEN_START + len(ALL_SPECIAL_TOKENS)

# Agent token string -> agent index (0-9, ordered as in AGENT_TOKENS)
AGENT_TOKEN_STR_TO_IDX: dict[str, int] = {}
for _idx, (_name, _token_str) in enumerate(AGENT_TOKENS.items()):
    AGENT_TOKEN_STR_TO_IDX[_token_str] = _idx

# Regex that matches any special token (longest first to avoid partial matches)
_SPECIAL_TOKEN_RE = re.compile(
    "(" + "|".join(re.escape(t) for t in sorted(SPECIAL_TOKEN_IDS, key=len, reverse=True)) + ")"
)


def setup_tokenizer():
    """Return a tiktoken GPT-2 encoder."""
    return tiktoken.get_encoding("gpt2")


def encode_text(text: str, tokenizer, block_size: int) -> list[int]:
    """
    Encode *text* into a list of integer token IDs.

    1. Split on special-token boundaries so that agent/control tokens are
       recognised as single units.
    2. Encode non-special chunks with tiktoken (GPT-2).
    3. Truncate tiktoken IDs >= SPECIAL_TOKEN_START to fit the reduced
       vocabulary.
    4. Truncate the full sequence to *block_size*.
    """
    parts = _SPECIAL_TOKEN_RE.split(text)
    token_ids: list[int] = []

    for part in parts:
        if not part:
            continue
        if part in SPECIAL_TOKEN_IDS:
            token_ids.append(SPECIAL_TOKEN_IDS[part])
        else:
            for tid in tokenizer.encode(part):
                # Collapse any tiktoken ID that falls in the special-token range
                if tid >= SPECIAL_TOKEN_START:
                    tid = tid % SPECIAL_TOKEN_START
                token_ids.append(tid)

    return token_ids[:block_size]


def decode_tokens(token_ids: list[int], tokenizer) -> str:
    """Best-effort decode of a mixed (tiktoken + special) token list."""
    # Build reverse mapping for special tokens
    id_to_special = {v: k for k, v in SPECIAL_TOKEN_IDS.items()}

    parts: list[str] = []
    buffer: list[int] = []  # accumulate consecutive regular tiktoken IDs

    for tid in token_ids:
        if tid in id_to_special:
            # Flush any buffered regular tokens first
            if buffer:
                try:
                    parts.append(tokenizer.decode(buffer))
                except Exception:
                    parts.append("<?>")
                buffer = []
            parts.append(id_to_special[tid])
        else:
            buffer.append(tid)

    if buffer:
        try:
            parts.append(tokenizer.decode(buffer))
        except Exception:
            parts.append("<?>")

    return "".join(parts)


# ============================================================
# Data Preparation
# ============================================================

def prepare_example(example, tokenizer, block_size, n_agents=5):
    """
    Turn a raw training example into (input_ids, target_ids, agent_hint).

    *agent_hint* is derived from the first agent token found in the text
    (e.g. ``<|syntax|>`` → index 1).  Clamped to [0, n_agents-1] so that
    it stays within the model's agent_gate dimension.
    """
    # Accept both plain strings and dicts with a "text" key
    text = example if isinstance(example, str) else example.get("text", str(example))

    # Derive agent_hint from the leading agent token
    agent_hint = 0  # default → ENGLISH (index 0)
    for token_str, idx in AGENT_TOKEN_STR_TO_IDX.items():
        if text.startswith(token_str):
            agent_hint = idx
            break

    # Clamp to valid agent range for the model config
    agent_hint = min(agent_hint, n_agents - 1)

    # Encode (request block_size+1 so we can split into input/target)
    token_ids = encode_text(text, tokenizer, block_size + 1)

    if len(token_ids) < 2:
        return None  # too short after tokenisation

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
        agent_hint,
    )


# ============================================================
# Learning-Rate Schedule
# ============================================================

def get_lr(epoch: int, epochs: int, warmup_epochs: int, base_lr: float, min_lr: float = 1e-5) -> float:
    """Cosine decay with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    if epoch >= epochs:
        return min_lr
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(path: str, model, config, epoch: int, loss: float, is_best: bool = False):
    """Save a training checkpoint with the required keys."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in config.__dict__.items()},
        "epoch": epoch,
        "loss": loss,
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path) or ".", "best_checkpoint.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(path: str, model, optimizer=None):
    """Resume from a saved checkpoint.  Returns (epoch, loss)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))


# ============================================================
# Training Loop
# ============================================================

def train(config, epochs, checkpoint_dir, resume=False, device="cpu"):
    """Main training loop — sequential, per-example (CPU-friendly)."""

    # Adjust vocab size to accommodate special tokens
    config.vocab_size = VOCAB_SIZE

    # ---- Model ----
    model = ALMCoder(config)
    model.to(device)

    # ---- Tokenizer ----
    tokenizer = setup_tokenizer()

    # ---- Data ----
    training_data = get_training_data()
    print(f"Loaded {len(training_data)} training examples")

    block_size = config.block_size

    examples = []
    for ex in training_data:
        prepared = prepare_example(ex, tokenizer, block_size, n_agents=config.n_agents)
        if prepared is not None:
            examples.append(prepared)
    print(f"Prepared {len(examples)} tokenised examples")

    # ---- Optimiser ----
    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # ---- Resume ----
    start_epoch = 0
    best_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

    if resume and os.path.exists(checkpoint_path):
        start_epoch, last_loss = load_checkpoint(checkpoint_path, model, optimizer)
        best_loss = last_loss
        print(f"Resumed from epoch {start_epoch}, loss {last_loss:.4f}")

    # ---- Print info ----
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  ALM-1-Coder Training")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}")
    print(f"  Vocab size  : {VOCAB_SIZE}")
    print(f"  Block size  : {block_size}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Special toks: {len(ALL_SPECIAL_TOKENS)}")
    print(f"  Agents      : {len(AGENT_TOKENS)}")
    print(f"{'='*60}\n")

    # ---- Epoch loop ----
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        lr = get_lr(epoch, epochs, warmup_epochs=10, base_lr=5e-4)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0
        num_examples = 0

        # Shuffle each epoch
        indices = torch.randperm(len(examples)).tolist()

        for idx in indices:
            input_ids, target_ids, agent_hint_val = examples[idx]

            input_ids = input_ids.unsqueeze(0).to(device)    # (1, T)
            target_ids = target_ids.unsqueeze(0).to(device)  # (1, T)

            # Build agent_hint tensor (same value for every position in seq)
            T = input_ids.size(1)
            agent_hint = torch.full(
                (1, T), agent_hint_val, dtype=torch.long, device=device
            )

            # Forward
            logits, loss, _thought_logs = model(
                input_ids, targets=target_ids, agent_hint=agent_hint
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_examples += 1

        avg_loss = total_loss / num_examples if num_examples > 0 else float("inf")

        # Best-loss tracking
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Save checkpoint (always latest; best separately)
        save_checkpoint(checkpoint_path, model, config, epoch, avg_loss, is_best=is_best)

        # Log every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == start_epoch or is_best:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:>4}/{epochs} | "
                f"Loss {avg_loss:.4f} | "
                f"Best {best_loss:.4f} | "
                f"LR {lr:.6f} | "
                f"Time {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s — best loss: {best_loss:.4f}")

    return model, config


# ============================================================
# Generation Test
# ============================================================

CODING_PROMPTS = [
    "def fibonacci(n):",
    "function sortByAge(",
    "class LinkedList:",
    "# Binary search in Python",
]


def generate_test(model, config, device="cpu"):
    """Quick generation test with four coding prompts."""
    tokenizer = setup_tokenizer()
    block_size = config.block_size

    model.eval()
    print(f"\n{'='*60}")
    print("  GENERATION TEST")
    print(f"{'='*60}")

    with torch.no_grad():
        for prompt in CODING_PROMPTS:
            tokens = encode_text(prompt, tokenizer, block_size)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=60,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                agent_name=None,
            )

            generated = decode_tokens(output_ids[0].tolist(), tokenizer)
            print(f"\n  Prompt : {prompt}")
            print(f"  Output : {generated}")

    print(f"{'='*60}")


# ============================================================
# NumPy Weight Export (for Pyodide)
# ============================================================

def export_weights(model, output_dir):
    """Export model weights to NumPy format for Pyodide inference."""
    export_dir = os.path.join(output_dir, "numpy_weights")
    os.makedirs(export_dir, exist_ok=True)

    npz_path = os.path.join(export_dir, "alm1_coder_weights.npz")
    model.export_weights_numpy(npz_path)

    # Also save individual .npy files for fine-grained loading
    count = 0
    for name, param in model.state_dict().items():
        safe_name = name.replace(".", "_")
        np.save(os.path.join(export_dir, f"{safe_name}.npy"), param.cpu().numpy())
        count += 1

    print(f"Exported {count} weight arrays to {export_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ALM-1-Coder")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu / cuda / mps)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip NumPy weight export")
    parser.add_argument("--no-gentest", action="store_true",
                        help="Skip post-training generation test")
    args = parser.parse_args()

    # ---- Device selection ----
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # ---- Config ----
    config = get_coder_small_config()

    # ---- Train ----
    model, config = train(
        config=config,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=device,
    )

    # ---- Generation test ----
    if not args.no_gentest:
        generate_test(model, config, device=device)

    # ---- Export weights ----
    if not args.no_export:
        export_weights(model, args.checkpoint_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()

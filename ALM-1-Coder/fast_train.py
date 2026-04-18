#!/usr/bin/env python3
"""Fast training script for ALM-1-Coder small config (~6.9M params).

Uses block_size=128, subset of examples, no dropout for speed.
Completes 50 epochs on CPU in reasonable time.
"""
import time, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from model import ALMCoder, ALMCoderConfig, get_coder_small_config
from train import (
    VOCAB_SIZE, setup_tokenizer, prepare_example,
    get_training_data, save_checkpoint
)

def main():
    # Config - small model ~6.9M params
    config = get_coder_small_config()
    config.vocab_size = VOCAB_SIZE
    config.dropout = 0.0  # disable dropout for speed

    model = ALMCoder(config)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, block_size={config.block_size}, "
          f"n_layer={config.n_layer}, n_agents={config.n_agents}")

    # Tokenizer & data
    tokenizer = setup_tokenizer()
    training_data = get_training_data()

    examples = []
    for ex in training_data:
        prepared = prepare_example(ex, tokenizer, config.block_size, n_agents=config.n_agents)
        if prepared is not None:
            examples.append(prepared)

    # Use subset for faster training (first 30 examples)
    MAX_EXAMPLES = 30
    if len(examples) > MAX_EXAMPLES:
        # Take diverse subset
        import random
        random.seed(42)
        examples = random.sample(examples, MAX_EXAMPLES)

    print(f"Training on {len(examples)} examples")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))

    EPOCHS = 50
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(EPOCHS):
        # Cosine LR with warmup
        warmup = 5
        if epoch < warmup:
            lr = 5e-4 * (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(EPOCHS - warmup, 1)
            lr = 1e-5 + 0.5 * (5e-4 - 1e-5) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        total_loss = 0.0
        indices = torch.randperm(len(examples)).tolist()

        for idx in indices:
            input_ids, target_ids, ah = examples[idx]
            input_ids = input_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
            T = input_ids.size(1)
            agent_hint = torch.full((1, T), ah, dtype=torch.long)

            logits, loss, _ = model(input_ids, targets=target_ids, agent_hint=agent_hint)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(examples)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        save_checkpoint(checkpoint_path, model, config, epoch, avg_loss, is_best=is_best)

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0 or is_best:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss {avg_loss:.4f} | "
                  f"Best {best_loss:.4f} | LR {lr:.6f} | Time {elapsed:.0f}s")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}min). Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

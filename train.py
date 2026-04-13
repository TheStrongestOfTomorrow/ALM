"""
ALM Training Script
Trains the ALM model on conversational data to make it talkable.
Uses per-example training for fast CPU training.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os
import sys
import time

from model import ALM, ALMConfig, get_alm_tiny_config, ASSISTANT_TOKEN, END_TOKEN
from data.train import get_training_data


def train():
    print("=" * 60)
    print("  ALM Training Pipeline")
    print("  Adaptive Learning Model - Making it Talkable!")
    print("=" * 60)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = get_alm_tiny_config()
    print(f"Config: {config.n_layer} layers, {config.n_head} heads, "
          f"{config.n_embd} emb, {config.n_experts} experts, {config.block_size} ctx")

    # Initialize model - resume from best checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'alm_tiny_best.pt')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = ALM.load_checkpoint(checkpoint_path, device=device)
    else:
        model = ALM(config).to(device)
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,}")

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load and prepare training data
    training_data = get_training_data()
    print(f"Training examples: {len(training_data)}")

    # Tokenize all data
    tokenized_data = []
    for text in training_data:
        tokens = enc.encode(text)
        if len(tokens) >= 4:
            # Truncate to block_size
            if len(tokens) > config.block_size:
                tokens = tokens[:config.block_size]
            tokenized_data.append(tokens)

    print(f"Tokenized examples: {len(tokenized_data)}")
    total_tokens = sum(len(t) for t in tokenized_data)
    print(f"Total training tokens: {total_tokens:,}")

    # Training setup - start with a fresh optimizer for continued training
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                                   betas=(0.9, 0.95))

    num_epochs = 150
    best_loss = float('inf')
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        # Shuffle data each epoch
        import random
        random.shuffle(tokenized_data)

        for tokens in tokenized_data:
            x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)

            _, loss = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Learning rate decay
        lr = 5e-4 * (0.95 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - start_time

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_checkpoint(os.path.join(checkpoint_dir, 'alm_tiny_best.pt'))

        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(os.path.join(checkpoint_dir, f'alm_tiny_epoch{epoch+1}.pt'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

    # Save final checkpoint
    model.save_checkpoint(os.path.join(checkpoint_dir, 'alm_tiny_final.pt'))

    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")

    # Quick test
    print("\n" + "=" * 60)
    print("  Quick Generation Test")
    print("=" * 60)
    model.eval()

    test_prompts = [
        "Hello! How are you?",
        "Write a Python function to reverse a string",
        "What is machine learning?",
        "What can you do?",
    ]

    for prompt in test_prompts:
        input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2
        )
        response = enc.decode(output_ids[0].tolist()[input_ids.size(1):])
        print(f"\nPrompt: {prompt}")
        print(f"ALM: {response[:200]}")

    return model


if __name__ == "__main__":
    train()

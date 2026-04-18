#!/usr/bin/env python3
"""
ALM-1-Coder Weight Quantization & Export
==========================================
Loads a trained checkpoint and exports:
1. alm1_coder_weights_q4.npz  — 4-bit min-max quantized (two 4-bit vals packed per uint8)
2. alm1_coder_weights_f16.npz — float16 (half precision) fallback
3. alm1_coder_config.json     — model configuration

4-bit quantization scheme:
  For each weight matrix W (float32):
    - Compute min_val, max_val per tensor
    - scale = (max_val - min_val) / 15.0
    - zero_point = min_val
    - quantized = round((W - zero_point) / scale)  → uint4 [0, 15]
    - Pack two consecutive uint4 values into one uint8 byte
  Stored per weight key:
    - "{key}_q4"   : uint8 packed array (half the elements of original)
    - "{key}_scale": float32 scalar
    - "{key}_zp"   : float32 scalar
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ALMCoder, ALMCoderConfig, get_coder_small_config
from train import VOCAB_SIZE


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    # Remove derived fields that aren't constructor args
    for key in ["n_experts"]:
        config_dict.pop(key, None)
    config = ALMCoderConfig(**config_dict)
    model = ALMCoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, Loss: {checkpoint.get('loss', '?'):.4f}")
    return model, config, checkpoint


def quantize_4bit(weight_np):
    """Quantize a float32 numpy array to 4-bit using min-max quantization.

    Returns:
        packed: uint8 array with two 4-bit values per byte
        scale: float32 scalar
        zero_point: float32 scalar
    """
    flat = weight_np.astype(np.float32).ravel()
    min_val = float(flat.min())
    max_val = float(flat.max())

    # Handle constant tensors
    if max_val - min_val < 1e-8:
        scale = np.float32(1.0)
        zero_point = np.float32(min_val)
        quantized = np.zeros(flat.shape, dtype=np.uint8)
    else:
        scale = np.float32((max_val - min_val) / 15.0)
        zero_point = np.float32(min_val)
        # Quantize: map to [0, 15]
        quantized = np.round((flat - zero_point) / scale).astype(np.uint8)
        quantized = np.clip(quantized, 0, 15)

    # Pack two 4-bit values into one uint8
    n = len(quantized)
    # Pad to even length if needed
    if n % 2 != 0:
        quantized = np.append(quantized, np.uint8(0))
        n += 1

    # Low nibble = first value, high nibble = second value
    low = quantized[0::2]         # even indices
    high = quantized[1::2]        # odd indices
    packed = (high.astype(np.uint8) << 4) | low.astype(np.uint8)

    return packed, scale, zero_point


def dequantize_4bit(packed, scale, zero_point, original_size):
    """Dequantize 4-bit packed data back to float32 (for verification)."""
    low = (packed & np.uint8(0x0F)).astype(np.float32)
    high = ((packed >> 4) & np.uint8(0x0F)).astype(np.float32)

    # Interleave
    result = np.empty(len(low) + len(high), dtype=np.float32)
    result[0::2] = low * scale + zero_point
    result[1::2] = high * scale + zero_point

    return result[:original_size]


def export_quantized(model, config, output_dir):
    """Export model weights in 4-bit quantized and float16 formats."""
    os.makedirs(output_dir, exist_ok=True)

    state_dict = model.state_dict()

    # ---- Float16 export ----
    f16_weights = {}
    for name, param in state_dict.items():
        f16_weights[name] = param.cpu().numpy().astype(np.float16)

    f16_path = os.path.join(output_dir, "alm1_coder_weights_f16.npz")
    np.savez_compressed(f16_path, **f16_weights)
    f16_size = os.path.getsize(f16_path)
    print(f"Float16 weights: {f16_size / (1024*1024):.2f} MB → {f16_path}")

    # ---- 4-bit quantized export ----
    q4_weights = {}
    for name, param in state_dict.items():
        weight_np = param.cpu().numpy()
        original_size = weight_np.size
        packed, scale, zero_point = quantize_4bit(weight_np)

        # Store with suffixed keys
        q4_weights[f"{name}_q4"] = packed
        q4_weights[f"{name}_scale"] = np.array([scale], dtype=np.float32)
        q4_weights[f"{name}_zp"] = np.array([zero_point], dtype=np.float32)

        # Verify dequantization
        deq = dequantize_4bit(packed, scale, zero_point, original_size)
        mse = np.mean((weight_np.ravel() - deq) ** 2)
        max_err = np.max(np.abs(weight_np.ravel() - deq))

    q4_path = os.path.join(output_dir, "alm1_coder_weights_q4.npz")
    np.savez_compressed(q4_path, **q4_weights)
    q4_size = os.path.getsize(q4_path)
    print(f"4-bit quantized: {q4_size / (1024*1024):.2f} MB → {q4_path}")

    # ---- Config JSON ----
    config_dict = {k: v for k, v in config.__dict__.items()}
    config_path = os.path.join(output_dir, "alm1_coder_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    config_size = os.path.getsize(config_path)
    print(f"Config JSON: {config_size} bytes → {config_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Export Summary")
    print(f"{'='*60}")
    print(f"  alm1_coder_weights_q4.npz  : {q4_size / (1024*1024):.2f} MB")
    print(f"  alm1_coder_weights_f16.npz  : {f16_size / (1024*1024):.2f} MB")
    print(f"  alm1_coder_config.json      : {config_size} bytes")
    print(f"  Compression ratio (q4/f16)  : {q4_size / f16_size:.2%}")
    print(f"  Compression ratio (q4/f32)  : {q4_size / (sum(p.numel() * 4 for p in state_dict.values()) / (1024*1024)):.2%}")

    # Quantization quality metrics
    print(f"\n  Quantization Quality (sample):")
    sample_keys = list(state_dict.keys())[:3]
    for name in sample_keys:
        weight_np = state_dict[name].cpu().numpy()
        packed = q4_weights[f"{name}_q4"]
        scale = q4_weights[f"{name}_scale"][0]
        zp = q4_weights[f"{name}_zp"][0]
        deq = dequantize_4bit(packed, scale, zp, weight_np.size)
        mse = np.mean((weight_np.ravel() - deq) ** 2)
        max_err = np.max(np.abs(weight_np.ravel() - deq))
        print(f"    {name}: MSE={mse:.6f}, MaxErr={max_err:.4f}")

    print(f"{'='*60}")

    return f16_path, q4_path, config_path


def main():
    checkpoint_path = os.path.join("checkpoints", "best_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join("checkpoints", "checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    model, config, checkpoint = load_checkpoint(checkpoint_path)

    output_dir = os.path.join("web")
    export_quantized(model, config, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TRM Model Export for ESP32-S3 Ternary Inference

Converts a PyTorch TRM checkpoint into a flat binary file that the
ESP32-S3 firmware can load from flash.

Binary format v4 (backward-compatible with v3 readers that skip unknown tail):
  Header (16 bytes):
    - magic:      uint32  0x54524D31 ("TRM1")
    - version:    uint32  4
    - vocab_size: uint32
    - num_layers: uint32  (L_LAYERS)

  Tensors (sequential):
    For each tensor:
      - size_bytes: uint32  (byte count of data blob)
      - scale:      float32 (quantization scale factor; 1.0 for raw float)
      - data:       uint8[] (raw bytes, length = size_bytes)

  Tensor order:
    1. embed_tokens          (INT8, shape [vocab_size, hidden_size])
    2. puzzle_emb            (float32, shape [PUZZLE_EMB_LEN, hidden_size])
    3. For each layer i in [0, L_LAYERS):
       a. qkv_proj.norm_weight   (float32, shape [hidden_size])
       b. qkv_proj               (2-bit packed ternary)
       c. o_proj.norm_weight     (float32, shape [hidden_size])
       d. o_proj                 (2-bit packed ternary)
       e. gate_up.norm_weight    (float32, shape [hidden_size])
       f. gate_up_proj           (2-bit packed ternary)
       g. down.norm_weight       (float32, shape [MLP_INTER])
       h. down_proj              (2-bit packed ternary)
    4. lm_head               (INT8, shape [vocab_size, hidden_size])
    5. H_init                (float32, shape [hidden_size])
    6. L_init                (float32, shape [hidden_size])
    7. q_head_weight         (float32, shape [2, hidden_size])   -- v4+
    8. q_head_bias           (float32, shape [2])                -- v4+

Weight packing (ESP32 simple format):
  4 ternary values per byte, 2 bits each (LSB first):
    0b00 = -1
    0b01 =  0
    0b10 = +1

Usage:
  python export_ternary.py \\
    --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/bitnet_sudoku/step_48828 \\
    --output esp32_trm/firmware/model_data/trm_ternary.bin \\
    --verify
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ============== Constants (must match model_config.h) ==============
HIDDEN_SIZE = 512
NUM_HEADS = 8
HEAD_DIM = 64
NUM_KV_HEADS = 8
MLP_INTER = 1536       # _find_multiple(round(4 * 512 * 2/3), 256)
QKV_OUT_SIZE = 1536     # (8 + 2*8) * 64
ATTN_OUT_SIZE = 512     # 8 * 64
GATE_UP_SIZE = 3072     # 2 * MLP_INTER
L_LAYERS = 2

MODEL_MAGIC = 0x54524D31
MODEL_VERSION = 4

PUZZLE_EMB_LEN = 16

WEIGHTS_PER_BYTE = 4


def _find_multiple(a, b):
    """Ceiling division helper matching the PyTorch model code."""
    return (-(a // -b)) * b


def quantize_to_ternary(weight: torch.Tensor):
    """
    Quantize a weight tensor to ternary {-1, 0, +1}.

    Mirrors the weight_quant() function from bitnet_linear.py:
      scale = 1.0 / mean(|w|)
      u = round(w * scale).clamp(-1, 1)

    Returns:
      ternary: int8 tensor with values in {-1, 0, 1}
      scale:   float scale factor (1.0 / mean(|w|))
    """
    abs_mean = weight.abs().mean().clamp(min=1e-5)
    scale = 1.0 / abs_mean.item()
    ternary = (weight * scale).round().clamp(-1, 1).to(torch.int8)
    return ternary, scale


def pack_ternary_esp32(ternary: torch.Tensor) -> np.ndarray:
    """
    Pack a ternary weight tensor into 2-bit format for ESP32.

    Mapping:
      -1 -> 0b00
       0 -> 0b01
      +1 -> 0b10

    4 values per byte, LSB first.

    Args:
      ternary: int8 tensor [M, K] with values in {-1, 0, 1}

    Returns:
      packed: uint8 ndarray [M, K//4]
    """
    t = ternary.numpy().astype(np.int8)
    M, K = t.shape
    assert K % 4 == 0, f"K must be divisible by 4, got {K}"

    # Map: -1 -> 0, 0 -> 1, +1 -> 2
    mapped = (t + 1).astype(np.uint8)  # {0, 1, 2}

    # Reshape and pack 4 per byte
    reshaped = mapped.reshape(M, K // 4, 4)
    packed = np.zeros((M, K // 4), dtype=np.uint8)
    for i in range(4):
        packed |= reshaped[:, :, i] << (i * 2)

    return packed


def quantize_to_int8(weight: torch.Tensor):
    """
    Quantize a full-precision weight tensor to INT8.

    Uses symmetric per-tensor quantization:
      scale = 127.0 / max(|w|)
      q = round(w * scale).clamp(-128, 127)

    Returns:
      quantized: int8 tensor
      scale:     float scale factor
    """
    abs_max = weight.abs().max().clamp(min=1e-5)
    scale = 127.0 / abs_max.item()
    quantized = (weight * scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def write_tensor(f, data: np.ndarray, scale: float):
    """Write a tensor blob to the binary file: [size_bytes, scale, data]."""
    data_bytes = data.tobytes()
    f.write(struct.pack('<I', len(data_bytes)))  # size
    f.write(struct.pack('<f', scale))             # scale
    f.write(data_bytes)                            # data


def extract_state_dict(checkpoint_path: str):
    """
    Load checkpoint and extract the model state dict.

    Handles both:
    1. Direct state_dict files
    2. Training checkpoint dicts with 'model_state_dict' key
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        # Training checkpoint — look for common keys
        for key in ['model_state_dict', 'model', 'state_dict', 'net']:
            if key in checkpoint:
                print(f"  Found state dict under key '{key}'")
                return checkpoint[key]
        # If the dict itself looks like a state dict (keys are param names)
        if any('weight' in k for k in checkpoint.keys()):
            print("  Checkpoint appears to be a raw state dict")
            return checkpoint
        # Show available keys for debugging
        print(f"  Available keys: {list(checkpoint.keys())}")
        raise ValueError("Cannot find model state dict in checkpoint")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")


def get_layer_key(state_dict, patterns):
    """Find a matching key in state dict from a list of patterns."""
    for pattern in patterns:
        if pattern in state_dict:
            return pattern
    # Try with 'inner.' prefix (the model wraps in ACT wrapper)
    for pattern in patterns:
        inner_pattern = f"inner.{pattern}"
        if inner_pattern in state_dict:
            return inner_pattern
    return None


def print_found_keys(state_dict):
    """Debug: print all keys with shapes."""
    print("\nAll state dict keys:")
    for key, value in sorted(state_dict.items()):
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {list(value.shape)} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    print()


def export_ternary(checkpoint_path: str, output_path: str, verbose: bool = True):
    """
    Main export function: convert checkpoint to ESP32 binary.
    """
    state_dict = extract_state_dict(checkpoint_path)

    if verbose:
        print_found_keys(state_dict)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Detect key prefix (might be 'inner.', empty, or 'model.')
    prefix = ""
    for key in state_dict.keys():
        if "embed_tokens" in key:
            prefix = key.split("embed_tokens")[0]
            break
    print(f"Detected key prefix: '{prefix}'")

    # ============== Extract and process each tensor ==============

    # 1. Embedding
    embed_key = f"{prefix}embed_tokens.embedding_weight"
    embed_weight = state_dict[embed_key].float()
    vocab_size = embed_weight.shape[0]
    print(f"Embedding: {list(embed_weight.shape)}, vocab_size={vocab_size}")

    embed_q, embed_scale = quantize_to_int8(embed_weight)
    print(f"  -> INT8, scale={embed_scale:.4f}")

    # 2. Puzzle embedding — pre-padded to [PUZZLE_EMB_LEN, HIDDEN_SIZE]
    puzzle_emb_key = f"{prefix}puzzle_emb.weights"
    if puzzle_emb_key in state_dict:
        puzzle_emb_raw = state_dict[puzzle_emb_key].float()  # [num_identifiers, puzzle_emb_ndim]
        print(f"Puzzle embedding raw: {list(puzzle_emb_raw.shape)}")
        # Pad to PUZZLE_EMB_LEN * HIDDEN_SIZE and reshape
        target_size = PUZZLE_EMB_LEN * HIDDEN_SIZE
        flat = puzzle_emb_raw.reshape(-1)
        if flat.shape[0] < target_size:
            flat = torch.nn.functional.pad(flat, (0, target_size - flat.shape[0]))
        puzzle_emb = flat[:target_size].reshape(PUZZLE_EMB_LEN, HIDDEN_SIZE)
    else:
        print("WARNING: No puzzle_emb found — using zeros")
        puzzle_emb = torch.zeros(PUZZLE_EMB_LEN, HIDDEN_SIZE)
    print(f"Puzzle embedding (padded): {list(puzzle_emb.shape)}")

    # 3. Transformer blocks (with norm_weight for each CastedBitLinear)
    layer_data = []
    for layer_idx in range(L_LAYERS):
        layer_prefix = f"{prefix}L_level.layers.{layer_idx}."
        layer_info = {}

        # QKV projection (CastedBitLinear)
        qkv_nw_key = f"{layer_prefix}self_attn.qkv_proj.norm_weight"
        qkv_nw = state_dict[qkv_nw_key].float()
        layer_info['qkv_norm_weight'] = qkv_nw
        print(f"Layer {layer_idx} qkv_proj.norm_weight: {list(qkv_nw.shape)}")

        qkv_key = f"{layer_prefix}self_attn.qkv_proj.weight"
        qkv_w = state_dict[qkv_key].float()
        print(f"Layer {layer_idx} qkv_proj: {list(qkv_w.shape)}")
        qkv_t, qkv_scale = quantize_to_ternary(qkv_w)
        qkv_packed = pack_ternary_esp32(qkv_t)
        layer_info['qkv'] = (qkv_packed, 1.0 / qkv_scale)
        print(f"  -> ternary packed: {qkv_packed.shape}, scale={1.0/qkv_scale:.6f}")

        # Output projection (CastedBitLinear)
        o_nw_key = f"{layer_prefix}self_attn.o_proj.norm_weight"
        o_nw = state_dict[o_nw_key].float()
        layer_info['o_norm_weight'] = o_nw
        print(f"Layer {layer_idx} o_proj.norm_weight: {list(o_nw.shape)}")

        o_key = f"{layer_prefix}self_attn.o_proj.weight"
        o_w = state_dict[o_key].float()
        print(f"Layer {layer_idx} o_proj: {list(o_w.shape)}")
        o_t, o_scale = quantize_to_ternary(o_w)
        o_packed = pack_ternary_esp32(o_t)
        layer_info['o'] = (o_packed, 1.0 / o_scale)
        print(f"  -> ternary packed: {o_packed.shape}, scale={1.0/o_scale:.6f}")

        # Gate+Up projection (CastedBitLinear)
        gate_up_nw_key = f"{layer_prefix}mlp.gate_up_proj.norm_weight"
        gate_up_nw = state_dict[gate_up_nw_key].float()
        layer_info['gate_up_norm_weight'] = gate_up_nw
        print(f"Layer {layer_idx} gate_up_proj.norm_weight: {list(gate_up_nw.shape)}")

        gate_up_key = f"{layer_prefix}mlp.gate_up_proj.weight"
        gate_up_w = state_dict[gate_up_key].float()
        print(f"Layer {layer_idx} gate_up_proj: {list(gate_up_w.shape)}")
        gate_up_t, gate_up_scale = quantize_to_ternary(gate_up_w)
        gate_up_packed = pack_ternary_esp32(gate_up_t)
        layer_info['gate_up'] = (gate_up_packed, 1.0 / gate_up_scale)
        print(f"  -> ternary packed: {gate_up_packed.shape}, scale={1.0/gate_up_scale:.6f}")

        # Down projection (CastedBitLinear)
        down_nw_key = f"{layer_prefix}mlp.down_proj.norm_weight"
        down_nw = state_dict[down_nw_key].float()
        layer_info['down_norm_weight'] = down_nw
        print(f"Layer {layer_idx} down_proj.norm_weight: {list(down_nw.shape)}")

        down_key = f"{layer_prefix}mlp.down_proj.weight"
        down_w = state_dict[down_key].float()
        print(f"Layer {layer_idx} down_proj: {list(down_w.shape)}")
        down_t, down_scale = quantize_to_ternary(down_w)
        down_packed = pack_ternary_esp32(down_t)
        layer_info['down'] = (down_packed, 1.0 / down_scale)
        print(f"  -> ternary packed: {down_packed.shape}, scale={1.0/down_scale:.6f}")

        layer_data.append(layer_info)

    # 4. LM head (CastedLinear — full precision → INT8)
    lm_head_key = f"{prefix}lm_head.weight"
    lm_head_w = state_dict[lm_head_key].float()
    print(f"LM head: {list(lm_head_w.shape)}")
    lm_head_q, lm_head_scale = quantize_to_int8(lm_head_w)
    print(f"  -> INT8, scale={lm_head_scale:.4f}")

    # 5. Initial carry states (H_init, L_init) — nn.Buffer [HIDDEN_SIZE]
    h_init_key = f"{prefix}H_init"
    l_init_key = f"{prefix}L_init"
    h_init = state_dict[h_init_key].float()
    l_init = state_dict[l_init_key].float()
    print(f"H_init: {list(h_init.shape)}")
    print(f"L_init: {list(l_init.shape)}")

    # 6. Q-head (halting head) — CastedLinear(hidden_size, 2, bias=True)
    q_head_w_key = f"{prefix}q_head.weight"
    q_head_b_key = f"{prefix}q_head.bias"
    q_head_weight = state_dict[q_head_w_key].float()  # [2, HIDDEN_SIZE]
    q_head_bias = state_dict[q_head_b_key].float()    # [2]
    print(f"q_head.weight: {list(q_head_weight.shape)}")
    print(f"q_head.bias:   {list(q_head_bias.shape)}")

    # ============== Write binary file ==============
    total_bytes = 0
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MODEL_MAGIC))
        f.write(struct.pack('<I', MODEL_VERSION))
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', L_LAYERS))
        total_bytes += 16

        # Embedding (INT8)
        write_tensor(f, embed_q.numpy(), embed_scale)
        total_bytes += 8 + embed_q.numpy().nbytes

        # Puzzle embedding (float32, pre-padded to [PUZZLE_EMB_LEN, HIDDEN_SIZE])
        puzzle_emb_np = puzzle_emb.numpy()
        write_tensor(f, puzzle_emb_np, 1.0)
        total_bytes += 8 + puzzle_emb_np.nbytes

        # Transformer blocks (norm_weight before each ternary projection)
        for layer_idx in range(L_LAYERS):
            ld = layer_data[layer_idx]
            for proj_name, nw_name in [
                ('qkv', 'qkv_norm_weight'),
                ('o', 'o_norm_weight'),
                ('gate_up', 'gate_up_norm_weight'),
                ('down', 'down_norm_weight'),
            ]:
                # Write norm_weight (float32)
                nw_np = ld[nw_name].numpy()
                write_tensor(f, nw_np, 1.0)
                total_bytes += 8 + nw_np.nbytes

                # Write packed ternary weights
                packed, scale = ld[proj_name]
                write_tensor(f, packed, scale)
                total_bytes += 8 + packed.nbytes

        # LM head (INT8)
        write_tensor(f, lm_head_q.numpy(), lm_head_scale)
        total_bytes += 8 + lm_head_q.numpy().nbytes

        # H_init and L_init (float32, no quantization — scale=1.0)
        write_tensor(f, h_init.numpy(), 1.0)
        total_bytes += 8 + h_init.numpy().nbytes
        write_tensor(f, l_init.numpy(), 1.0)
        total_bytes += 8 + l_init.numpy().nbytes

        # Q-head weight and bias (float32, v4+)
        q_head_w_np = q_head_weight.numpy()
        q_head_b_np = q_head_bias.numpy()
        write_tensor(f, q_head_w_np, 1.0)
        total_bytes += 8 + q_head_w_np.nbytes
        write_tensor(f, q_head_b_np, 1.0)
        total_bytes += 8 + q_head_b_np.nbytes

    print(f"\n{'='*50}")
    print(f"Export complete: {output_path}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes / (1024*1024):.2f} MB)")
    print(f"Vocab size: {vocab_size}")
    print(f"Layers: {L_LAYERS}")
    print(f"{'='*50}")

    # Print per-component breakdown
    print(f"\nSize breakdown:")
    embed_bytes = embed_q.numpy().nbytes
    puzzle_emb_bytes = puzzle_emb_np.nbytes
    lm_head_bytes = lm_head_q.numpy().nbytes
    init_bytes = h_init.numpy().nbytes + l_init.numpy().nbytes
    q_head_bytes = q_head_w_np.nbytes + q_head_b_np.nbytes
    ternary_bytes = sum(
        ld[name][0].nbytes
        for ld in layer_data
        for name in ['qkv', 'o', 'gate_up', 'down']
    )
    norm_weight_bytes = sum(
        ld[name].numpy().nbytes
        for ld in layer_data
        for name in ['qkv_norm_weight', 'o_norm_weight', 'gate_up_norm_weight', 'down_norm_weight']
    )
    print(f"  Header:       16 bytes")
    print(f"  Embedding:    {embed_bytes:,} bytes ({embed_bytes/1024:.1f} KB)")
    print(f"  Puzzle emb:   {puzzle_emb_bytes:,} bytes ({puzzle_emb_bytes/1024:.1f} KB)")
    print(f"  Norm weights: {norm_weight_bytes:,} bytes ({norm_weight_bytes/1024:.1f} KB)")
    print(f"  Ternary:      {ternary_bytes:,} bytes ({ternary_bytes/1024:.1f} KB)")
    print(f"  LM head:      {lm_head_bytes:,} bytes ({lm_head_bytes/1024:.1f} KB)")
    print(f"  H_init:       {h_init.numpy().nbytes:,} bytes ({h_init.numpy().nbytes/1024:.1f} KB)")
    print(f"  L_init:       {l_init.numpy().nbytes:,} bytes ({l_init.numpy().nbytes/1024:.1f} KB)")
    print(f"  Q-head:       {q_head_bytes:,} bytes ({q_head_bytes/1024:.1f} KB)")
    print(f"  Metadata:     {total_bytes - 16 - embed_bytes - puzzle_emb_bytes - norm_weight_bytes - ternary_bytes - lm_head_bytes - init_bytes - q_head_bytes} bytes")

    return total_bytes


def verify_export(output_path: str):
    """Read back the binary and verify header integrity."""
    with open(output_path, 'rb') as f:
        magic, version, vocab_size, num_layers = struct.unpack('<IIII', f.read(16))

    print(f"\nVerification:")
    print(f"  Magic:   0x{magic:08X} {'✅' if magic == MODEL_MAGIC else '❌'}")
    print(f"  Version: {version} {'✅' if version == MODEL_VERSION else '❌'}")
    print(f"  Vocab:   {vocab_size}")
    print(f"  Layers:  {num_layers} {'✅' if num_layers == L_LAYERS else '❌'}")

    # Count total tensors
    with open(output_path, 'rb') as f:
        f.seek(16)  # skip header
        tensor_count = 0
        total_data = 0
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size, scale = struct.unpack('<If', header)
            f.seek(size, 1)  # skip data
            tensor_count += 1
            total_data += size

    # v4: embed + puzzle_emb + layers*(norm_weight + ternary)*4 + lm_head + h_init + l_init + q_head_w + q_head_b
    expected_tensors = 1 + 1 + (L_LAYERS * 8) + 1 + 2 + 2
    print(f"  Tensors: {tensor_count} {'✅' if tensor_count == expected_tensors else '❌'}")
    print(f"           (expected {expected_tensors}: 1 embed + 1 puzzle_emb + {L_LAYERS}*8 layer + 1 lm_head + 2 init + 2 q_head)")
    print(f"  Data:    {total_data:,} bytes ({total_data/(1024*1024):.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TRM model for ESP32-S3 ternary inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint file')
    parser.add_argument('--output', type=str, default='esp32_trm/firmware/model_data/trm_ternary.bin',
                        help='Output binary file path')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the exported file after writing')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed export info')
    args = parser.parse_args()

    export_ternary(args.checkpoint, args.output, args.verbose)

    if args.verify:
        verify_export(args.output)

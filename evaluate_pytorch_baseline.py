#!/usr/bin/env python3
"""
PyTorch Baseline Evaluation for TRM Accuracy Verification (Sudoku).

Evaluates the TRM model on the Sudoku test subset. For Sudoku there is only
1 puzzle identifier, so the model always uses the same puzzle embedding.
We evaluate two configurations to match ESP32:

  1. bitnet_with_emb (ACT=1):  Direct ESP32 comparison target
  2. bitnet_with_emb (ACT=16): Full ACT evaluation (accuracy ceiling)

Usage:
    python evaluate_pytorch_baseline.py \
        --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/bitnet_sudoku/step_48828 \
        --config checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/bitnet_sudoku/all_config.yaml \
        --test-subset esp32_trm/test_subset.json \
        --output-dir esp32_trm/results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add TinyRecursiveModels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TinyRecursiveModels"))

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


def _strip_checkpoint_prefix(state_dict):
    """
    Remove checkpoint key prefixes for loading into TinyRecursiveReasoningModel_ACTV1.

    Checkpoint keys: _orig_mod.model.inner.L_level.layers.0...
    Model expects:   inner.L_level.layers.0...

    Steps: strip '_orig_mod.' and 'model.' prefixes.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        # Remove _orig_mod. from torch.compile
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        # Remove extra model. wrapper (training wrapper)
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        new_sd[new_key] = v
    return new_sd


def load_model_and_config(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load model from checkpoint and config."""
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    arch_config = config["arch"]
    data_config = config.get("data", {})

    print(f"  Architecture: {arch_config['name']}")
    print(f"  Hidden size:  {arch_config['hidden_size']}")
    print(f"  use_bitnet:   {arch_config['use_bitnet']}")
    print(f"  Dataset:      {data_config.get('name', 'unknown')}")

    # Read dataset-specific values
    vocab_size = data_config.get("vocab_size", 11)
    seq_len = data_config.get("seq_len", 81)
    num_puzzle_identifiers = data_config.get("num_puzzle_identifiers", 1)

    config_dict = {
        "batch_size": 1,
        "seq_len": seq_len,
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "vocab_size": vocab_size,
        "H_cycles": arch_config["H_cycles"],
        "L_cycles": arch_config["L_cycles"],
        "H_layers": arch_config.get("H_layers", 0),
        "L_layers": arch_config["L_layers"],
        "hidden_size": arch_config["hidden_size"],
        "expansion": arch_config["expansion"],
        "num_heads": arch_config["num_heads"],
        "pos_encodings": arch_config["pos_encodings"],
        "halt_max_steps": arch_config["halt_max_steps"],
        "halt_exploration_prob": arch_config.get("halt_exploration_prob", 0.1),
        "forward_dtype": arch_config.get("forward_dtype", "bfloat16"),
        "puzzle_emb_ndim": arch_config.get("puzzle_emb_ndim", 0),
        "puzzle_emb_len": arch_config.get("puzzle_emb_len", 16),
        "use_bitnet": arch_config["use_bitnet"],
        "mlp_t": arch_config.get("mlp_t", False),
        "no_ACT_continue": arch_config.get("no_ACT_continue", True),
        "rms_norm_eps": arch_config.get("rms_norm_eps", 1e-5),
        "rope_theta": arch_config.get("rope_theta", 10000.0),
    }

    print(f"  vocab_size:   {vocab_size}")
    print(f"  seq_len:      {seq_len}")
    print(f"  num_puzzle_identifiers: {num_puzzle_identifiers}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in checkpoint:
                sd = _strip_checkpoint_prefix(checkpoint[key])
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(f"  Loaded state dict from key '{key}'")
                if missing:
                    print(f"  WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
                if unexpected:
                    print(f"  WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
                break
        else:
            if any("weight" in k for k in checkpoint.keys()):
                sd = _strip_checkpoint_prefix(checkpoint)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print("  Loaded raw state dict")
                if missing:
                    print(f"  WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
                if unexpected:
                    print(f"  WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model = model.to(device)
    model.eval()
    return model, config_dict


@torch.no_grad()
def run_act_inference(
    model: TinyRecursiveReasoningModel_ACTV1,
    input_tokens: torch.Tensor,
    puzzle_identifiers: torch.Tensor,
    act_steps: int,
    haltable: bool,
    halt_rule: str,
    device: str,
):
    """
    Run ACT wrapper and return final argmax predictions.

    Two modes:
      - Fixed-step (haltable=False): run exactly `act_steps` improvement steps.
      - Haltable (haltable=True): run up to `act_steps` steps, stopping early if the
        learned halting criterion triggers (based on q_halt/q_continue logits).
    """
    batch = {
        "inputs": input_tokens.to(device),
        "puzzle_identifiers": puzzle_identifiers.to(device),
    }

    with torch.device(device):
        carry = model.initial_carry(batch)
    outputs = None
    steps_used = 0
    for _ in range(act_steps):
        carry, outputs = model(carry=carry, batch=batch)
        steps_used += 1

        if haltable:
            # NOTE: In this codebase, the ACT wrapper's internal `carry.halted`
            # does NOT implement learned halting during eval; it is constrained
            # to max steps for batching consistency. For a true "haltable at eval"
            # comparison, we implement the halting criterion here using the
            # halting head logits returned in outputs.
            q_halt = outputs["q_halt_logits"]
            q_cont = outputs["q_continue_logits"]

            if halt_rule == "auto":
                # Match training-time criterion controlled by config.no_ACT_continue
                if getattr(model.config, "no_ACT_continue", True):
                    halted = q_halt > 0
                else:
                    halted = q_halt > q_cont
            elif halt_rule == "q_halt_gt0":
                halted = q_halt > 0
            elif halt_rule == "q_halt_gt_q_continue":
                halted = q_halt > q_cont
            else:
                raise ValueError(f"Unknown halt_rule: {halt_rule}")

            # Batch size is 1 in this script, but keep it generic.
            if bool(halted.all()):
                break

    assert outputs is not None
    logits = outputs["logits"]  # [B, seq_len, vocab]
    preds = logits.argmax(dim=-1)
    return preds.detach().cpu(), steps_used


def evaluate_config(
    model: TinyRecursiveReasoningModel_ACTV1,
    puzzles: list,
    config_name: str,
    device: str = "cpu",
    act_steps: int = 1,
    haltable: bool = False,
    halt_rule: str = "auto",
):
    """Evaluate model on puzzle subset for a given configuration."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"  Puzzles:     {len(puzzles)}")
    print(f"  ACT steps:   {act_steps}")
    if haltable:
        print(f"  Haltable:    YES (halt_rule={halt_rule})")
    print(f"{'='*60}")

    inner = model.inner
    model_seq_len = int(inner.config.seq_len)
    results = []
    steps_used_all = []

    for i, puzzle in enumerate(puzzles):
        seq_len = int(puzzle["seq_len"])

        tokens = puzzle["input_tokens"]
        # For Sudoku: seq_len is always 81, matching model seq_len
        if len(tokens) > model_seq_len:
            tokens = tokens[:model_seq_len]
        elif len(tokens) < model_seq_len:
            tokens = tokens + [0] * (model_seq_len - len(tokens))

        input_tokens = torch.tensor([tokens], dtype=torch.int32)

        # Sudoku has 1 puzzle identifier (always 0)
        puzzle_id = torch.tensor([0], dtype=torch.int64)

        t0 = time.perf_counter()
        preds, steps_used = run_act_inference(
            model,
            input_tokens,
            puzzle_id,
            act_steps=act_steps,
            haltable=haltable,
            halt_rule=halt_rule,
            device=device,
        )
        t1 = time.perf_counter()

        pred_list = preds[0, :seq_len].tolist()

        steps_used_all.append(int(steps_used))
        results.append({
            "example_index": puzzle.get("example_index"),
            "group_id": puzzle.get("group_id", 0),
            "puzzle_identifier": 0,
            "seq_len": seq_len,
            "predictions": pred_list,
            "time_ms": round((t1 - t0) * 1000, 1),
            "steps_used": int(steps_used),
        })

        if (i + 1) % 50 == 0 or i == 0:
            if haltable and steps_used_all:
                avg_steps = sum(steps_used_all) / len(steps_used_all)
                print(f"  [{i+1}/{len(puzzles)}] seq_len={seq_len}, time={t1-t0:.3f}s, steps_used={steps_used} (avg_steps={avg_steps:.2f})")
            else:
                print(f"  [{i+1}/{len(puzzles)}] seq_len={seq_len}, time={t1-t0:.3f}s")

    return results, steps_used_all


def main():
    parser = argparse.ArgumentParser(description="PyTorch baseline evaluation for TRM accuracy (Sudoku)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to all_config.yaml")
    parser.add_argument("--test-subset", type=str, default="esp32_trm/test_subset.json",
                        help="Path to test subset JSON")
    parser.add_argument("--output-dir", type=str, default="esp32_trm/results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--act-steps", type=int, nargs="+", default=[1],
                        help="ACT steps to evaluate (e.g. --act-steps 1 16)")
    parser.add_argument("--haltable", action="store_true",
                        help="Enable haltable eval: run up to N steps but stop early using halting head logits")
    parser.add_argument("--halt-rule", type=str, default="auto",
                        choices=["auto", "q_halt_gt0", "q_halt_gt_q_continue"],
                        help="Halting criterion when --haltable is set. 'auto' matches config.no_ACT_continue.")
    parser.add_argument("--swap-norm-weights", type=str, default=None,
                        help="Path to a donor checkpoint whose norm_weight parameters will replace the main model's.")
    parser.add_argument("--simulate-esp32-truncation", action="store_true",
                        help="Patch activation_quant to use truncation instead of rounding, matching ESP32 firmware.")
    args = parser.parse_args()

    # Auto-detect CUDA
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda:0"
        print(f"Auto-detected CUDA, using {args.device}")

    # Load test subset
    print(f"Loading test subset: {args.test_subset}")
    with open(args.test_subset) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    print(f"  {len(puzzles)} puzzles loaded")
    print(f"  Dataset: {data.get('metadata', {}).get('dataset', 'unknown')}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config_dict = load_model_and_config(args.checkpoint, args.config, args.device)

    # --- Experiment flags ---
    experiment_tags = []

    # Swap norm_weight parameters from a donor checkpoint
    if args.swap_norm_weights:
        print(f"\n[experiment] Swapping norm_weights from: {args.swap_norm_weights}")
        donor_ckpt = torch.load(args.swap_norm_weights, map_location=args.device, weights_only=False)
        if not isinstance(donor_ckpt, dict):
            raise TypeError(f"Donor checkpoint is not a dict: {type(donor_ckpt)}")
        donor_sd = _strip_checkpoint_prefix(donor_ckpt)
        swapped = 0
        for name, param in model.named_parameters():
            if "norm_weight" not in name:
                continue
            if name in donor_sd:
                old_ratio = param.data.abs().max().item() / param.data.abs().mean().clamp(min=1e-8).item()
                param.data.copy_(donor_sd[name].to(param.dtype).to(param.device))
                new_ratio = param.data.abs().max().item() / param.data.abs().mean().clamp(min=1e-8).item()
                print(f"  Swapped {name}: max/mean {old_ratio:.2f} -> {new_ratio:.2f}")
                swapped += 1
            else:
                print(f"  WARNING: {name} not found in donor checkpoint")
        print(f"  Total: {swapped} norm_weight parameters swapped")
        experiment_tags.append("swapped_nw")

    # Patch activation_quant to use truncation (matching ESP32 firmware)
    if args.simulate_esp32_truncation:
        import models.bitnet_linear as _bl

        def _activation_quant_truncate(x: torch.Tensor) -> torch.Tensor:
            scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
            y = (x * scale).trunc().clamp_(-128, 127) / scale
            return y

        _bl.activation_quant = _activation_quant_truncate
        print("[experiment] Patched activation_quant to use truncation (ESP32 mode)")
        experiment_tags.append("trunc")

    tag_suffix = ("_" + "_".join(experiment_tags)) if experiment_tags else ""

    # Evaluate for each ACT step count
    for act_steps in args.act_steps:
        if args.haltable:
            config_name = f"bitnet_with_emb_haltable_max{act_steps}{tag_suffix}"
            display = f"BitNet + Puzzle Emb (Haltable, max_steps={act_steps})"
            if experiment_tags:
                display += f" [{', '.join(experiment_tags)}]"
        else:
            config_name = f"bitnet_with_emb_act{act_steps}{tag_suffix}"
            display = f"BitNet + Puzzle Emb (ACT={act_steps})"
            if experiment_tags:
                display += f" [{', '.join(experiment_tags)}]"

        results, steps_used_all = evaluate_config(
            model,
            puzzles,
            config_name=display,
            device=args.device,
            act_steps=act_steps,
            haltable=args.haltable,
            halt_rule=args.halt_rule,
        )

        # Save results
        output_path = os.path.join(args.output_dir, f"results_pytorch_{config_name}.json")
        output_data = {
            "config": config_name,
            "act_steps": act_steps,
            "haltable": bool(args.haltable),
            "halt_rule": args.halt_rule if args.haltable else None,
            "avg_steps_used": (sum(steps_used_all) / len(steps_used_all)) if steps_used_all else None,
            "num_puzzles": len(results),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved to: {output_path}")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()

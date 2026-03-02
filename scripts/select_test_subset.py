#!/usr/bin/env python3
"""
Select a Sudoku test subset for ESP32 accuracy evaluation.

Sudoku datasets in this repo are stored as:
  - all__inputs.npy:             [num_examples, 81]
  - all__labels.npy:             [num_examples, 81]
  - all__puzzle_indices.npy:     [num_puzzles + 1] prefix-sum boundaries
  - all__group_indices.npy:      [num_groups + 1] prefix-sum boundaries
  - all__puzzle_identifiers.npy: [num_puzzles] puzzle identifier IDs

For Sudoku:
  - Fixed seq_len = 81 (9x9 grid)
  - vocab_size = 11
  - num_puzzle_identifiers = 1 (all puzzles share the same identifier)
  - No variable length or padding — every position is meaningful

This script selects a random subset of puzzles for evaluation.
All 81 positions are included (no truncation needed).

Usage (run from trm-bitnet parent; data-dir points to TinyRecursiveModels):
    python esp32_trm/scripts/select_test_subset.py \
        --data-dir TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000/test \
        --output esp32_trm/test_subset.json \
        --num-puzzles 200
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Select Sudoku test subset for ESP32 evaluation")
    parser.add_argument("--data-dir", type=str,
                        default="TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000/test",
                        help="Path to test data directory containing .npy files")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_root, "test_subset.json"),
                        help="Output JSON file path")
    parser.add_argument("--num-puzzles", type=int, default=200,
                        help="Number of puzzles to select (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Load test data
    data_dir = args.data_dir
    print(f"Loading test data from: {data_dir}")

    inputs_path = os.path.join(data_dir, "all__inputs.npy")
    labels_path = os.path.join(data_dir, "all__labels.npy")

    for path in [inputs_path, labels_path]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            print("Please ensure the .npy data files are available.")
            sys.exit(1)

    # Load data (use memmap for large datasets)
    inputs = np.load(inputs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    print(f"  Inputs shape:  {inputs.shape}, dtype: {inputs.dtype}")
    print(f"  Labels shape:  {labels.shape}, dtype: {labels.dtype}")

    num_total = len(inputs)
    seq_len = inputs.shape[1] if inputs.ndim > 1 else 81
    print(f"  Total examples: {num_total}")
    print(f"  Seq len:        {seq_len}")

    # Optional: load puzzle identifiers and group indices if they exist
    puzzle_identifiers_path = os.path.join(data_dir, "all__puzzle_identifiers.npy")
    group_indices_path = os.path.join(data_dir, "all__group_indices.npy")
    puzzle_indices_path = os.path.join(data_dir, "all__puzzle_indices.npy")

    has_groups = os.path.exists(group_indices_path) and os.path.exists(puzzle_indices_path)
    has_puzzle_ids = os.path.exists(puzzle_identifiers_path)

    if has_puzzle_ids:
        puzzle_identifiers = np.load(puzzle_identifiers_path, mmap_mode="r")
        print(f"  Puzzle identifiers: {puzzle_identifiers.shape}, dtype: {puzzle_identifiers.dtype}")
    else:
        puzzle_identifiers = None

    # Select random subset
    num_select = min(args.num_puzzles, num_total) if args.num_puzzles > 0 else num_total
    indices = rng.choice(num_total, size=num_select, replace=False)
    indices.sort()  # sort for reproducible ordering

    print(f"\nSelecting {num_select} puzzles from {num_total} total")

    selected = []
    for idx in indices:
        inp = inputs[idx]
        lab = labels[idx]

        entry = {
            "example_index": int(idx),
            "group_id": 0,  # Sudoku has 1 puzzle type
            "puzzle_identifier": 0,  # Single puzzle identifier
            "seq_len": int(seq_len),
            "input_tokens": inp[:seq_len].astype(np.uint8).tolist(),
            "label_tokens": lab[:seq_len].astype(np.uint8).tolist(),
        }

        # Add puzzle identifier if available
        if has_groups and has_puzzle_ids:
            # For Sudoku with 1 identifier, all map to 0
            entry["puzzle_identifier"] = 0

        selected.append(entry)

    print(f"\nSelection results:")
    print(f"  Selected: {len(selected)} puzzles")

    if len(selected) == 0:
        print("ERROR: No puzzles selected!")
        sys.exit(1)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "metadata": {
            "data_dir": args.data_dir,
            "dataset": "sudoku",
            "seq_len": int(seq_len),
            "vocab_size": 11,
            "seed": args.seed,
            "total_examples": int(num_total),
            "selected_count": len(selected),
        },
        "puzzles": selected,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {args.output}")
    file_size = os.path.getsize(args.output)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split the *remaining* Sudoku ESP32 evaluation puzzles into 4 chunks (a/b/c/d).

We assume you previously created two subset files (split_a + split_b) and have
partial ESP32 results for each. This script:
  - loads split_a + split_b
  - removes puzzles already present in results_a/results_b (predictions != None)
  - combines the remaining puzzles and splits them into 4 roughly-equal chunks
  - writes 4 new subset JSON files (a, b, c, d)

Example (run from esp32_trm repo root):
  python scripts/split_remaining_into_4.py \
    --subset-a test_subset_split_a.json \
    --subset-b test_subset_split_b.json \
    --results-a results/results_esp32_act16_haltable_a.json \
    --results-b results/results_esp32_act16_haltable_b.json \
    --out-dir . \
    --out-prefix test_subset_remaining_
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Set, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _done_example_indices(results_path: str) -> Set[int]:
    if not results_path:
        return set()
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)
    d = _load_json(results_path)
    out: Set[int] = set()
    for r in d.get("results", []) or []:
        ex = r.get("example_index")
        preds = r.get("predictions")
        if ex is None:
            continue
        if preds is None:
            continue
        out.add(int(ex))
    return out


def _split_even(xs: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    if n <= 0:
        raise ValueError("n must be > 0")
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    # round-robin gives stable and very even distribution even if you later
    # remove a few failed puzzles and re-run the split.
    for i, x in enumerate(xs):
        chunks[i % n].append(x)
    return chunks


def _summarize_chunks(chunks: List[List[Dict[str, Any]]]) -> str:
    sizes = [len(c) for c in chunks]
    return f"chunks={len(chunks)} sizes={sizes} total={sum(sizes)}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Split remaining ESP32 Sudoku subset into 4 chunks")
    ap.add_argument("--subset-a", required=True, help="Path to original subset A JSON (100 puzzles)")
    ap.add_argument("--subset-b", required=True, help="Path to original subset B JSON (100 puzzles)")
    ap.add_argument("--results-a", required=True, help="Path to ESP32 results JSON for subset A")
    ap.add_argument("--results-b", required=True, help="Path to ESP32 results JSON for subset B")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=_root, help="Output directory for new subset JSON files")
    ap.add_argument(
        "--out-prefix",
        default="test_subset_remaining_",
        help="Output filename prefix (suffix will be a/b/c/d + .json)",
    )
    ap.add_argument(
        "--order",
        choices=["example_index", "original"],
        default="example_index",
        help="Ordering before split: 'example_index' (deterministic) or 'original' (keep a then b file order)",
    )
    args = ap.parse_args()

    subset_a = _load_json(args.subset_a)
    subset_b = _load_json(args.subset_b)
    puzzles_a: List[Dict[str, Any]] = subset_a.get("puzzles", []) or []
    puzzles_b: List[Dict[str, Any]] = subset_b.get("puzzles", []) or []

    done_a = _done_example_indices(args.results_a)
    done_b = _done_example_indices(args.results_b)

    rem_a = [p for p in puzzles_a if int(p.get("example_index", -1)) not in done_a]
    rem_b = [p for p in puzzles_b if int(p.get("example_index", -1)) not in done_b]

    remaining = rem_a + rem_b
    if args.order == "example_index":
        remaining.sort(key=lambda p: int(p.get("example_index", -1)))

    # Split into 4 equal-ish chunks.
    chunks = _split_even(remaining, 4)

    os.makedirs(args.out_dir, exist_ok=True)

    # Merge metadata (both subsets should have same dataset/seed/vocab/seq_len).
    meta_a = dict(subset_a.get("metadata", {}) or {})
    meta_b = dict(subset_b.get("metadata", {}) or {})
    meta_out = dict(meta_a)
    # Keep a/b differences visible but non-breaking.
    meta_out["source_subsets"] = [os.path.basename(args.subset_a), os.path.basename(args.subset_b)]
    meta_out["source_results"] = [os.path.basename(args.results_a), os.path.basename(args.results_b)]
    meta_out["remaining_total"] = len(remaining)
    meta_out["remaining_from_a"] = len(rem_a)
    meta_out["remaining_from_b"] = len(rem_b)
    meta_out["done_in_a"] = len(done_a)
    meta_out["done_in_b"] = len(done_b)
    if meta_b and meta_b != meta_a:
        meta_out["note"] = "subset_a/ subset_b metadata differed; base metadata taken from subset_a"

    letters = ["a", "b", "c", "d"]
    out_paths: List[str] = []
    for letter, chunk in zip(letters, chunks):
        out_data = {
            "metadata": {
                **meta_out,
                "chunk": letter,
                "chunk_size": len(chunk),
                "num_chunks": 4,
            },
            "puzzles": chunk,
        }
        out_path = os.path.join(args.out_dir, f"{args.out_prefix}{letter}.json")
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        out_paths.append(out_path)

    print("Remaining puzzles (subset_a):", len(rem_a), "/", len(puzzles_a))
    print("Remaining puzzles (subset_b):", len(rem_b), "/", len(puzzles_b))
    print("Combined remaining:", len(remaining))
    print("Split:", _summarize_chunks(chunks))
    print("Wrote:")
    for p in out_paths:
        print("  -", p)


if __name__ == "__main__":
    main()


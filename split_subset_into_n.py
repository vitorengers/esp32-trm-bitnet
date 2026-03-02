#!/usr/bin/env python3
"""
Split an existing ESP32 test-subset JSON into N smaller subset JSONs.

Safety:
  - Never overwrites output files unless --force is passed.
  - Optionally excludes puzzles already completed in an existing results JSON
    (predictions != None).

Examples:
  # Split subset into 3 chunks
  python3 esp32_trm/split_subset_into_n.py \
    --subset esp32_trm/test_subset_remaining_c.json \
    --num-chunks 3 \
    --out-dir esp32_trm \
    --out-prefix test_subset_remaining_c_reassign_

  # Split only the remaining puzzles (resume safety)
  python3 esp32_trm/split_subset_into_n.py \
    --subset esp32_trm/test_subset_remaining_c.json \
    --exclude-results esp32_trm/results/results_esp32_act16_haltable_remaining_c.json \
    --num-chunks 3 \
    --out-dir esp32_trm \
    --out-prefix test_subset_remaining_c_reassign_
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set


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
        if ex is None or preds is None:
            continue
        out.add(int(ex))
    return out


def _split_round_robin(xs: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for i, x in enumerate(xs):
        chunks[i % n].append(x)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Split subset JSON into N chunks (safe, no overwrite by default)")
    ap.add_argument("--subset", required=True, help="Input subset JSON (has keys: metadata, puzzles)")
    ap.add_argument(
        "--exclude-results",
        default="",
        help="Optional results JSON; puzzles with example_index already completed will be excluded.",
    )
    ap.add_argument("--num-chunks", type=int, required=True, help="Number of chunks to produce")
    ap.add_argument("--out-dir", default="esp32_trm", help="Output directory")
    ap.add_argument("--out-prefix", required=True, help="Output prefix; will write <prefix>{0..n-1}.json")
    ap.add_argument(
        "--name-style",
        choices=["letters", "numbers"],
        default="letters",
        help="Chunk suffix style: letters (a,b,c,...) or numbers (0,1,2,...)",
    )
    ap.add_argument("--force", action="store_true", help="Allow overwriting existing output files")
    args = ap.parse_args()

    if args.num_chunks <= 0:
        raise SystemExit("--num-chunks must be > 0")

    subset = _load_json(args.subset)
    puzzles: List[Dict[str, Any]] = subset.get("puzzles", []) or []

    done = _done_example_indices(args.exclude_results) if args.exclude_results else set()
    if done:
        puzzles = [p for p in puzzles if int(p.get("example_index", -1)) not in done]

    puzzles.sort(key=lambda p: int(p.get("example_index", -1)))
    chunks = _split_round_robin(puzzles, args.num_chunks)

    os.makedirs(args.out_dir, exist_ok=True)

    def suffix(i: int) -> str:
        if args.name_style == "numbers":
            return str(i)
        # letters
        return chr(ord("a") + i)

    out_paths: List[str] = []
    for i, ch in enumerate(chunks):
        out_name = f"{args.out_prefix}{suffix(i)}.json"
        out_path = os.path.join(args.out_dir, out_name)
        if (not args.force) and os.path.exists(out_path):
            raise SystemExit(f"Refusing to overwrite existing file: {out_path} (pass --force to overwrite)")

        meta = dict(subset.get("metadata", {}) or {})
        meta["source_subset"] = os.path.basename(args.subset)
        if args.exclude_results:
            meta["excluded_results"] = os.path.basename(args.exclude_results)
        meta["num_chunks"] = args.num_chunks
        meta["chunk"] = suffix(i)
        meta["chunk_size"] = len(ch)
        meta["excluded_done"] = len(done)

        out_data = {"metadata": meta, "puzzles": ch}
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        out_paths.append(out_path)

    sizes = [len(c) for c in chunks]
    print(f"Input puzzles: {len(subset.get('puzzles', []) or [])}")
    print(f"Excluded done: {len(done)}")
    print(f"Output chunks: {args.num_chunks} sizes={sizes} total={sum(sizes)}")
    print("Wrote:")
    for p in out_paths:
        print("  -", p)


if __name__ == "__main__":
    main()


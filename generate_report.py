#!/usr/bin/env python3
"""
Report Generator for TRM Accuracy Verification (Sudoku).

Loads result JSONs from PyTorch and ESP32 evaluations, computes accuracy
metrics, and generates a markdown report.

For Sudoku: all 81 positions are meaningful (no padding to skip).

Metrics:
  - Token Accuracy: % of correct tokens (all 81 positions)
  - Sequence Exact Match: % of perfectly predicted sequences
  - ESP32 vs PyTorch agreement (deployment fidelity)

Usage:
    python generate_report.py \
        --test-subset esp32_trm/test_subset.json \
        --results-dir esp32_trm/results \
        --output esp32_trm/accuracy_report.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime


def compute_metrics(puzzles, results, ignore_label_id=0):
    """
    Compute accuracy metrics comparing predictions against labels.

    For Sudoku: all 81 positions are meaningful. The ignore_label_id (default 0)
    is used to skip positions where the label is 0 (blank/given cells that the
    model is not required to predict). Set ignore_label_id=-1 to count all positions.

    Returns dict with:
      - token_accuracy: % correct non-ignored tokens
      - exact_match: % perfectly matched sequences (non-ignored positions)
      - total_tokens: total counted tokens
      - correct_tokens: correctly predicted tokens
      - total_sequences: total sequences
      - exact_sequences: exactly matched sequences
      - avg_time_ms: average inference time (if available)
    """
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    exact_sequences = 0
    total_time_ms = 0
    timed_count = 0

    # Build lookup from puzzle example_index -> result
    result_map = {}
    for r in results:
        key = r.get("example_index")
        result_map[key] = r

    for puzzle in puzzles:
        idx = puzzle.get("example_index")
        if idx not in result_map:
            continue

        r = result_map[idx]
        if r.get("predictions") is None:
            continue  # skip errors

        labels = puzzle["label_tokens"]
        preds = r["predictions"]
        seq_len = min(len(labels), len(preds))

        seq_correct = True
        for pos in range(seq_len):
            if ignore_label_id >= 0 and labels[pos] == ignore_label_id:
                continue  # skip positions with ignore label
            total_tokens += 1
            if preds[pos] == labels[pos]:
                correct_tokens += 1
            else:
                seq_correct = False

        total_sequences += 1
        if seq_correct:
            exact_sequences += 1

        if r.get("time_ms") is not None:
            total_time_ms += r["time_ms"]
            timed_count += 1

    token_accuracy = correct_tokens / total_tokens * 100 if total_tokens > 0 else 0
    exact_match = exact_sequences / total_sequences * 100 if total_sequences > 0 else 0
    avg_time_ms = total_time_ms / timed_count if timed_count > 0 else None

    return {
        "token_accuracy": token_accuracy,
        "exact_match": exact_match,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "total_sequences": total_sequences,
        "exact_sequences": exact_sequences,
        "avg_time_ms": avg_time_ms,
    }


def compute_agreement(puzzles, results_a, results_b):
    """
    Compute token-level agreement between two sets of predictions.
    This measures deployment fidelity independent of ground truth.
    """
    map_a = {r.get("example_index"): r for r in results_a if r.get("predictions")}
    map_b = {r.get("example_index"): r for r in results_b if r.get("predictions")}

    total_tokens = 0
    agree_tokens = 0
    total_sequences = 0
    agree_sequences = 0

    for puzzle in puzzles:
        idx = puzzle.get("example_index")
        if idx not in map_a or idx not in map_b:
            continue

        preds_a = map_a[idx]["predictions"]
        preds_b = map_b[idx]["predictions"]
        seq_len = min(len(preds_a), len(preds_b))

        seq_agree = True
        for pos in range(seq_len):
            total_tokens += 1
            if preds_a[pos] == preds_b[pos]:
                agree_tokens += 1
            else:
                seq_agree = False

        total_sequences += 1
        if seq_agree:
            agree_sequences += 1

    return {
        "token_agreement": agree_tokens / total_tokens * 100 if total_tokens > 0 else 0,
        "sequence_agreement": agree_sequences / total_sequences * 100 if total_sequences > 0 else 0,
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
    }


def generate_report(puzzles, all_results, output_path, metadata=None):
    """Generate markdown report."""
    lines = []
    lines.append("# TRM Sudoku Accuracy Verification Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTest subset: {len(puzzles)} puzzles")
    if metadata:
        lines.append(f"Dataset: {metadata.get('dataset', 'sudoku')}")
        lines.append(f"Seq len: {metadata.get('seq_len', 81)}")
        lines.append(f"Vocab size: {metadata.get('vocab_size', 11)}")
    lines.append("")

    # Compute metrics for each configuration
    configs = {}
    for name, data in all_results.items():
        results = data.get("results", [])
        # For Sudoku: ignore_label_id=0 to skip blank/given cells
        metrics = compute_metrics(puzzles, results, ignore_label_id=0)
        configs[name] = metrics

    # Summary table
    lines.append("## Accuracy Comparison")
    lines.append("")
    lines.append("| Configuration | Token Accuracy | Exact Match | Avg Time |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for name, m in configs.items():
        time_str = f"{m['avg_time_ms']:.0f}ms" if m["avg_time_ms"] else "N/A"
        display_name = name.replace("_", " ").title()
        lines.append(f"| {display_name} | {m['token_accuracy']:.2f}% ({m['correct_tokens']}/{m['total_tokens']}) "
                      f"| {m['exact_match']:.2f}% ({m['exact_sequences']}/{m['total_sequences']}) "
                      f"| {time_str} |")

    lines.append("")

    # ESP32 vs PyTorch agreement
    esp32_key = None
    pytorch_target_key = None
    for name, data in all_results.items():
        name_l = name.lower()
        if "esp32" in name_l:
            esp32_key = name

        # Prefer explicit act_steps metadata if present (PyTorch baseline exports include it)
        if data.get("act_steps") == 1:
            pytorch_target_key = name

    # Fallback: match ACT=1 by name (avoid the 'act16 contains act1' pitfall)
    if not pytorch_target_key:
        for name in all_results:
            name_l = name.lower()
            if re.search(r"(?:^|_)act1(?:$|_)", name_l):
                pytorch_target_key = name
                break

    # Fallback: find any pytorch key
    if not pytorch_target_key:
        for name in all_results:
            if "pytorch" in name.lower() or "bitnet" in name.lower():
                pytorch_target_key = name
                break

    if esp32_key and pytorch_target_key:
        agreement = compute_agreement(
            puzzles,
            all_results[esp32_key]["results"],
            all_results[pytorch_target_key]["results"]
        )
        lines.append("## ESP32 vs PyTorch Agreement")
        lines.append("")
        lines.append(f"Comparison target: `{pytorch_target_key}`")
        lines.append("")
        lines.append(f"- **Token Agreement**: {agreement['token_agreement']:.2f}% "
                      f"({agreement['total_tokens']} tokens)")
        lines.append(f"- **Sequence Agreement**: {agreement['sequence_agreement']:.2f}% "
                      f"({agreement['total_sequences']} sequences)")
        lines.append("")
        lines.append("This measures how closely the ESP32 deployment matches the PyTorch "
                      "ternary model (independent of ground truth correctness).")
        lines.append("")

    # Timing analysis
    if esp32_key and all_results[esp32_key].get("total_esp_time_ms"):
        data = all_results[esp32_key]
        lines.append("## ESP32 Timing Analysis")
        lines.append("")
        lines.append(f"- Total ESP inference time: {data['total_esp_time_ms']/1000:.1f}s "
                      f"({data['total_esp_time_ms']/1000/60:.1f}min)")
        lines.append(f"- Total wall time: {data.get('total_wall_time_s', 0):.1f}s")
        lines.append(f"- Puzzles evaluated: {data['num_puzzles']}")
        lines.append(f"- Errors: {data.get('errors', 0)}")

        times = [r["time_ms"] for r in data["results"] if r.get("time_ms")]
        if times:
            lines.append(f"- Avg time per puzzle: {sum(times)/len(times):.0f}ms")
            lines.append(f"- Min time: {min(times)}ms")
            lines.append(f"- Max time: {max(times)}ms")
        lines.append("")

    # Detailed per-config analysis
    lines.append("## Detailed Metrics")
    lines.append("")
    for name, m in configs.items():
        display_name = name.replace("_", " ").title()
        lines.append(f"### {display_name}")
        lines.append(f"- Token accuracy: {m['token_accuracy']:.4f}%")
        lines.append(f"- Exact match: {m['exact_match']:.4f}%")
        lines.append(f"- Tokens: {m['correct_tokens']}/{m['total_tokens']} correct")
        lines.append(f"- Sequences: {m['exact_sequences']}/{m['total_sequences']} exact")
        if m["avg_time_ms"]:
            lines.append(f"- Average time: {m['avg_time_ms']:.1f}ms")
        lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Generate Sudoku accuracy verification report")
    parser.add_argument("--test-subset", type=str, default="esp32_trm/test_subset.json",
                        help="Path to test subset JSON")
    parser.add_argument("--results-dir", type=str, default="esp32_trm/results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default="esp32_trm/accuracy_report.md",
                        help="Output markdown report path")
    args = parser.parse_args()

    # Load test subset
    print(f"Loading test subset: {args.test_subset}")
    with open(args.test_subset) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    metadata = data.get("metadata", {})
    print(f"  {len(puzzles)} puzzles")

    # Load all result files
    all_results = {}
    if os.path.isdir(args.results_dir):
        for fname in sorted(os.listdir(args.results_dir)):
            if fname.startswith("results_") and fname.endswith(".json"):
                path = os.path.join(args.results_dir, fname)
                print(f"Loading: {path}")
                with open(path) as f:
                    result_data = json.load(f)
                config_name = result_data.get("config", fname.replace("results_", "").replace(".json", ""))
                all_results[config_name] = result_data

    if not all_results:
        print("ERROR: No result files found in", args.results_dir)
        sys.exit(1)

    print(f"\nFound {len(all_results)} result configurations: {list(all_results.keys())}")

    # Generate report
    generate_report(puzzles, all_results, args.output, metadata)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate all publication-quality figures for the dissertation.
Outputs PDF figures to /home/engers/workspace/masters/figuras/.

Usage:
    python dissertation_graphs.py [--output-dir /path/to/figuras]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path("/home/engers/workspace/masters/figuras")

# ── Styling ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "paper": "#7f7f7f",
    "round": "#1f77b4",
    "trunc": "#ff7f0e",
    "fixed": "#2ca02c",
    "haltable": "#d62728",
    "both_correct": "#2ca02c",
    "pc_only": "#1f77b4",
    "esp_only": "#ff7f0e",
    "both_wrong": "#d62728",
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_ground_truth(subset_path):
    data = load_json(subset_path)
    puzzles = data.get("puzzles", data) if isinstance(data, dict) else data
    if isinstance(puzzles, dict):
        puzzles = puzzles.get("puzzles", [])
    gt = {}
    for p in puzzles:
        gt[p["example_index"]] = p["label_tokens"]
    return gt


def load_results(result_path):
    data = load_json(result_path)
    results = data.get("results", [])
    out = {}
    for r in results:
        if r.get("predictions") is not None:
            out[r["example_index"]] = r
    return out


def merge_esp32_results(result_dir, pattern="results_esp32_round_haltable_*.json"):
    merged = {}
    result_dir = Path(result_dir)
    for f in sorted(result_dir.glob(pattern)):
        data = load_json(f)
        for r in data.get("results", []):
            if r.get("predictions") is not None:
                idx = r["example_index"]
                if idx not in merged:
                    merged[idx] = r
    return merged


def compute_puzzle_metrics(predictions, label_tokens):
    correct = sum(1 for p, l in zip(predictions, label_tokens) if p == l)
    total = len(label_tokens)
    token_acc = correct / total if total > 0 else 0.0
    exact = 1 if correct == total else 0
    return token_acc, exact


# ── Graph 1: Paper Baseline Comparison ───────────────────────────────────────

def graph_baseline_comparison(output_dir):
    models = ["Paper TRM-Att\n(float weights)", "bf16-round\n(ternary)", "bf16-trunc\n(ternary)"]
    exact_match = [74.7, 72.36, 64.85]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    bar_colors = [COLORS["paper"], COLORS["round"], COLORS["trunc"]]

    for i, val in enumerate(exact_match):
        ax.bar(x[i], val, 0.5, color=bar_colors[i], edgecolor="black", linewidth=0.5)
        ax.text(x[i], val + 0.8, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 85)
    ax.set_title("Full Dataset, Fixed 16 Steps (422,786 puzzles)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_baseline_comparison.pdf")
    plt.close(fig)
    print(f"  Saved fig_baseline_comparison.pdf")


# ── Graph 2: Haltable vs Fixed ───────────────────────────────────────────────

def graph_haltable_vs_fixed(output_dir):
    categories = ["bf16-round\nFixed 16", "bf16-round\nHaltable", "bf16-trunc\nFixed 16", "bf16-trunc\nHaltable"]
    exact_match = [72.36, 72.20, 64.85, 64.76]
    avg_steps = [16.0, 6.59, 16.0, 7.84]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    colors = [COLORS["fixed"], COLORS["haltable"], COLORS["fixed"], COLORS["haltable"]]

    for i, (val, steps) in enumerate(zip(exact_match, avg_steps)):
        ax.bar(x[i], val, 0.55, color=colors[i], edgecolor="black", linewidth=0.5)
        label = f"{val:.2f}%\n({steps:.1f} steps)"
        ax.text(x[i], val + 0.8, label, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 82)
    ax.set_title("Haltable vs Fixed-Step Evaluation (Full Dataset)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["fixed"], edgecolor="black", label="Fixed 16"),
                       Patch(facecolor=COLORS["haltable"], edgecolor="black", label="Haltable")]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_haltable_vs_fixed.pdf")
    plt.close(fig)
    print(f"  Saved fig_haltable_vs_fixed.pdf")


# ── Graph 3: ACT Steps Distribution ─────────────────────────────────────────

def graph_act_steps_distribution(output_dir):
    pc_haltable_path = SCRIPT_DIR / "results" / "results_pytorch_bitnet_with_emb_haltable_max16.json"
    if not pc_haltable_path.exists():
        print(f"  SKIP fig_act_steps_distribution.pdf (missing {pc_haltable_path})")
        return

    data = load_json(pc_haltable_path)
    steps_list = [r.get("steps_used", 16) for r in data.get("results", []) if r.get("predictions") is not None]

    if not steps_list:
        print(f"  SKIP fig_act_steps_distribution.pdf (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0.5, 17.5, 1)
    counts, _, bars = ax.hist(steps_list, bins=bins, color=COLORS["round"], edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    f"{int(count)}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of ACT Steps Used")
    ax.set_ylabel("Number of Puzzles")
    ax.set_xticks(range(1, 17))
    ax.set_title(f"ACT Halting Step Distribution (bf16-round, 200-puzzle subset, avg={np.mean(steps_list):.2f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_act_steps_distribution.pdf")
    plt.close(fig)
    print(f"  Saved fig_act_steps_distribution.pdf")


# ── Graph 4: Platform Congruence ─────────────────────────────────────────────

def graph_congruence(output_dir, model_name="round"):
    gt_path = SCRIPT_DIR / "test_subset.json"
    if not gt_path.exists():
        print(f"  SKIP fig_congruence_{model_name}.pdf (missing ground truth)")
        return

    gt = load_ground_truth(gt_path)

    if model_name == "round":
        pc_path = SCRIPT_DIR / "results" / "results_pytorch_bitnet_with_emb_haltable_max16.json"
        esp_dir = SCRIPT_DIR / "results_round_esp32"
        esp_pattern = "results_esp32_round_haltable_*.json"
    elif model_name == "trunc":
        pc_path = SCRIPT_DIR / "results" / "results_pytorch_bitnet_with_emb_haltable_max16_trunc.json"
        esp_dir = SCRIPT_DIR / "results_trunc_esp32"
        esp_pattern = "results_esp32_trunc_haltable_*.json"
    else:
        print(f"  SKIP fig_congruence_{model_name}.pdf (unknown model)")
        return

    if not pc_path.exists():
        print(f"  SKIP fig_congruence_{model_name}.pdf (missing PC results: {pc_path})")
        return
    if not esp_dir.exists():
        print(f"  SKIP fig_congruence_{model_name}.pdf (missing ESP32 results: {esp_dir})")
        return

    pc_results = load_results(pc_path)
    esp_results = merge_esp32_results(esp_dir, esp_pattern)

    common_indices = set(pc_results.keys()) & set(esp_results.keys()) & set(gt.keys())
    if len(common_indices) < 10:
        print(f"  SKIP fig_congruence_{model_name}.pdf (only {len(common_indices)} common puzzles)")
        return

    both_correct = 0
    pc_only = 0
    esp_only = 0
    both_wrong = 0

    for idx in sorted(common_indices):
        labels = gt[idx]
        pc_pred = pc_results[idx]["predictions"]
        esp_pred = esp_results[idx]["predictions"]

        pc_exact = all(p == l for p, l in zip(pc_pred, labels))
        esp_exact = all(p == l for p, l in zip(esp_pred, labels))

        if pc_exact and esp_exact:
            both_correct += 1
        elif pc_exact:
            pc_only += 1
        elif esp_exact:
            esp_only += 1
        else:
            both_wrong += 1

    total = len(common_indices)
    categories = ["Both\nCorrect", "PC Only", "ESP32 Only", "Both\nWrong"]
    values = [both_correct, pc_only, esp_only, both_wrong]
    colors = [COLORS["both_correct"], COLORS["pc_only"], COLORS["esp_only"], COLORS["both_wrong"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of Puzzles")
    ax.set_title(f"Puzzle-Level Congruence: PC vs ESP32 (bf16-{model_name}, n={total})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.2 + 5)

    fig.tight_layout()
    fig.savefig(output_dir / f"fig_congruence_{model_name}.pdf")
    plt.close(fig)
    print(f"  Saved fig_congruence_{model_name}.pdf")


# ── Graph 5: Token Accuracy Scatter ──────────────────────────────────────────

def graph_token_accuracy_scatter(output_dir, model_name="round"):
    gt_path = SCRIPT_DIR / "test_subset.json"
    if not gt_path.exists():
        print(f"  SKIP fig_token_accuracy_scatter.pdf (missing ground truth)")
        return

    gt = load_ground_truth(gt_path)

    if model_name == "round":
        pc_path = SCRIPT_DIR / "results" / "results_pytorch_bitnet_with_emb_haltable_max16.json"
        esp_dir = SCRIPT_DIR / "results_round_esp32"
        esp_pattern = "results_esp32_round_haltable_*.json"
    elif model_name == "trunc":
        pc_path = SCRIPT_DIR / "results" / "results_pytorch_bitnet_with_emb_haltable_max16_trunc.json"
        esp_dir = SCRIPT_DIR / "results_trunc_esp32"
        esp_pattern = "results_esp32_trunc_haltable_*.json"
    else:
        print(f"  SKIP fig_token_accuracy_scatter.pdf (unknown model: {model_name})")
        return

    if not pc_path.exists() or not esp_dir.exists():
        print(f"  SKIP fig_token_accuracy_scatter_{model_name}.pdf (missing data)")
        return

    pc_results = load_results(pc_path)
    esp_results = merge_esp32_results(esp_dir, esp_pattern)
    common_indices = set(pc_results.keys()) & set(esp_results.keys()) & set(gt.keys())

    if len(common_indices) < 10:
        print(f"  SKIP fig_token_accuracy_scatter_{model_name}.pdf (only {len(common_indices)} common puzzles)")
        return

    pc_accs, esp_accs, categories = [], [], []
    for idx in sorted(common_indices):
        labels = gt[idx]
        pc_ta, pc_ex = compute_puzzle_metrics(pc_results[idx]["predictions"], labels)
        esp_ta, esp_ex = compute_puzzle_metrics(esp_results[idx]["predictions"], labels)
        pc_accs.append(pc_ta * 100)
        esp_accs.append(esp_ta * 100)
        if pc_ex and esp_ex:
            categories.append("both_correct")
        elif pc_ex:
            categories.append("pc_only")
        elif esp_ex:
            categories.append("esp_only")
        else:
            categories.append("both_wrong")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot([40, 105], [40, 105], "k--", alpha=0.3, linewidth=0.8, label="Perfect agreement")

    cat_map = {
        "both_correct": ("Both correct", COLORS["both_correct"], "o", 30),
        "pc_only": ("PC only", COLORS["pc_only"], "^", 40),
        "esp_only": ("ESP32 only", COLORS["esp_only"], "v", 40),
        "both_wrong": ("Both wrong", COLORS["both_wrong"], "x", 30),
    }

    for cat_key, (label, color, marker, size) in cat_map.items():
        mask = [c == cat_key for c in categories]
        xs = [v for v, m in zip(pc_accs, mask) if m]
        ys = [v for v, m in zip(esp_accs, mask) if m]
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, s=size, label=f"{label} ({len(xs)})", alpha=0.7, edgecolors="none")

    ax.set_xlabel("PC Token Accuracy (%)")
    ax.set_ylabel("ESP32 Token Accuracy (%)")
    ax.set_title(f"Per-Puzzle Token Accuracy: PC vs ESP32 (bf16-{model_name})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(40, 105)
    ax.set_ylim(40, 105)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    suffix = f"_{model_name}" if model_name != "round" else ""
    fig.tight_layout()
    fig.savefig(output_dir / f"fig_token_accuracy_scatter{suffix}.pdf")
    plt.close(fig)
    print(f"  Saved fig_token_accuracy_scatter{suffix}.pdf")


# ── Graph 6: ESP32 Timing Distribution ───────────────────────────────────────

def graph_esp32_timing(output_dir):
    esp_dir = SCRIPT_DIR / "results_round_esp32"
    if not esp_dir.exists():
        print(f"  SKIP fig_esp32_timing.pdf (missing ESP32 results)")
        return

    esp_results = merge_esp32_results(esp_dir, "results_esp32_round_haltable_*.json")
    times_s = [r["time_ms"] / 1000.0 for r in esp_results.values() if r.get("time_ms")]

    if len(times_s) < 5:
        print(f"  SKIP fig_esp32_timing.pdf (only {len(times_s)} timing entries)")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(times_s, bins=30, color=COLORS["round"], edgecolor="black", linewidth=0.5, alpha=0.85)

    mean_t = np.mean(times_s)
    median_t = np.median(times_s)
    ax.axvline(mean_t, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_t:.0f}s ({mean_t/60:.1f} min)")
    ax.axvline(median_t, color="green", linestyle=":", linewidth=1.2, label=f"Median: {median_t:.0f}s ({median_t/60:.1f} min)")

    ax.set_xlabel("Inference Time per Puzzle (seconds)")
    ax.set_ylabel("Number of Puzzles")
    ax.set_title(f"ESP32 Per-Puzzle Inference Time (bf16-round, n={len(times_s)})")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_esp32_timing.pdf")
    plt.close(fig)
    print(f"  Saved fig_esp32_timing.pdf")


# ── Graph 7: Round vs Trunc Summary ──────────────────────────────────────────

def graph_round_vs_trunc_summary(output_dir):
    scenarios = [
        "PC Full\nFixed 16",
        "PC Full\nHaltable",
        "PC 200\nHaltable",
        "ESP32 200\nHaltable",
    ]
    round_vals = [72.36, 72.20, 66.00, 65.50]
    trunc_vals = [64.85, 64.76, 58.50, 57.50]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scenarios))
    width = 0.32

    ax.bar(x - width / 2, round_vals, width,
           color=COLORS["round"], edgecolor="black", linewidth=0.5, label="bf16-round")

    for i, v in enumerate(round_vals):
        ax.text(x[i] - width / 2, v + 0.8, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.bar(x + width / 2, trunc_vals, width,
           color=COLORS["trunc"], edgecolor="black", linewidth=0.5, label="bf16-trunc")

    for i, v in enumerate(trunc_vals):
        ax.text(x[i] + width / 2, v + 0.8, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    for i in range(len(scenarios)):
        gap = round_vals[i] - trunc_vals[i]
        mid_y = (round_vals[i] + trunc_vals[i]) / 2
        ax.annotate(f"Δ {gap:.2f} pp", xy=(x[i] + width / 2 + 0.05, mid_y),
                    fontsize=8, color="gray", style="italic")

    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 82)
    ax.set_title("Round vs Truncation: Exact Match Across All Scenarios")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_round_vs_trunc_summary.pdf")
    plt.close(fig)
    print(f"  Saved fig_round_vs_trunc_summary.pdf")


# ── Graph 8: Computation Savings ─────────────────────────────────────────────

def graph_computation_savings(output_dir):
    labels = ["bf16-round\nFixed 16", "bf16-round\nHaltable", "bf16-trunc\nFixed 16", "bf16-trunc\nHaltable"]
    exact_match = [72.36, 72.20, 64.85, 64.76]
    avg_steps = [16.0, 6.59, 16.0, 7.84]
    colors = [COLORS["round"], COLORS["round"], COLORS["trunc"], COLORS["trunc"]]
    alphas = [1.0, 0.65, 1.0, 0.65]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.5

    for i in range(len(labels)):
        ax1.bar(x[i], exact_match[i], width, color=colors[i], edgecolor="black",
                linewidth=0.5, alpha=alphas[i])

    ax1.set_ylabel("Exact Match Accuracy (%)", color="black")
    ax1.set_ylim(60, 76)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, avg_steps, "ko-", markersize=8, linewidth=2, label="Avg ACT Steps")
    for i, s in enumerate(avg_steps):
        ax2.annotate(f"{s:.1f}", (x[i], s), textcoords="offset points",
                     xytext=(12, 5), fontsize=9, fontweight="bold")
    ax2.set_ylabel("Average ACT Steps", color="black")
    ax2.set_ylim(0, 20)

    for i, val in enumerate(exact_match):
        ax1.text(x[i], val + 0.3, f"{val:.2f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title("Computation-Accuracy Trade-off (Full Dataset)")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_computation_savings.pdf")
    plt.close(fig)
    print(f"  Saved fig_computation_savings.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dissertation figures")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures to {output_dir}\n")

    print("[1/10] Paper baseline comparison...")
    graph_baseline_comparison(output_dir)

    print("[2/10] Haltable vs fixed...")
    graph_haltable_vs_fixed(output_dir)

    print("[3/10] ACT steps distribution...")
    graph_act_steps_distribution(output_dir)

    print("[4/10] Platform congruence (round)...")
    graph_congruence(output_dir, "round")

    print("[5/10] Platform congruence (trunc)...")
    graph_congruence(output_dir, "trunc")

    print("[6/10] Token accuracy scatter (round)...")
    graph_token_accuracy_scatter(output_dir, "round")

    print("[7/10] Token accuracy scatter (trunc)...")
    graph_token_accuracy_scatter(output_dir, "trunc")

    print("[8/10] ESP32 timing distribution...")
    graph_esp32_timing(output_dir)

    print("[9/10] Round vs trunc summary...")
    graph_round_vs_trunc_summary(output_dir)

    print("[10/10] Computation savings...")
    graph_computation_savings(output_dir)

    print("\nDone! All figures saved to", output_dir)


if __name__ == "__main__":
    main()

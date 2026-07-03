# Results directory map

Definitive evaluation wave (the results reported in the dissertation), produced from
checkpoints `bitnet_round_v3/step_58590`, `bitnet_trunc_v3/step_65100`, and
`float_paper_repro/step_65100` on the 200-puzzle test subset (`test_subset.json`,
chunks `test_subset_chunk_*.json`):

| Directory | Contents |
|---|---|
| `round_v3/` | PC (PyTorch) evaluation of the bf16-round ternary model: fixed 1/16 ACT steps and haltable modes |
| `trunc_v3/` | PC evaluation of the bf16-trunc ternary model, same protocol |
| `float_baseline/` | PC evaluation of the float32 baseline, same protocol |
| `round_v3_esp32/` | ESP32-S3 serial evaluation of the round model (four boards, chunks a-d) |
| `trunc_v3_esp32/` | ESP32-S3 serial evaluation of the trunc model (four boards, chunks a-d) |
| `round_v3_mismatch/` | PC evaluation of the round checkpoint with truncation-mode activation quantization (deployment-mismatch experiment) |
| `benchmark_float32_*.json` (this directory's root) | Fifth-board float32 matmul timing benchmarks (kernel comparison) |

Aggregate fields inside JSONs from resumed serial runs are unreliable; step averages
must be recomputed from the per-puzzle records.

## Superseded (first-generation runs, kept for history; do not use)

`round_esp32/`, `trunc_esp32/`, `fp32_esp32/`, `fp32_subset/`, `fp32_subset_haltable/`,
`bf16_subset_haltable/`, `no_puzzle_emb/`, `round_trunc_mismatch/` (superseded by
`round_v3_mismatch/`), and the loose `results_*.json` files in this directory. These
came from earlier checkpoints or earlier evaluation-protocol iterations and do not
match the dissertation's tables.

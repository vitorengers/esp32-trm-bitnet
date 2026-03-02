/**
 * TRM Model Configuration Header
 * 
 * Auto-generated dimensions matching the PyTorch TRM architecture.
 * Config source: config/arch/trm.yaml
 * 
 * Architecture: TinyRecursiveReasoningModel_ACTV1
 *   - hidden_size: 512
 *   - num_heads: 8 (MHA, no GQA)
 *   - head_dim: 64
 *   - SwiGLU expansion: 4 → inter = round(4 * 512 * 2/3) rounded up to 256 = 1536
 *   - L_layers: 2 (transformer blocks per reasoning cycle)
 *   - L_cycles: 6 (reasoning cycles per step)
 *   - H_cycles: 3 (high-level cycles)
 *   - halt_max_steps: 16 (ACT maximum reasoning steps)
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

/* ============== Model Dimensions ============== */
#define HIDDEN_SIZE         512
#define VOCAB_SIZE          11      /* Sudoku tokens (0=pad/blank, 1-9=digits, 10=ignore) */
#define NUM_HEADS           8
#define HEAD_DIM            64      /* HIDDEN_SIZE / NUM_HEADS */
#define NUM_KV_HEADS        8       /* MHA: same as NUM_HEADS */
#define MLP_INTER           1536    /* round(4 * 512 * 2/3) rounded to 256 */

/* QKV projection output size: (NUM_HEADS + 2*NUM_KV_HEADS) * HEAD_DIM */
#define QKV_OUT_SIZE        1536    /* (8 + 2*8) * 64 */
#define ATTN_OUT_SIZE       512     /* NUM_HEADS * HEAD_DIM */

/* SwiGLU: gate_up_proj outputs 2 * MLP_INTER, then split */
#define GATE_UP_SIZE        3072    /* 2 * MLP_INTER */

/* ============== Architecture ============== */
#define L_LAYERS            2       /* Transformer blocks per reasoning cycle */
#define L_CYCLES            6       /* Reasoning cycles per ACT step */
#define H_CYCLES            3       /* High-level cycles */
#define HALT_MAX_STEPS      16      /* ACT maximum reasoning steps */

/* ============== Puzzle Embedding ============== */
#define PUZZLE_EMB_LEN      16      /* Pre-padded puzzle embedding positions */
#define PUZZLE_EMB_HIDDEN   512     /* = HIDDEN_SIZE */

/* ============== Ternary Packing ============== */
/* 4 ternary weights packed per byte (2 bits each) */
/* Values: 0b00 = -1, 0b01 = 0, 0b10 = +1 */
#define WEIGHTS_PER_BYTE    4

/* ============== RMSNorm ============== */
#define RMS_NORM_EPS        1e-5f

/* ============== RoPE ============== */
#define ROPE_THETA          10000.0f

/* ============== Memory Layout ============== */
/* Maximum sequence length for RoPE precomputation */
#define MAX_SEQ_LEN         256

/* ============== Inference ============== */
#define BATCH_SIZE          1       /* Single-sample inference on MCU */

#endif /* MODEL_CONFIG_H */

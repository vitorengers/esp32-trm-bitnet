/**
 * TRM Inference Engine for ESP32-S3
 * 
 * Runs one transformer block (attention + SwiGLU MLP) with residual connections.
 */

#ifndef TRM_ENGINE_H
#define TRM_ENGINE_H

#include "model_loader.h"

/**
 * Run a single transformer block forward pass (Post-Norm).
 * 
 * Computes: x = RMSNorm(x + Attention(x))
 *           x = RMSNorm(x + MLP(x))
 * 
 * @param output    Output [seq_len, HIDDEN_SIZE]
 * @param input     Input [seq_len, HIDDEN_SIZE] (may alias output for in-place)
 * @param model     Model weights
 * @param block_idx Which transformer block (0..L_LAYERS-1)
 * @param seq_len   Sequence length
 * @param scratch   Scratch buffer
 */
void trm_block_forward(
    float *output,
    const float *input,
    const TRMModel *model,
    int block_idx,
    int seq_len,
    float *scratch
);

/**
 * Run one reasoning step (L_CYCLES iterations of L_LAYERS blocks).
 * This is the inner loop of the recursive reasoning.
 * 
 * @param hidden     Hidden state [seq_len, HIDDEN_SIZE], modified in-place
 * @param model      Model weights
 * @param seq_len    Sequence length
 * @param scratch    Scratch buffer
 */
void trm_reasoning_step(
    float *hidden,
    const TRMModel *model,
    int seq_len,
    float *scratch
);

/**
 * Returns the scratch buffer size needed for inference.
 */
int trm_scratch_size(int seq_len);

/**
 * Run ternary matmul benchmark (standalone kernel test).
 * @param iterations Number of iterations to run
 */
void trm_benchmark_ternary_matmul(int iterations);

/* ============== Full Inference Pipeline ============== */

/**
 * Embedding lookup: convert token IDs to float hidden states.
 * Dequantizes INT8 embeddings and scales by sqrt(HIDDEN_SIZE).
 * 
 * @param output  Output [seq_len, HIDDEN_SIZE] float
 * @param tokens  Input token IDs [seq_len] (values 0..VOCAB_SIZE-1)
 * @param seq_len Number of tokens
 * @param model   Model weights (for embed_tokens, embed_scale)
 */
void trm_embed(
    float *output,
    const uint8_t *tokens,
    int seq_len,
    const TRMModel *model
);

/**
 * L_level call: hidden = L_LAYERS blocks(hidden + injection).
 * Adds injection to hidden in-place, then runs all L_LAYERS blocks.
 * 
 * @param hidden    Hidden state [seq_len, HIDDEN_SIZE], modified in-place
 * @param injection Injection vector [seq_len, HIDDEN_SIZE] to add before blocks
 * @param model     Model weights
 * @param seq_len   Sequence length
 * @param scratch   Scratch buffer (from trm_scratch_size)
 */
void trm_l_level(
    float *hidden,
    const float *injection,
    const TRMModel *model,
    int seq_len,
    float *scratch
);

/**
 * Output head: project hidden states to logits via lm_head, then argmax.
 * 
 * @param pred_tokens  Output predicted token IDs [seq_len]
 * @param hidden       Hidden state [seq_len, HIDDEN_SIZE]
 * @param seq_len      Sequence length
 * @param model        Model weights (for lm_head, lm_head_scale)
 */
void trm_output_head(
    uint8_t *pred_tokens,
    const float *hidden,
    int seq_len,
    const TRMModel *model
);

/**
 * Full end-to-end inference: embed -> dual-z reasoning -> output head.
 * Implements the complete TRM inner forward pass matching PyTorch:
 *   input_emb = embed(tokens) * sqrt(hidden_size)
 *   z_H = H_init, z_L = L_init
 *   for h in H_CYCLES:
 *       for l in L_CYCLES: z_L = L_level(z_L, z_H + input_emb)
 *       z_H = L_level(z_H, z_L)
 *   pred = argmax(lm_head(z_H))
 * 
 * @param pred_tokens   Output predicted token IDs [seq_len]
 * @param input_tokens  Input token IDs [seq_len]
 * @param seq_len       Sequence length
 * @param model         Model weights
 * @param scratch       Scratch buffer (from trm_scratch_size)
 * @return              Inference time in microseconds
 */
int64_t trm_full_inference(
    uint8_t *pred_tokens,
    const uint8_t *input_tokens,
    int seq_len,
    const TRMModel *model,
    float *scratch
);

/**
 * Full inference with ACT (Adaptive Computation Time) halting.
 * Runs up to max_act_steps independent inner forward passes.
 * After each step, checks q_head: if q_halt > 0, stops early.
 * Each ACT step re-initializes z_H/z_L (matching PyTorch ACT wrapper
 * where carry is reset when halted=True).
 * 
 * @param pred_tokens     Output predicted token IDs [seq_len]
 * @param input_tokens    Input token IDs [seq_len]
 * @param seq_len         Sequence length
 * @param model           Model weights (must have q_head for halting)
 * @param scratch         Scratch buffer (from trm_scratch_size)
 * @param max_act_steps   Maximum ACT steps (1..HALT_MAX_STEPS)
 * @param steps_used      Output: how many steps were actually used
 * @return                Inference time in microseconds
 */
int64_t trm_full_inference_act(
    uint8_t *pred_tokens,
    const uint8_t *input_tokens,
    int seq_len,
    const TRMModel *model,
    float *scratch,
    int max_act_steps,
    int *steps_used
);

/**
 * Returns the total PSRAM memory needed for full inference buffers.
 * Includes z_H, z_L, input_emb, injection_buf, and scratch.
 * 
 * @param seq_len  Sequence length
 * @return         Total bytes needed
 */
int trm_full_inference_mem(int seq_len);

#endif /* TRM_ENGINE_H */

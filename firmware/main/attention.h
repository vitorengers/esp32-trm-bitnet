/**
 * Multi-Head Attention for ESP32-S3
 * 
 * Implements QKV projection, RoPE, scaled dot-product attention, and output projection.
 * Uses the ternary matmul kernel for all linear projections.
 */

#ifndef ATTENTION_H
#define ATTENTION_H

#include "model_config.h"
#include <stdint.h>

/**
 * Layer weights for one attention module.
 */
typedef struct {
    const uint8_t *qkv_weights;   /* Packed ternary [QKV_OUT_SIZE, HIDDEN_SIZE / 4] */
    float qkv_scale;
    const uint8_t *o_weights;     /* Packed ternary [HIDDEN_SIZE, ATTN_OUT_SIZE / 4] */
    float o_scale;
    /* Pre-unpacked INT8 weights (optional, allocated by 'u' command) */
    int8_t *qkv_unpacked;        /* [QKV_OUT_SIZE, HIDDEN_SIZE] as INT8 {-1,0,+1} */
    int8_t *o_unpacked;          /* [HIDDEN_SIZE, ATTN_OUT_SIZE] as INT8 */
} AttentionWeights;

/**
 * Precomputed RoPE cos/sin values.
 */
typedef struct {
    float *cos;   /* [MAX_SEQ_LEN, HEAD_DIM] */
    float *sin;   /* [MAX_SEQ_LEN, HEAD_DIM] */
} RoPECache;

/**
 * Initialize RoPE cache (call once at startup).
 * @param cache   RoPE cache to initialize
 */
void rope_init(RoPECache *cache);

/**
 * Free RoPE cache.
 */
void rope_free(RoPECache *cache);

/**
 * Run multi-head attention with BitNet per-layer RMSNorm.
 * 
 * @param output           Output [seq_len, HIDDEN_SIZE]
 * @param input            Input [seq_len, HIDDEN_SIZE]
 * @param weights          Attention layer weights
 * @param qkv_norm_weight  BitNet RMSNorm weight for QKV projection [HIDDEN_SIZE], or NULL
 * @param o_norm_weight    BitNet RMSNorm weight for O projection [HIDDEN_SIZE], or NULL
 * @param rope             Precomputed RoPE cache
 * @param seq_len          Sequence length
 * @param scratch          Scratch buffer (must be at least attn_scratch_size() bytes)
 */
void attention_forward(
    float *output,
    const float *input,
    const AttentionWeights *weights,
    const float *qkv_norm_weight,
    const float *o_norm_weight,
    const RoPECache *rope,
    int seq_len,
    float *scratch
);

/**
 * Returns the scratch buffer size needed for attention.
 */
int attn_scratch_size(int seq_len);

#endif /* ATTENTION_H */

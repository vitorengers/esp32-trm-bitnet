/**
 * Model Weight Loader for ESP32-S3
 * 
 * Loads packed ternary weights from flash (SPIFFS partition) into PSRAM.
 */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "model_config.h"
#include "attention.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * Complete model weights structure.
 * All weight pointers point into PSRAM.
 */
typedef struct {
    /* Embedding: stored as INT8 with scale (not ternary) */
    int8_t *embed_tokens;       /* [vocab_size, HIDDEN_SIZE] */
    float embed_scale;
    int vocab_size;

    /* Puzzle embedding: float32, pre-padded [PUZZLE_EMB_LEN, HIDDEN_SIZE] */
    float *puzzle_emb;          /* [PUZZLE_EMB_LEN, HIDDEN_SIZE] float32 */

    /* L_LAYERS transformer blocks */
    struct {
        AttentionWeights attn;

        /* BitNet per-layer RMSNorm learned weights (float32) */
        float *qkv_norm_weight;     /* [HIDDEN_SIZE] */
        float *o_norm_weight;       /* [HIDDEN_SIZE] */
        float *gate_up_norm_weight; /* [HIDDEN_SIZE] */
        float *down_norm_weight;    /* [MLP_INTER] */

        /* SwiGLU MLP */
        uint8_t *gate_up_weights;   /* Packed ternary [GATE_UP_SIZE, HIDDEN_SIZE / 4] */
        float gate_up_scale;
        uint8_t *down_weights;      /* Packed ternary [HIDDEN_SIZE, MLP_INTER / 4] */
        float down_scale;
        /* Pre-unpacked INT8 weights (optional, allocated by 'u' command) */
        int8_t *gate_up_unpacked;   /* [GATE_UP_SIZE, HIDDEN_SIZE] as INT8 */
        int8_t *down_unpacked;      /* [HIDDEN_SIZE, MLP_INTER] as INT8 */
    } blocks[L_LAYERS];

    /* Output head: stored as INT8 with scale (not ternary) */
    int8_t *lm_head;            /* [vocab_size, HIDDEN_SIZE] */
    float lm_head_scale;

    /* ACT halt head (q_head): linear [2, HIDDEN_SIZE] + bias [2] */
    float *q_head_weight;       /* [2, HIDDEN_SIZE] float32, NULL if v3 or earlier */
    float *q_head_bias;         /* [2] float32, NULL if v3 or earlier */

    /* Initial carry states for dual-z reasoning */
    float *h_init;              /* [HIDDEN_SIZE] initial z_H state */
    float *l_init;              /* [HIDDEN_SIZE] initial z_L state */

    /* RoPE cache */
    RoPECache rope;

    /* Total bytes allocated in PSRAM */
    size_t total_allocated;

    /* Pre-unpack state */
    bool weights_unpacked;        /* true if INT8 unpacked weights available */
    size_t unpacked_allocated;    /* bytes used by unpacked weights */
} TRMModel;

/**
 * Load model weights from flash partition into PSRAM.
 * @param model     Model structure to populate
 * @param partition_label   Name of the SPIFFS partition (e.g., "model")
 * @return 0 on success, -1 on error
 */
int model_load(TRMModel *model, const char *partition_label);

/**
 * Pre-unpack all ternary weights from 2-bit packed → INT8 in PSRAM.
 * Allocates ~6.50 MB of additional PSRAM.
 * @return 0 on success, -1 on failure (out of memory)
 */
int model_unpack_weights(TRMModel *model);

/**
 * Free pre-unpacked weight buffers, restore to packed-only state.
 */
void model_free_unpacked(TRMModel *model);

/**
 * Free all model memory.
 */
void model_free(TRMModel *model);

/**
 * Print model memory usage statistics.
 */
void model_print_stats(const TRMModel *model);

#endif /* MODEL_LOADER_H */

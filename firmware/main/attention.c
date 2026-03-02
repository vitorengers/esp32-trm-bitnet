/**
 * Multi-Head Attention for ESP32-S3
 */

#include "attention.h"
#include "trm_ternary.h"
#include "rmsnorm.h"
#include "model_config.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "esp_heap_caps.h"

/* ============== RoPE ============== */

void rope_init(RoPECache *cache)
{
    cache->cos = (float *)heap_caps_malloc(MAX_SEQ_LEN * HEAD_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    cache->sin = (float *)heap_caps_malloc(MAX_SEQ_LEN * HEAD_DIM * sizeof(float), MALLOC_CAP_SPIRAM);

    for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / (float)HEAD_DIM);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            /* Store as [pos, head_dim] with repeated cos/sin for rotate_half pattern */
            cache->cos[pos * HEAD_DIM + i] = c;
            cache->cos[pos * HEAD_DIM + HEAD_DIM / 2 + i] = c;
            cache->sin[pos * HEAD_DIM + i] = s;
            cache->sin[pos * HEAD_DIM + HEAD_DIM / 2 + i] = s;
        }
    }
}

void rope_free(RoPECache *cache)
{
    if (cache->cos) heap_caps_free(cache->cos);
    if (cache->sin) heap_caps_free(cache->sin);
    cache->cos = NULL;
    cache->sin = NULL;
}

/* Apply RoPE to a single head vector [HEAD_DIM] at position pos */
static void apply_rope(float *q, const RoPECache *rope, int pos)
{
    const float *c = rope->cos + pos * HEAD_DIM;
    const float *s = rope->sin + pos * HEAD_DIM;

    /* rotate_half: x1 = q[0..HD/2], x2 = q[HD/2..HD]
     * result = q * cos + (-x2, x1) * sin */
    float tmp[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        tmp[i] = -q[HEAD_DIM / 2 + i];
        tmp[HEAD_DIM / 2 + i] = q[i];
    }
    for (int i = 0; i < HEAD_DIM; i++) {
        q[i] = q[i] * c[i] + tmp[i] * s[i];
    }
}

/* ============== Softmax ============== */

static void softmax(float *x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/* ============== Attention Forward ============== */

int attn_scratch_size(int seq_len)
{
    /* BitNet norm buffer + QKV buffer + attention scores + attention output */
    int norm_buf = seq_len * HIDDEN_SIZE;   /* for BitNet RMSNorm before projections */
    int qkv_buf = seq_len * QKV_OUT_SIZE;
    int attn_scores = NUM_HEADS * seq_len * seq_len;
    int attn_out = seq_len * ATTN_OUT_SIZE;
    return (norm_buf + qkv_buf + attn_scores + attn_out) * sizeof(float);
}

void attention_forward(
    float *output,
    const float *input,
    const AttentionWeights *weights,
    const float *qkv_norm_weight,
    const float *o_norm_weight,
    const RoPECache *rope,
    int seq_len,
    float *scratch)
{
    /* Layout pointers in scratch */
    float *norm_buf = scratch;
    float *qkv = norm_buf + seq_len * HIDDEN_SIZE;
    float *scores = qkv + seq_len * QKV_OUT_SIZE;
    float *attn_out = scores + NUM_HEADS * seq_len * seq_len;

    /* Step 1: BitNet RMSNorm before QKV projection */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_weighted(
            norm_buf + s * HIDDEN_SIZE,
            input + s * HIDDEN_SIZE,
            qkv_norm_weight,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }

    /* Step 2: QKV projection using ternary matmul on normed input */
#ifdef USE_FLOAT32_MATMUL
    if (weights->qkv_unpacked) {
        float32_matmul_from_int8(
            qkv, norm_buf, weights->qkv_unpacked,
            weights->qkv_scale,
            seq_len, QKV_OUT_SIZE, HIDDEN_SIZE
        );
    } else {
        float32_matmul_from_packed(
            qkv, norm_buf, weights->qkv_weights,
            weights->qkv_scale,
            seq_len, QKV_OUT_SIZE, HIDDEN_SIZE
        );
    }
#else
    if (weights->qkv_unpacked) {
        ternary_matmul_batched_simd_preunpacked(
            qkv, norm_buf, weights->qkv_unpacked,
            weights->qkv_scale,
            seq_len, QKV_OUT_SIZE, HIDDEN_SIZE
        );
    } else {
        ternary_matmul_batched_simd(
            qkv, norm_buf, weights->qkv_weights,
            weights->qkv_scale,
            seq_len, QKV_OUT_SIZE, HIDDEN_SIZE
        );
    }
#endif

    /* Step 2: Split Q, K, V and apply RoPE to Q and K */
    /* QKV layout: [seq_len, (NUM_HEADS + 2*NUM_KV_HEADS) * HEAD_DIM] */
    /* Q: [0, NUM_HEADS * HEAD_DIM)  */
    /* K: [NUM_HEADS * HEAD_DIM, (NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM) */
    /* V: [(NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM, QKV_OUT_SIZE) */
    const int q_offset = 0;
    const int k_offset = NUM_HEADS * HEAD_DIM;
    const int v_offset = (NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM;

    /* Apply RoPE to Q and K for each position */
    for (int pos = 0; pos < seq_len; pos++) {
        float *qkv_pos = qkv + pos * QKV_OUT_SIZE;
        /* Apply to each Q head */
        for (int h = 0; h < NUM_HEADS; h++) {
            apply_rope(qkv_pos + q_offset + h * HEAD_DIM, rope, pos);
        }
        /* Apply to each K head */
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            apply_rope(qkv_pos + k_offset + h * HEAD_DIM, rope, pos);
        }
    }

    /* Step 3: Compute attention scores and apply softmax per head */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; h++) {
        int kv_h = h;  /* MHA: 1-to-1 mapping */

        /* Compute Q @ K^T for this head */
        for (int qi = 0; qi < seq_len; qi++) {
            const float *q_vec = qkv + qi * QKV_OUT_SIZE + q_offset + h * HEAD_DIM;
            float *score_row = scores + h * seq_len * seq_len + qi * seq_len;

            for (int ki = 0; ki < seq_len; ki++) {
                const float *k_vec = qkv + ki * QKV_OUT_SIZE + k_offset + kv_h * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                score_row[ki] = dot * scale;
            }
            /* Softmax over keys (no causal mask for inference) */
            softmax(score_row, seq_len);
        }

        /* Step 4: Weighted sum of values */
        for (int qi = 0; qi < seq_len; qi++) {
            float *out_vec = attn_out + qi * ATTN_OUT_SIZE + h * HEAD_DIM;
            const float *score_row = scores + h * seq_len * seq_len + qi * seq_len;

            memset(out_vec, 0, HEAD_DIM * sizeof(float));
            for (int ki = 0; ki < seq_len; ki++) {
                const float *v_vec = qkv + ki * QKV_OUT_SIZE + v_offset + kv_h * HEAD_DIM;
                float w = score_row[ki];
                for (int d = 0; d < HEAD_DIM; d++) {
                    out_vec[d] += w * v_vec[d];
                }
            }
        }
    }

    /* Step 5: BitNet RMSNorm before output projection */
    /* Reuse norm_buf for the O projection norm (input dim is ATTN_OUT_SIZE = HIDDEN_SIZE) */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_weighted(
            norm_buf + s * ATTN_OUT_SIZE,
            attn_out + s * ATTN_OUT_SIZE,
            o_norm_weight,
            ATTN_OUT_SIZE,
            RMS_NORM_EPS
        );
    }

    /* Step 6: Output projection using ternary matmul on normed attention output */
#ifdef USE_FLOAT32_MATMUL
    if (weights->o_unpacked) {
        float32_matmul_from_int8(
            output, norm_buf, weights->o_unpacked,
            weights->o_scale,
            seq_len, HIDDEN_SIZE, ATTN_OUT_SIZE
        );
    } else {
        float32_matmul_from_packed(
            output, norm_buf, weights->o_weights,
            weights->o_scale,
            seq_len, HIDDEN_SIZE, ATTN_OUT_SIZE
        );
    }
#else
    if (weights->o_unpacked) {
        ternary_matmul_batched_simd_preunpacked(
            output, norm_buf, weights->o_unpacked,
            weights->o_scale,
            seq_len, HIDDEN_SIZE, ATTN_OUT_SIZE
        );
    } else {
        ternary_matmul_batched_simd(
            output, norm_buf, weights->o_weights,
            weights->o_scale,
            seq_len, HIDDEN_SIZE, ATTN_OUT_SIZE
        );
    }
#endif
}

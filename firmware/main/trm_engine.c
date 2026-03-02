/**
 * TRM Inference Engine for ESP32-S3
 */

#include "trm_engine.h"
#include "trm_ternary.h"
#include "rmsnorm.h"
#include "attention.h"
#include "benchmark.h"
#include "model_config.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static const char *TAG = "trm_engine";

/* ============== SiLU Activation ============== */
static void silu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* ============== Transformer Block ============== */

void trm_block_forward(
    float *output,
    const float *input,
    const TRMModel *model,
    int block_idx,
    int seq_len,
    float *scratch)
{
    const int hidden_bytes = seq_len * HIDDEN_SIZE * sizeof(float);
    const int total_hidden = seq_len * HIDDEN_SIZE;

    /* Layout scratch:
     * [0] temp_buf: seq_len * HIDDEN_SIZE  (for sublayer output / residual sum)
     * [1] attn_out: seq_len * HIDDEN_SIZE
     * [2] mlp_buf:  seq_len * GATE_UP_SIZE (largest intermediate)
     * [3] mlp_norm_buf: seq_len * MLP_INTER (for BitNet RMSNorm before down_proj;
     *                   also reused for gate_up norm since HIDDEN_SIZE <= MLP_INTER)
     * [4] attn_scratch: for attention internal use
     */
    float *temp_buf = scratch;
    float *attn_out = temp_buf + seq_len * HIDDEN_SIZE;
    float *mlp_buf = attn_out + seq_len * HIDDEN_SIZE;
    float *mlp_norm_buf = mlp_buf + seq_len * GATE_UP_SIZE;
    float *attn_scratch = mlp_norm_buf + seq_len * MLP_INTER;

    /* Copy input to output as working buffer */
    if (output != input) {
        memcpy(output, input, hidden_bytes);
    }

    /*
     * POST-NORM architecture (matches PyTorch TRM):
     *   h = rms_norm(h + self_attn(h))         -- attn on raw h
     *   h = rms_norm(h + mlp(h))               -- mlp on normed h
     *
     * BitNet: each ternary matmul inside attention and MLP is preceded by
     * a learned RMSNorm (norm_weight). This is handled inside attention_forward
     * and the MLP section below.
     */

    /* === Attention sub-layer (Post-Norm) === */
    /* 1. Attention on RAW input (no pre-norm!) */
    /* BitNet RMSNorm before QKV and O projections handled inside attention_forward */
    attention_forward(
        attn_out, output,
        &model->blocks[block_idx].attn,
        model->blocks[block_idx].qkv_norm_weight,
        model->blocks[block_idx].o_norm_weight,
        &model->rope,
        seq_len,
        attn_scratch
    );

    /* 2. Residual add: temp = output + attn_out */
    for (int i = 0; i < total_hidden; i++) {
        temp_buf[i] = output[i] + attn_out[i];
    }

    /* 3. Post-RMSNorm: output = RMSNorm(temp) */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_out(
            output + s * HIDDEN_SIZE,
            temp_buf + s * HIDDEN_SIZE,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }

    /* === MLP sub-layer (Post-Norm) === */
    /* 1. BitNet RMSNorm before gate_up projection */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_weighted(
            mlp_norm_buf + s * HIDDEN_SIZE,
            output + s * HIDDEN_SIZE,
            model->blocks[block_idx].gate_up_norm_weight,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }

    /* 2. SwiGLU gate_up on the normed output */
#ifdef USE_FLOAT32_MATMUL
    if (model->blocks[block_idx].gate_up_unpacked) {
        float32_matmul_from_int8(
            mlp_buf, mlp_norm_buf,
            model->blocks[block_idx].gate_up_unpacked,
            model->blocks[block_idx].gate_up_scale,
            seq_len, GATE_UP_SIZE, HIDDEN_SIZE
        );
    } else {
        float32_matmul_from_packed(
            mlp_buf, mlp_norm_buf,
            model->blocks[block_idx].gate_up_weights,
            model->blocks[block_idx].gate_up_scale,
            seq_len, GATE_UP_SIZE, HIDDEN_SIZE
        );
    }
#else
    if (model->blocks[block_idx].gate_up_unpacked) {
        ternary_matmul_batched_simd_preunpacked(
            mlp_buf, mlp_norm_buf,
            model->blocks[block_idx].gate_up_unpacked,
            model->blocks[block_idx].gate_up_scale,
            seq_len, GATE_UP_SIZE, HIDDEN_SIZE
        );
    } else {
        ternary_matmul_batched_simd(
            mlp_buf, mlp_norm_buf,
            model->blocks[block_idx].gate_up_weights,
            model->blocks[block_idx].gate_up_scale,
            seq_len, GATE_UP_SIZE, HIDDEN_SIZE
        );
    }
#endif

    /* 3. Split gate and up, apply SiLU, multiply */
    for (int s = 0; s < seq_len; s++) {
        float *gate = mlp_buf + s * GATE_UP_SIZE;
        float *up = gate + MLP_INTER;

        /* SiLU on gate */
        silu_inplace(gate, MLP_INTER);

        /* gate * up (element-wise) — result in first MLP_INTER elements */
        for (int i = 0; i < MLP_INTER; i++) {
            gate[i] *= up[i];
        }
    }

    /* 4. BitNet RMSNorm before down projection */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_weighted(
            mlp_norm_buf + s * MLP_INTER,
            mlp_buf + s * GATE_UP_SIZE,  /* first MLP_INTER floats of gate_up output */
            model->blocks[block_idx].down_norm_weight,
            MLP_INTER,
            RMS_NORM_EPS
        );
    }

    /* 5. Down projection on normed intermediate: down_out stored in temp_buf */
#ifdef USE_FLOAT32_MATMUL
    if (model->blocks[block_idx].down_unpacked) {
        float32_matmul_from_int8(
            temp_buf, mlp_norm_buf,
            model->blocks[block_idx].down_unpacked,
            model->blocks[block_idx].down_scale,
            seq_len, HIDDEN_SIZE, MLP_INTER
        );
    } else {
        float32_matmul_from_packed(
            temp_buf, mlp_norm_buf,
            model->blocks[block_idx].down_weights,
            model->blocks[block_idx].down_scale,
            seq_len, HIDDEN_SIZE, MLP_INTER
        );
    }
#else
    if (model->blocks[block_idx].down_unpacked) {
        ternary_matmul_batched_simd_preunpacked(
            temp_buf, mlp_norm_buf,
            model->blocks[block_idx].down_unpacked,
            model->blocks[block_idx].down_scale,
            seq_len, HIDDEN_SIZE, MLP_INTER
        );
    } else {
        ternary_matmul_batched_simd(
            temp_buf, mlp_norm_buf,
            model->blocks[block_idx].down_weights,
            model->blocks[block_idx].down_scale,
            seq_len, HIDDEN_SIZE, MLP_INTER
        );
    }
#endif

    /* 6. Residual add: temp = output + down_out (in-place in temp_buf) */
    for (int i = 0; i < total_hidden; i++) {
        temp_buf[i] += output[i];
    }

    /* 7. Post-RMSNorm: output = RMSNorm(temp) */
    for (int s = 0; s < seq_len; s++) {
        rmsnorm_out(
            output + s * HIDDEN_SIZE,
            temp_buf + s * HIDDEN_SIZE,
            HIDDEN_SIZE,
            RMS_NORM_EPS
        );
    }
}

/* ============== Reasoning Step ============== */

void trm_reasoning_step(
    float *hidden,
    const TRMModel *model,
    int seq_len,
    float *scratch)
{
    for (int cycle = 0; cycle < L_CYCLES; cycle++) {
        for (int layer = 0; layer < L_LAYERS; layer++) {
            trm_block_forward(hidden, hidden, model, layer, seq_len, scratch);
        }
    }
}

/* ============== Scratch Size ============== */

int trm_scratch_size(int seq_len)
{
    int norm_buf = seq_len * HIDDEN_SIZE;       /* temp_buf */
    int attn_out = seq_len * HIDDEN_SIZE;       /* attn_out */
    int mlp_buf = seq_len * GATE_UP_SIZE;       /* mlp_buf */
    int mlp_norm = seq_len * MLP_INTER;         /* mlp_norm_buf (BitNet RMSNorm) */
    int attn_scr = attn_scratch_size(seq_len) / sizeof(float);
    return (norm_buf + attn_out + mlp_buf + mlp_norm + attn_scr) * sizeof(float);
}

/* ============== Embedding Lookup ============== */

void trm_embed(
    float *output,
    const uint8_t *tokens,
    int seq_len,
    const TRMModel *model)
{
    /*
     * PyTorch: embedding = embed_tokens(input) * sqrt(hidden_size)
     * With puzzle_emb prepending:
     *   1. First PUZZLE_EMB_LEN positions = puzzle_emb (pre-padded, already float32)
     *   2. Next seq_len positions = embed_tokens(input) dequantized from INT8
     *   3. Scale entire output by sqrt(HIDDEN_SIZE)
     *
     * Output layout: [PUZZLE_EMB_LEN + seq_len, HIDDEN_SIZE]
     */
    const float inv_scale = 1.0f / model->embed_scale;
    const float sqrt_d = sqrtf((float)HIDDEN_SIZE);
    const float embed_factor = inv_scale * sqrt_d;

    /* 1. Prepend puzzle_emb (scaled by sqrt(d)) */
    if (model->puzzle_emb) {
        for (int s = 0; s < PUZZLE_EMB_LEN; s++) {
            const float *src = model->puzzle_emb + s * HIDDEN_SIZE;
            float *dst = output + s * HIDDEN_SIZE;
            for (int d = 0; d < HIDDEN_SIZE; d++) {
                dst[d] = src[d] * sqrt_d;
            }
        }
    } else {
        memset(output, 0, PUZZLE_EMB_LEN * HIDDEN_SIZE * sizeof(float));
    }

    /* 2. Embed input tokens after puzzle positions */
    float *token_output = output + PUZZLE_EMB_LEN * HIDDEN_SIZE;
    for (int s = 0; s < seq_len; s++) {
        int token_id = (int)tokens[s];
        if (token_id >= model->vocab_size) token_id = 0;  /* safety clamp */

        const int8_t *row = model->embed_tokens + token_id * HIDDEN_SIZE;
        float *out_row = token_output + s * HIDDEN_SIZE;

        for (int d = 0; d < HIDDEN_SIZE; d++) {
            out_row[d] = (float)row[d] * embed_factor;
        }
    }
}

/* ============== L_level (Reasoning Module) ============== */

void trm_l_level(
    float *hidden,
    const float *injection,
    const TRMModel *model,
    int seq_len,
    float *scratch)
{
    /*
     * PyTorch ReasoningModule.forward(hidden_states, input_injection):
     *   hidden_states = hidden_states + input_injection
     *   for layer in self.layers:
     *       hidden_states = layer(hidden_states)
     *   return hidden_states
     */
    const int total = seq_len * HIDDEN_SIZE;

    /* Add injection to hidden in-place */
    for (int i = 0; i < total; i++) {
        hidden[i] += injection[i];
    }

    /* Run L_LAYERS blocks */
    for (int layer = 0; layer < L_LAYERS; layer++) {
        trm_block_forward(hidden, hidden, model, layer, seq_len, scratch);
    }
}

/* ============== Output Head ============== */

void trm_output_head(
    uint8_t *pred_tokens,
    const float *hidden,
    int seq_len,
    const TRMModel *model)
{
    /*
     * PyTorch: output = self.lm_head(z_H)  =>  logits = z_H @ lm_head.T
     * lm_head is [vocab_size, HIDDEN_SIZE] stored as INT8 with lm_head_scale
     * logits[s][v] = sum_d(hidden[s][d] * lm_head[v][d]) / scale
     * pred[s] = argmax_v(logits[s][v])
     *
     * Skip the first PUZZLE_EMB_LEN positions (puzzle embedding prefix).
     * hidden has total_seq_len = PUZZLE_EMB_LEN + seq_len positions.
     * pred_tokens has seq_len entries (only the token positions).
     */
    const float inv_scale = 1.0f / model->lm_head_scale;

    /* Skip puzzle positions — start reading from position PUZZLE_EMB_LEN */
    const float *h_start = hidden + PUZZLE_EMB_LEN * HIDDEN_SIZE;

    for (int s = 0; s < seq_len; s++) {
        const float *h = h_start + s * HIDDEN_SIZE;
        int best_token = 0;

        /* Compute logit for token 0 as initial best */
        float best_logit = 0.0f;
        {
            const int8_t *w0 = model->lm_head;
            for (int d = 0; d < HIDDEN_SIZE; d++) {
                best_logit += h[d] * (float)w0[d];
            }
            best_logit *= inv_scale;
        }

        /* Compare with remaining tokens */
        for (int v = 1; v < model->vocab_size; v++) {
            const int8_t *w = model->lm_head + v * HIDDEN_SIZE;
            float logit = 0.0f;
            for (int d = 0; d < HIDDEN_SIZE; d++) {
                logit += h[d] * (float)w[d];
            }
            logit *= inv_scale;

            if (logit > best_logit) {
                best_logit = logit;
                best_token = v;
            }
        }

        pred_tokens[s] = (uint8_t)best_token;
    }
}

/* ============== Full Inference (Dual-z) ============== */

int64_t trm_full_inference(
    uint8_t *pred_tokens,
    const uint8_t *input_tokens,
    int seq_len,
    const TRMModel *model,
    float *scratch)
{
    /*
     * Full TRM inner forward pass matching PyTorch exactly:
     *
     *   input_emb = [puzzle_emb | embed(tokens)] * sqrt(hidden_size)
     *   total_seq = PUZZLE_EMB_LEN + seq_len
     *   z_H = broadcast(H_init, total_seq)
     *   z_L = broadcast(L_init, total_seq)
     *   for h in H_CYCLES:
     *       for l in L_CYCLES:
     *           z_L = L_level(z_L, z_H + input_emb)
     *       z_H = L_level(z_H, z_L)
     *   pred = argmax(lm_head(z_H[PUZZLE_EMB_LEN:]))
     */
    const int total_seq = seq_len + PUZZLE_EMB_LEN;
    const int hidden_floats = total_seq * HIDDEN_SIZE;

    int64_t t_start = esp_timer_get_time();

    /*
     * Memory layout in the caller-provided scratch region:
     *   - First: z_H [total_seq * HIDDEN_SIZE]
     *   - Then:  z_L [total_seq * HIDDEN_SIZE]
     *   - Then:  input_emb [total_seq * HIDDEN_SIZE]
     *   - Then:  injection_buf [total_seq * HIDDEN_SIZE]
     *   - Then:  block_scratch [trm_scratch_size(total_seq)]
     */
    float *z_H = scratch;
    float *z_L = z_H + hidden_floats;
    float *input_emb = z_L + hidden_floats;
    float *injection_buf = input_emb + hidden_floats;
    float *block_scratch = injection_buf + hidden_floats;

    /* 1. Compute input embeddings (prepends puzzle_emb, then token embeddings) */
    trm_embed(input_emb, input_tokens, seq_len, model);

    /* 2. Initialize carry: broadcast H_init/L_init to all total_seq positions */
    for (int s = 0; s < total_seq; s++) {
        memcpy(z_H + s * HIDDEN_SIZE, model->h_init, HIDDEN_SIZE * sizeof(float));
        memcpy(z_L + s * HIDDEN_SIZE, model->l_init, HIDDEN_SIZE * sizeof(float));
    }

    /* 3. Dual-z reasoning loop */
    for (int h = 0; h < H_CYCLES; h++) {
        /* L_CYCLES iterations on z_L with injection = z_H + input_emb */
        for (int l = 0; l < L_CYCLES; l++) {
            /* Compute injection = z_H + input_emb */
            for (int i = 0; i < hidden_floats; i++) {
                injection_buf[i] = z_H[i] + input_emb[i];
            }
            /* z_L = L_level(z_L, z_H + input_emb) */
            trm_l_level(z_L, injection_buf, model, total_seq, block_scratch);

            /* Yield to WDT periodically */
            vTaskDelay(1);
        }

        /* z_H = L_level(z_H, z_L) */
        trm_l_level(z_H, z_L, model, total_seq, block_scratch);

        /* Yield to WDT */
        vTaskDelay(1);
    }

    /* 4. Output head: logits = lm_head(z_H), skip puzzle positions, pred = argmax */
    trm_output_head(pred_tokens, z_H, seq_len, model);

    int64_t t_end = esp_timer_get_time();
    return t_end - t_start;
}

/* ============== Full Inference with ACT (Adaptive Computation Time) ============== */

int64_t trm_full_inference_act(
    uint8_t *pred_tokens,
    const uint8_t *input_tokens,
    int seq_len,
    const TRMModel *model,
    float *scratch,
    int max_act_steps,
    int *steps_used)
{
    const int total_seq = seq_len + PUZZLE_EMB_LEN;
    const int hidden_floats = total_seq * HIDDEN_SIZE;

    int64_t t_start = esp_timer_get_time();

    float *z_H = scratch;
    float *z_L = z_H + hidden_floats;
    float *input_emb = z_L + hidden_floats;
    float *injection_buf = input_emb + hidden_floats;
    float *block_scratch = injection_buf + hidden_floats;

    /* Compute input embeddings once (same for all ACT steps) */
    trm_embed(input_emb, input_tokens, seq_len, model);

    int actual_steps = max_act_steps;

    /* IMPORTANT: ACT steps must carry state forward across steps.
     * PyTorch ACT wrapper resets carry only on the *first* step (initial carry has halted=True),
     * then keeps the inner carry for subsequent steps until the last step.
     *
     * Therefore we initialize z_H/z_L once, then run repeated improvement steps in-place. */
    for (int s = 0; s < total_seq; s++) {
        memcpy(z_H + s * HIDDEN_SIZE, model->h_init, HIDDEN_SIZE * sizeof(float));
        memcpy(z_L + s * HIDDEN_SIZE, model->l_init, HIDDEN_SIZE * sizeof(float));
    }

    for (int act_step = 0; act_step < max_act_steps; act_step++) {
        /* Dual-z reasoning loop (one inner forward pass) */
        for (int h = 0; h < H_CYCLES; h++) {
            for (int l = 0; l < L_CYCLES; l++) {
                for (int i = 0; i < hidden_floats; i++) {
                    injection_buf[i] = z_H[i] + input_emb[i];
                }
                trm_l_level(z_L, injection_buf, model, total_seq, block_scratch);
                vTaskDelay(1);
            }
            trm_l_level(z_H, z_L, model, total_seq, block_scratch);
            vTaskDelay(1);
        }

        /* Check halting via q_head on z_H at position 0 (first puzzle_emb position) */
        if (model->q_head_weight && act_step < max_act_steps - 1) {
            /* q_halt = dot(z_H[0], q_head_weight[0]) + q_head_bias[0] */
            float q_halt = 0.0f;
            const float *zh0 = z_H;  /* position 0 of z_H */
            const float *w0 = model->q_head_weight;  /* row 0 of [2, HIDDEN_SIZE] */
            for (int d = 0; d < HIDDEN_SIZE; d++) {
                q_halt += zh0[d] * w0[d];
            }
            q_halt += model->q_head_bias[0];

            if (q_halt > 0.0f) {
                actual_steps = act_step + 1;
                /* IMPORTANT: do not print logs here.
                 * The text eval protocol ('T' command) is line-based and expects
                 * only TIME_MS / STEPS_USED / PRED lines. ESP_LOG output shares
                 * the same UART and will break framing. */
                break;
            }
        }

        actual_steps = act_step + 1;
    }

    /* Output head: logits = lm_head(z_H), skip puzzle positions, pred = argmax */
    trm_output_head(pred_tokens, z_H, seq_len, model);

    if (steps_used) *steps_used = actual_steps;

    int64_t t_end = esp_timer_get_time();
    return t_end - t_start;
}

/* ============== Full Inference Memory Calculation ============== */

int trm_full_inference_mem(int seq_len)
{
    int total_seq = seq_len + PUZZLE_EMB_LEN;
    int hidden_floats = total_seq * HIDDEN_SIZE;
    /* z_H + z_L + input_emb + injection_buf + block_scratch */
    return (4 * hidden_floats) * sizeof(float) + trm_scratch_size(total_seq);
}

/* ============== Standalone Benchmark ============== */

void trm_benchmark_ternary_matmul(int iterations)
{
    ESP_LOGI(TAG, "Benchmarking ternary matmul...");

    /* Allocate test data in PSRAM */
    int M = HIDDEN_SIZE;  /* 512 */
    int K = HIDDEN_SIZE;  /* 512 */
    int packed_K = K / WEIGHTS_PER_BYTE;

    float *input = (float *)heap_caps_malloc(K * sizeof(float), MALLOC_CAP_SPIRAM);
    float *output = (float *)heap_caps_malloc(M * sizeof(float), MALLOC_CAP_SPIRAM);
    uint8_t *weights = (uint8_t *)heap_caps_malloc(M * packed_K, MALLOC_CAP_SPIRAM);

    if (!input || !output || !weights) {
        ESP_LOGE(TAG, "Failed to allocate benchmark buffers");
        return;
    }

    /* Fill with test data */
    for (int i = 0; i < K; i++) input[i] = (float)(i % 7 - 3) * 0.1f;
    for (int i = 0; i < M * packed_K; i++) weights[i] = (uint8_t)(i * 37 % 256);

    /* Warmup */
    ternary_matmul(output, input, weights, 1.0f, M, K);

    /* Benchmark */
    BenchmarkResult result;
    result.name = "Ternary MatMul 512x512";
    result.iterations = iterations;
    result.min_us = 1e9f;
    result.max_us = 0.0f;

    int64_t start = benchmark_time_us();
    for (int i = 0; i < iterations; i++) {
        int64_t t0 = benchmark_time_us();
        ternary_matmul(output, input, weights, 1.0f, M, K);
        int64_t t1 = benchmark_time_us();
        float elapsed = (float)(t1 - t0);
        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
    }
    int64_t end = benchmark_time_us();

    result.total_us = end - start;
    result.avg_us = (float)result.total_us / (float)iterations;
    benchmark_print(&result);

    /* Also benchmark the larger gate_up dimension */
    int M2 = GATE_UP_SIZE;  /* 3072 */
    float *output2 = (float *)heap_caps_malloc(M2 * sizeof(float), MALLOC_CAP_SPIRAM);
    uint8_t *weights2 = (uint8_t *)heap_caps_malloc(M2 * packed_K, MALLOC_CAP_SPIRAM);
    if (output2 && weights2) {
        for (int i = 0; i < M2 * packed_K; i++) weights2[i] = (uint8_t)(i * 37 % 256);

        ternary_matmul(output2, input, weights2, 1.0f, M2, K);

        BenchmarkResult result2;
        result2.name = "Ternary MatMul 3072x512 (gate_up)";
        result2.iterations = iterations;
        result2.min_us = 1e9f;
        result2.max_us = 0.0f;

        start = benchmark_time_us();
        for (int i = 0; i < iterations; i++) {
            int64_t t0 = benchmark_time_us();
            ternary_matmul(output2, input, weights2, 1.0f, M2, K);
            int64_t t1 = benchmark_time_us();
            float elapsed = (float)(t1 - t0);
            if (elapsed < result2.min_us) result2.min_us = elapsed;
            if (elapsed > result2.max_us) result2.max_us = elapsed;
        }
        end = benchmark_time_us();

        result2.total_us = end - start;
        result2.avg_us = (float)result2.total_us / (float)iterations;
        benchmark_print(&result2);
    }

    /* === INT8-optimized kernel benchmarks === */
    printf("\n--- INT8-optimized kernel ---\n");

    /* INT8: 512x512 */
    BenchmarkResult r3;
    r3.name = "Ternary MatMul 512x512 (INT8)";
    r3.iterations = iterations;
    r3.min_us = 1e9f;
    r3.max_us = 0.0f;

    ternary_matmul_int8(output, input, weights, 1.0f, M, K);  /* warmup */

    start = benchmark_time_us();
    for (int i = 0; i < iterations; i++) {
        int64_t t0 = benchmark_time_us();
        ternary_matmul_int8(output, input, weights, 1.0f, M, K);
        int64_t t1 = benchmark_time_us();
        float elapsed = (float)(t1 - t0);
        if (elapsed < r3.min_us) r3.min_us = elapsed;
        if (elapsed > r3.max_us) r3.max_us = elapsed;
    }
    end = benchmark_time_us();
    r3.total_us = end - start;
    r3.avg_us = (float)r3.total_us / (float)iterations;
    benchmark_print(&r3);

    /* INT8: 3072x512 */
    if (output2 && weights2) {
        BenchmarkResult r4;
        r4.name = "Ternary MatMul 3072x512 (INT8)";
        r4.iterations = iterations;
        r4.min_us = 1e9f;
        r4.max_us = 0.0f;

        ternary_matmul_int8(output2, input, weights2, 1.0f, M2, K);  /* warmup */

        start = benchmark_time_us();
        for (int i = 0; i < iterations; i++) {
            int64_t t0 = benchmark_time_us();
            ternary_matmul_int8(output2, input, weights2, 1.0f, M2, K);
            int64_t t1 = benchmark_time_us();
            float elapsed = (float)(t1 - t0);
            if (elapsed < r4.min_us) r4.min_us = elapsed;
            if (elapsed > r4.max_us) r4.max_us = elapsed;
        }
        end = benchmark_time_us();
        r4.total_us = end - start;
        r4.avg_us = (float)r4.total_us / (float)iterations;
        benchmark_print(&r4);
    }

    /* === SIMD-optimized (PIE) kernel benchmarks === */
    printf("\n--- SIMD (PIE) kernel ---\n");

    /* SIMD: 512x512 */
    BenchmarkResult r5;
    r5.name = "Ternary MatMul 512x512 (SIMD)";
    r5.iterations = iterations;
    r5.min_us = 1e9f;
    r5.max_us = 0.0f;

    ternary_matmul_simd(output, input, weights, 1.0f, M, K);  /* warmup */

    start = benchmark_time_us();
    for (int i = 0; i < iterations; i++) {
        int64_t t0 = benchmark_time_us();
        ternary_matmul_simd(output, input, weights, 1.0f, M, K);
        int64_t t1 = benchmark_time_us();
        float elapsed = (float)(t1 - t0);
        if (elapsed < r5.min_us) r5.min_us = elapsed;
        if (elapsed > r5.max_us) r5.max_us = elapsed;
    }
    end = benchmark_time_us();
    r5.total_us = end - start;
    r5.avg_us = (float)r5.total_us / (float)iterations;
    benchmark_print(&r5);

    /* SIMD: 3072x512 */
    if (output2 && weights2) {
        BenchmarkResult r6;
        r6.name = "Ternary MatMul 3072x512 (SIMD)";
        r6.iterations = iterations;
        r6.min_us = 1e9f;
        r6.max_us = 0.0f;

        ternary_matmul_simd(output2, input, weights2, 1.0f, M2, K);  /* warmup */

        start = benchmark_time_us();
        for (int i = 0; i < iterations; i++) {
            int64_t t0 = benchmark_time_us();
            ternary_matmul_simd(output2, input, weights2, 1.0f, M2, K);
            int64_t t1 = benchmark_time_us();
            float elapsed = (float)(t1 - t0);
            if (elapsed < r6.min_us) r6.min_us = elapsed;
            if (elapsed > r6.max_us) r6.max_us = elapsed;
        }
        end = benchmark_time_us();
        r6.total_us = end - start;
        r6.avg_us = (float)r6.total_us / (float)iterations;
        benchmark_print(&r6);
    }

    /* === Additional model dimension benchmarks === */
    printf("\n--- Additional model dimensions ---\n");

    /* 1536x512 (QKV projection) */
    {
        int Ma = QKV_OUT_SIZE;  /* 1536 */
        int Ka = HIDDEN_SIZE;   /* 512 */
        int packed_Ka = Ka / WEIGHTS_PER_BYTE;
        float *out_a = (float *)heap_caps_malloc(Ma * sizeof(float), MALLOC_CAP_SPIRAM);
        uint8_t *w_a = (uint8_t *)heap_caps_malloc(Ma * packed_Ka, MALLOC_CAP_SPIRAM);
        if (out_a && w_a) {
            for (int i = 0; i < Ma * packed_Ka; i++) w_a[i] = (uint8_t)(i * 37 % 256);

            /* FP32 */
            ternary_matmul(out_a, input, w_a, 1.0f, Ma, Ka);
            BenchmarkResult ra;
            ra.name = "FP32 1536x512 (QKV)";
            ra.iterations = iterations;
            ra.min_us = 1e9f; ra.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul(out_a, input, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < ra.min_us) ra.min_us = elapsed;
                if (elapsed > ra.max_us) ra.max_us = elapsed;
            }
            end = benchmark_time_us();
            ra.total_us = end - start;
            ra.avg_us = (float)ra.total_us / (float)iterations;
            benchmark_print(&ra);

            /* INT8 */
            ternary_matmul_int8(out_a, input, w_a, 1.0f, Ma, Ka);
            BenchmarkResult rb;
            rb.name = "INT8 1536x512 (QKV)";
            rb.iterations = iterations;
            rb.min_us = 1e9f; rb.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul_int8(out_a, input, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < rb.min_us) rb.min_us = elapsed;
                if (elapsed > rb.max_us) rb.max_us = elapsed;
            }
            end = benchmark_time_us();
            rb.total_us = end - start;
            rb.avg_us = (float)rb.total_us / (float)iterations;
            benchmark_print(&rb);

            /* SIMD */
            ternary_matmul_simd(out_a, input, w_a, 1.0f, Ma, Ka);
            BenchmarkResult rc;
            rc.name = "SIMD 1536x512 (QKV)";
            rc.iterations = iterations;
            rc.min_us = 1e9f; rc.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul_simd(out_a, input, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < rc.min_us) rc.min_us = elapsed;
                if (elapsed > rc.max_us) rc.max_us = elapsed;
            }
            end = benchmark_time_us();
            rc.total_us = end - start;
            rc.avg_us = (float)rc.total_us / (float)iterations;
            benchmark_print(&rc);

            heap_caps_free(out_a);
            heap_caps_free(w_a);
        } else {
            ESP_LOGW(TAG, "Skipped 1536x512 — alloc failed");
            if (out_a) heap_caps_free(out_a);
            if (w_a) heap_caps_free(w_a);
        }
    }

    /* 512x1536 (Down MLP) */
    {
        int Ma = HIDDEN_SIZE;   /* 512 */
        int Ka = MLP_INTER;     /* 1536 */
        int packed_Ka = Ka / WEIGHTS_PER_BYTE;
        float *in_a = (float *)heap_caps_malloc(Ka * sizeof(float), MALLOC_CAP_SPIRAM);
        float *out_a = (float *)heap_caps_malloc(Ma * sizeof(float), MALLOC_CAP_SPIRAM);
        uint8_t *w_a = (uint8_t *)heap_caps_malloc(Ma * packed_Ka, MALLOC_CAP_SPIRAM);
        if (in_a && out_a && w_a) {
            for (int i = 0; i < Ka; i++) in_a[i] = (float)(i % 7 - 3) * 0.1f;
            for (int i = 0; i < Ma * packed_Ka; i++) w_a[i] = (uint8_t)(i * 37 % 256);

            /* FP32 */
            ternary_matmul(out_a, in_a, w_a, 1.0f, Ma, Ka);
            BenchmarkResult ra;
            ra.name = "FP32 512x1536 (down MLP)";
            ra.iterations = iterations;
            ra.min_us = 1e9f; ra.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul(out_a, in_a, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < ra.min_us) ra.min_us = elapsed;
                if (elapsed > ra.max_us) ra.max_us = elapsed;
            }
            end = benchmark_time_us();
            ra.total_us = end - start;
            ra.avg_us = (float)ra.total_us / (float)iterations;
            benchmark_print(&ra);

            /* INT8 */
            ternary_matmul_int8(out_a, in_a, w_a, 1.0f, Ma, Ka);
            BenchmarkResult rb;
            rb.name = "INT8 512x1536 (down MLP)";
            rb.iterations = iterations;
            rb.min_us = 1e9f; rb.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul_int8(out_a, in_a, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < rb.min_us) rb.min_us = elapsed;
                if (elapsed > rb.max_us) rb.max_us = elapsed;
            }
            end = benchmark_time_us();
            rb.total_us = end - start;
            rb.avg_us = (float)rb.total_us / (float)iterations;
            benchmark_print(&rb);

            /* SIMD */
            ternary_matmul_simd(out_a, in_a, w_a, 1.0f, Ma, Ka);
            BenchmarkResult rc;
            rc.name = "SIMD 512x1536 (down MLP)";
            rc.iterations = iterations;
            rc.min_us = 1e9f; rc.max_us = 0.0f;
            start = benchmark_time_us();
            for (int i = 0; i < iterations; i++) {
                int64_t t0 = benchmark_time_us();
                ternary_matmul_simd(out_a, in_a, w_a, 1.0f, Ma, Ka);
                int64_t t1 = benchmark_time_us();
                float elapsed = (float)(t1 - t0);
                if (elapsed < rc.min_us) rc.min_us = elapsed;
                if (elapsed > rc.max_us) rc.max_us = elapsed;
            }
            end = benchmark_time_us();
            rc.total_us = end - start;
            rc.avg_us = (float)rc.total_us / (float)iterations;
            benchmark_print(&rc);

            heap_caps_free(in_a);
            heap_caps_free(out_a);
            heap_caps_free(w_a);
        } else {
            ESP_LOGW(TAG, "Skipped 512x1536 — alloc failed");
            if (in_a) heap_caps_free(in_a);
            if (out_a) heap_caps_free(out_a);
            if (w_a) heap_caps_free(w_a);
        }
    }

    heap_caps_free(input);
    heap_caps_free(output);
    heap_caps_free(weights);
    if (output2) heap_caps_free(output2);
    if (weights2) heap_caps_free(weights2);
}

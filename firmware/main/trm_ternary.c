/**
 * Ternary Matrix Multiplication for ESP32-S3
 * 
 * Port of the TL2 lookup-table strategy from AVX2 to scalar C.
 * The inner loop uses ONLY additions and subtractions — zero multiplications.
 * 
 * Weight packing (simple 2-bit, ESP32 version):
 *   4 weights per byte, 2 bits each (LSB first):
 *     0b00 = -1
 *     0b01 =  0
 *     0b10 = +1
 */

#include "trm_ternary.h"
#include "model_config.h"
#include <string.h>
#include <math.h>

void ternary_matmul(
    float *output,
    const float *input,
    const uint8_t *weights,
    float scale,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        const uint8_t *w_row = weights + m * k_packed;

        for (int kp = 0; kp < k_packed; kp++) {
            uint8_t packed = w_row[kp];
            int base_k = kp * WEIGHTS_PER_BYTE;

            /* Unpack 4 ternary values and accumulate */
            /* w0 = bits [1:0] */
            int w0 = (packed & 0x03);
            /* w1 = bits [3:2] */
            int w1 = ((packed >> 2) & 0x03);
            /* w2 = bits [5:4] */
            int w2 = ((packed >> 4) & 0x03);
            /* w3 = bits [7:6] */
            int w3 = ((packed >> 6) & 0x03);

            /* Map: 0b00 -> -1, 0b01 -> 0, 0b10 -> +1 */
            /* This is: value = ternary_val - 1 */
            /* For -1 (0b00): acc -= input[k] */
            /* For  0 (0b01): acc += 0 (skip) */
            /* For +1 (0b10): acc += input[k] */

            /* Branch-free using lookup: val = w - 1, then acc += val * input */
            /* But we avoid multiplication entirely! */
            /* Instead: if w==0 => subtract, if w==2 => add, if w==1 => skip */

            if (w0 == 0) acc -= input[base_k + 0];
            else if (w0 == 2) acc += input[base_k + 0];

            if (w1 == 0) acc -= input[base_k + 1];
            else if (w1 == 2) acc += input[base_k + 1];

            if (w2 == 0) acc -= input[base_k + 2];
            else if (w2 == 2) acc += input[base_k + 2];

            if (w3 == 0) acc -= input[base_k + 3];
            else if (w3 == 2) acc += input[base_k + 3];
        }

        output[m] = acc * scale;
    }
}

/* Helper: lookup table for ternary decode */
/* Indexed by 2-bit value, gives -1.0, 0.0, or +1.0 */
static const float TERNARY_LUT[4] = { -1.0f, 0.0f, 1.0f, 0.0f };

void ternary_matmul_batched(
    float *output,
    const float *input,
    const uint8_t *weights,
    float scale,
    int B,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            const uint8_t *w_row = weights + m * k_packed;

            for (int kp = 0; kp < k_packed; kp++) {
                uint8_t packed = w_row[kp];
                int base_k = kp * WEIGHTS_PER_BYTE;

                /* Unroll 4 weights per byte using branch-free LUT */
                float w0 = TERNARY_LUT[packed & 0x03];
                float w1 = TERNARY_LUT[(packed >> 2) & 0x03];
                float w2 = TERNARY_LUT[(packed >> 4) & 0x03];
                float w3 = TERNARY_LUT[(packed >> 6) & 0x03];

                /* These are NOT multiplications in the numerical sense —
                 * the LUT values are {-1, 0, +1}, so the compiler can
                 * optimize w * input to conditional add/sub/nop.
                 * But even as FP multiply, (-1)*x and (+1)*x are trivial. */
                acc += w0 * in_b[base_k + 0];
                acc += w1 * in_b[base_k + 1];
                acc += w2 * in_b[base_k + 2];
                acc += w3 * in_b[base_k + 3];
            }

            out_b[m] = acc * scale;
        }
    }
}


/* =================================================================
 * INT8-optimized ternary matmul — Optimization 1 + 4
 *
 * Key improvements over FP32 version:
 *   1. Input quantized to INT8 (per-vector absmax scaling)
 *   2. INT32 accumulation: (ternary_val - 1) * int8_input
 *   3. uint32_t reads: 4 packed bytes = 16 weights per iteration
 *   4. Only integer arithmetic in inner loop (no FPU)
 *   5. Dequantization applied once per output element
 * ================================================================= */

/* Quantize float vector to INT8 with symmetric absmax scaling.
 * Returns the scale factor: original = quantized * scale */
static float quantize_to_int8(int8_t *out, const float *in, int n)
{
    /* Find absmax */
    float absmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = in[i];
        if (v < 0) v = -v;
        if (v > absmax) absmax = v;
    }

    if (absmax < 1e-10f) {
        memset(out, 0, n);
        return 0.0f;
    }

    float inv_scale = 127.0f / absmax;
    for (int i = 0; i < n; i++) {
#ifdef USE_TRUNC
        int v = (int)(in[i] * inv_scale);
#else
        int v = (int)roundf(in[i] * inv_scale);
#endif
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }

    return absmax / 127.0f;  /* scale = absmax / 127 */
}

void ternary_matmul_int8(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    /* Quantize input once (amortized over all M rows) */
    int8_t q_input[1536];  /* Max K in our model */
    float in_scale = quantize_to_int8(q_input, input, K);
    float combined_scale = in_scale * weight_scale;

    for (int m = 0; m < M; m++) {
        int32_t acc = 0;
        const uint8_t *w_row = weights + m * k_packed;

        /* Process 4 packed bytes (16 weights) per iteration.
         * K is always a multiple of 16 in our model (512, 1536). */
        for (int kp = 0; kp < k_packed; kp += 4) {
            uint32_t p = *(const uint32_t *)(w_row + kp);
            const int8_t *q = q_input + (kp << 2);

            /* Fully unrolled: 16 weights from 4 packed bytes */
            acc += ((int)((p      ) & 3) - 1) * (int)q[ 0];
            acc += ((int)((p >>  2) & 3) - 1) * (int)q[ 1];
            acc += ((int)((p >>  4) & 3) - 1) * (int)q[ 2];
            acc += ((int)((p >>  6) & 3) - 1) * (int)q[ 3];
            acc += ((int)((p >>  8) & 3) - 1) * (int)q[ 4];
            acc += ((int)((p >> 10) & 3) - 1) * (int)q[ 5];
            acc += ((int)((p >> 12) & 3) - 1) * (int)q[ 6];
            acc += ((int)((p >> 14) & 3) - 1) * (int)q[ 7];
            acc += ((int)((p >> 16) & 3) - 1) * (int)q[ 8];
            acc += ((int)((p >> 18) & 3) - 1) * (int)q[ 9];
            acc += ((int)((p >> 20) & 3) - 1) * (int)q[10];
            acc += ((int)((p >> 22) & 3) - 1) * (int)q[11];
            acc += ((int)((p >> 24) & 3) - 1) * (int)q[12];
            acc += ((int)((p >> 26) & 3) - 1) * (int)q[13];
            acc += ((int)((p >> 28) & 3) - 1) * (int)q[14];
            acc += ((int)((p >> 30)     ) - 1) * (int)q[15];
        }

        output[m] = (float)acc * combined_scale;
    }
}

void ternary_matmul_batched_int8(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int B,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        /* Quantize this batch element to INT8 */
        int8_t q_input[1536];  /* Max K in our model */
        float in_scale = quantize_to_int8(q_input, in_b, K);
        float combined_scale = in_scale * weight_scale;

        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            const uint8_t *w_row = weights + m * k_packed;

            for (int kp = 0; kp < k_packed; kp += 4) {
                uint32_t p = *(const uint32_t *)(w_row + kp);
                const int8_t *q = q_input + (kp << 2);

                acc += ((int)((p      ) & 3) - 1) * (int)q[ 0];
                acc += ((int)((p >>  2) & 3) - 1) * (int)q[ 1];
                acc += ((int)((p >>  4) & 3) - 1) * (int)q[ 2];
                acc += ((int)((p >>  6) & 3) - 1) * (int)q[ 3];
                acc += ((int)((p >>  8) & 3) - 1) * (int)q[ 4];
                acc += ((int)((p >> 10) & 3) - 1) * (int)q[ 5];
                acc += ((int)((p >> 12) & 3) - 1) * (int)q[ 6];
                acc += ((int)((p >> 14) & 3) - 1) * (int)q[ 7];
                acc += ((int)((p >> 16) & 3) - 1) * (int)q[ 8];
                acc += ((int)((p >> 18) & 3) - 1) * (int)q[ 9];
                acc += ((int)((p >> 20) & 3) - 1) * (int)q[10];
                acc += ((int)((p >> 22) & 3) - 1) * (int)q[11];
                acc += ((int)((p >> 24) & 3) - 1) * (int)q[12];
                acc += ((int)((p >> 26) & 3) - 1) * (int)q[13];
                acc += ((int)((p >> 28) & 3) - 1) * (int)q[14];
                acc += ((int)((p >> 30)     ) - 1) * (int)q[15];
            }

            out_b[m] = (float)acc * combined_scale;
        }
    }
}

/* ============================================================
 * Float32 matmul from INT8 weights (for timing comparison).
 * Dequantizes each INT8 weight to float32, then does standard
 * multiply-accumulate. Same interface as the preunpacked variant.
 * ============================================================ */

void float32_matmul_from_int8(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int B,
    int M,
    int K)
{
    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        for (int m = 0; m < M; m++) {
            const int8_t *w_row = weights_int8 + m * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (float)w_row[k] * in_b[k];
            }
            out_b[m] = acc * weight_scale;
        }
    }
}

/* Float32 matmul from 2-bit packed ternary weights.
 * Unpacks each weight on the fly: 0b00=-1, 0b01=0, 0b10=+1.
 * Used when pre-unpacked INT8 weights are not available. */
void float32_matmul_from_packed(
    float *output,
    const float *input,
    const uint8_t *packed_weights,
    float weight_scale,
    int B,
    int M,
    int K)
{
    const int packed_K = K / 4;

    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        for (int m = 0; m < M; m++) {
            const uint8_t *pw = packed_weights + m * packed_K;
            float acc = 0.0f;
            int k = 0;
            for (int pk = 0; pk < packed_K; pk++) {
                uint8_t byte = pw[pk];
                acc += (float)((int)(byte & 3) - 1)       * in_b[k];
                acc += (float)((int)((byte >> 2) & 3) - 1) * in_b[k + 1];
                acc += (float)((int)((byte >> 4) & 3) - 1) * in_b[k + 2];
                acc += (float)((int)((byte >> 6) & 3) - 1) * in_b[k + 3];
                k += 4;
            }
            out_b[m] = acc * weight_scale;
        }
    }
}

/**
 * SIMD-Optimized Ternary Matrix Multiplication for ESP32-S3
 *
 * Strategy:
 *   1. Unpack 2-bit packed ternary weights → INT8 buffer via 256-entry LUT
 *   2. Quantize float input → INT8 (per-vector absmax)
 *   3. SIMD dot product using PIE ee.vmulas.s8.accx (16 MACs per cycle)
 *   4. Dequantize result back to float
 *
 * Expected speedup: ~5-8x over scalar FP32 baseline (9.83ms → ~1.5-2ms for 512×512)
 */

#include "trm_ternary.h"
#include "model_config.h"
#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "esp_attr.h"

/* ============================================================
 * External SIMD assembly function (in simd_dotprod.S)
 * ============================================================ */
extern int32_t simd_dotprod_s8(const int8_t *a, const int8_t *b, int len);

/* ============================================================
 * 256-entry Unpack LUT: packed_byte → 4 × INT8 {-1, 0, +1}
 *
 * Each byte encodes 4 ternary values (2 bits each):
 *   0b00 = -1, 0b01 = 0, 0b10 = +1
 *
 * LUT[byte_value] = { w0, w1, w2, w3 } where wi = ((byte >> 2i) & 3) - 1
 * ============================================================ */

/* Build unpack LUT at compile time */
#define W(b, i) ((int8_t)((((b) >> ((i)*2)) & 3) - 1))
#define LUT_ENTRY(b) { W(b,0), W(b,1), W(b,2), W(b,3) }

static const int8_t UNPACK_LUT[256][4] = {
    LUT_ENTRY(0x00), LUT_ENTRY(0x01), LUT_ENTRY(0x02), LUT_ENTRY(0x03),
    LUT_ENTRY(0x04), LUT_ENTRY(0x05), LUT_ENTRY(0x06), LUT_ENTRY(0x07),
    LUT_ENTRY(0x08), LUT_ENTRY(0x09), LUT_ENTRY(0x0A), LUT_ENTRY(0x0B),
    LUT_ENTRY(0x0C), LUT_ENTRY(0x0D), LUT_ENTRY(0x0E), LUT_ENTRY(0x0F),
    LUT_ENTRY(0x10), LUT_ENTRY(0x11), LUT_ENTRY(0x12), LUT_ENTRY(0x13),
    LUT_ENTRY(0x14), LUT_ENTRY(0x15), LUT_ENTRY(0x16), LUT_ENTRY(0x17),
    LUT_ENTRY(0x18), LUT_ENTRY(0x19), LUT_ENTRY(0x1A), LUT_ENTRY(0x1B),
    LUT_ENTRY(0x1C), LUT_ENTRY(0x1D), LUT_ENTRY(0x1E), LUT_ENTRY(0x1F),
    LUT_ENTRY(0x20), LUT_ENTRY(0x21), LUT_ENTRY(0x22), LUT_ENTRY(0x23),
    LUT_ENTRY(0x24), LUT_ENTRY(0x25), LUT_ENTRY(0x26), LUT_ENTRY(0x27),
    LUT_ENTRY(0x28), LUT_ENTRY(0x29), LUT_ENTRY(0x2A), LUT_ENTRY(0x2B),
    LUT_ENTRY(0x2C), LUT_ENTRY(0x2D), LUT_ENTRY(0x2E), LUT_ENTRY(0x2F),
    LUT_ENTRY(0x30), LUT_ENTRY(0x31), LUT_ENTRY(0x32), LUT_ENTRY(0x33),
    LUT_ENTRY(0x34), LUT_ENTRY(0x35), LUT_ENTRY(0x36), LUT_ENTRY(0x37),
    LUT_ENTRY(0x38), LUT_ENTRY(0x39), LUT_ENTRY(0x3A), LUT_ENTRY(0x3B),
    LUT_ENTRY(0x3C), LUT_ENTRY(0x3D), LUT_ENTRY(0x3E), LUT_ENTRY(0x3F),
    LUT_ENTRY(0x40), LUT_ENTRY(0x41), LUT_ENTRY(0x42), LUT_ENTRY(0x43),
    LUT_ENTRY(0x44), LUT_ENTRY(0x45), LUT_ENTRY(0x46), LUT_ENTRY(0x47),
    LUT_ENTRY(0x48), LUT_ENTRY(0x49), LUT_ENTRY(0x4A), LUT_ENTRY(0x4B),
    LUT_ENTRY(0x4C), LUT_ENTRY(0x4D), LUT_ENTRY(0x4E), LUT_ENTRY(0x4F),
    LUT_ENTRY(0x50), LUT_ENTRY(0x51), LUT_ENTRY(0x52), LUT_ENTRY(0x53),
    LUT_ENTRY(0x54), LUT_ENTRY(0x55), LUT_ENTRY(0x56), LUT_ENTRY(0x57),
    LUT_ENTRY(0x58), LUT_ENTRY(0x59), LUT_ENTRY(0x5A), LUT_ENTRY(0x5B),
    LUT_ENTRY(0x5C), LUT_ENTRY(0x5D), LUT_ENTRY(0x5E), LUT_ENTRY(0x5F),
    LUT_ENTRY(0x60), LUT_ENTRY(0x61), LUT_ENTRY(0x62), LUT_ENTRY(0x63),
    LUT_ENTRY(0x64), LUT_ENTRY(0x65), LUT_ENTRY(0x66), LUT_ENTRY(0x67),
    LUT_ENTRY(0x68), LUT_ENTRY(0x69), LUT_ENTRY(0x6A), LUT_ENTRY(0x6B),
    LUT_ENTRY(0x6C), LUT_ENTRY(0x6D), LUT_ENTRY(0x6E), LUT_ENTRY(0x6F),
    LUT_ENTRY(0x70), LUT_ENTRY(0x71), LUT_ENTRY(0x72), LUT_ENTRY(0x73),
    LUT_ENTRY(0x74), LUT_ENTRY(0x75), LUT_ENTRY(0x76), LUT_ENTRY(0x77),
    LUT_ENTRY(0x78), LUT_ENTRY(0x79), LUT_ENTRY(0x7A), LUT_ENTRY(0x7B),
    LUT_ENTRY(0x7C), LUT_ENTRY(0x7D), LUT_ENTRY(0x7E), LUT_ENTRY(0x7F),
    LUT_ENTRY(0x80), LUT_ENTRY(0x81), LUT_ENTRY(0x82), LUT_ENTRY(0x83),
    LUT_ENTRY(0x84), LUT_ENTRY(0x85), LUT_ENTRY(0x86), LUT_ENTRY(0x87),
    LUT_ENTRY(0x88), LUT_ENTRY(0x89), LUT_ENTRY(0x8A), LUT_ENTRY(0x8B),
    LUT_ENTRY(0x8C), LUT_ENTRY(0x8D), LUT_ENTRY(0x8E), LUT_ENTRY(0x8F),
    LUT_ENTRY(0x90), LUT_ENTRY(0x91), LUT_ENTRY(0x92), LUT_ENTRY(0x93),
    LUT_ENTRY(0x94), LUT_ENTRY(0x95), LUT_ENTRY(0x96), LUT_ENTRY(0x97),
    LUT_ENTRY(0x98), LUT_ENTRY(0x99), LUT_ENTRY(0x9A), LUT_ENTRY(0x9B),
    LUT_ENTRY(0x9C), LUT_ENTRY(0x9D), LUT_ENTRY(0x9E), LUT_ENTRY(0x9F),
    LUT_ENTRY(0xA0), LUT_ENTRY(0xA1), LUT_ENTRY(0xA2), LUT_ENTRY(0xA3),
    LUT_ENTRY(0xA4), LUT_ENTRY(0xA5), LUT_ENTRY(0xA6), LUT_ENTRY(0xA7),
    LUT_ENTRY(0xA8), LUT_ENTRY(0xA9), LUT_ENTRY(0xAA), LUT_ENTRY(0xAB),
    LUT_ENTRY(0xAC), LUT_ENTRY(0xAD), LUT_ENTRY(0xAE), LUT_ENTRY(0xAF),
    LUT_ENTRY(0xB0), LUT_ENTRY(0xB1), LUT_ENTRY(0xB2), LUT_ENTRY(0xB3),
    LUT_ENTRY(0xB4), LUT_ENTRY(0xB5), LUT_ENTRY(0xB6), LUT_ENTRY(0xB7),
    LUT_ENTRY(0xB8), LUT_ENTRY(0xB9), LUT_ENTRY(0xBA), LUT_ENTRY(0xBB),
    LUT_ENTRY(0xBC), LUT_ENTRY(0xBD), LUT_ENTRY(0xBE), LUT_ENTRY(0xBF),
    LUT_ENTRY(0xC0), LUT_ENTRY(0xC1), LUT_ENTRY(0xC2), LUT_ENTRY(0xC3),
    LUT_ENTRY(0xC4), LUT_ENTRY(0xC5), LUT_ENTRY(0xC6), LUT_ENTRY(0xC7),
    LUT_ENTRY(0xC8), LUT_ENTRY(0xC9), LUT_ENTRY(0xCA), LUT_ENTRY(0xCB),
    LUT_ENTRY(0xCC), LUT_ENTRY(0xCD), LUT_ENTRY(0xCE), LUT_ENTRY(0xCF),
    LUT_ENTRY(0xD0), LUT_ENTRY(0xD1), LUT_ENTRY(0xD2), LUT_ENTRY(0xD3),
    LUT_ENTRY(0xD4), LUT_ENTRY(0xD5), LUT_ENTRY(0xD6), LUT_ENTRY(0xD7),
    LUT_ENTRY(0xD8), LUT_ENTRY(0xD9), LUT_ENTRY(0xDA), LUT_ENTRY(0xDB),
    LUT_ENTRY(0xDC), LUT_ENTRY(0xDD), LUT_ENTRY(0xDE), LUT_ENTRY(0xDF),
    LUT_ENTRY(0xE0), LUT_ENTRY(0xE1), LUT_ENTRY(0xE2), LUT_ENTRY(0xE3),
    LUT_ENTRY(0xE4), LUT_ENTRY(0xE5), LUT_ENTRY(0xE6), LUT_ENTRY(0xE7),
    LUT_ENTRY(0xE8), LUT_ENTRY(0xE9), LUT_ENTRY(0xEA), LUT_ENTRY(0xEB),
    LUT_ENTRY(0xEC), LUT_ENTRY(0xED), LUT_ENTRY(0xEE), LUT_ENTRY(0xEF),
    LUT_ENTRY(0xF0), LUT_ENTRY(0xF1), LUT_ENTRY(0xF2), LUT_ENTRY(0xF3),
    LUT_ENTRY(0xF4), LUT_ENTRY(0xF5), LUT_ENTRY(0xF6), LUT_ENTRY(0xF7),
    LUT_ENTRY(0xF8), LUT_ENTRY(0xF9), LUT_ENTRY(0xFA), LUT_ENTRY(0xFB),
    LUT_ENTRY(0xFC), LUT_ENTRY(0xFD), LUT_ENTRY(0xFE), LUT_ENTRY(0xFF),
};

/* ============================================================
 * Unpack packed ternary weights → INT8 buffer
 *
 * Reads k_packed bytes, writes K INT8 values {-1, 0, +1}
 * Uses LUT for speed: ~3 cycles per byte = ~384 cycles for K=512
 * ============================================================ */
static void unpack_ternary_row(
    int8_t *out,
    const uint8_t *packed,
    int k_packed)
{
    for (int i = 0; i < k_packed; i++) {
        const int8_t *entry = UNPACK_LUT[packed[i]];
        out[i * 4 + 0] = entry[0];
        out[i * 4 + 1] = entry[1];
        out[i * 4 + 2] = entry[2];
        out[i * 4 + 3] = entry[3];
    }
}

/* ============================================================
 * Public: unpack an entire M×K ternary weight matrix
 * Used by model_unpack_weights() for pre-unpacking at load time
 * ============================================================ */
void unpack_ternary_matrix(int8_t *out, const uint8_t *packed, int M, int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;
    for (int m = 0; m < M; m++) {
        unpack_ternary_row(out + m * K, packed + m * k_packed, k_packed);
    }
}

/* ============================================================
 * Quantize float vector → INT8 (symmetric absmax)
 * Returns scale: original = quantized * scale
 * Output must be 16-byte aligned for SIMD
 * ============================================================ */
static float quantize_input_simd(int8_t *out, const float *in, int n)
{
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

    return absmax / 127.0f;
}

/* ============================================================
 * SIMD Ternary MatMul: output = W_ternary @ input * scale
 *
 * For each output row:
 *   1. Unpack weight row: 2-bit packed → INT8 via LUT (~384 cycles)
 *   2. SIMD dot product: INT8 input × INT8 weight (~32 cycles for K=512)
 *   3. Dequantize: float result = int32_result * combined_scale
 * ============================================================ */

void ternary_matmul_simd(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    /* Aligned buffers for SIMD (in internal SRAM for speed) */
    /* Use static buffers to avoid stack allocation issues */
    static int8_t q_input[1536] __attribute__((aligned(16)));
    static int8_t w_unpacked[1536] __attribute__((aligned(16)));

    /* Quantize input once */
    float in_scale = quantize_input_simd(q_input, input, K);
    float combined_scale = in_scale * weight_scale;

    for (int m = 0; m < M; m++) {
        /* Unpack this weight row from 2-bit → INT8 */
        unpack_ternary_row(w_unpacked, weights + m * k_packed, k_packed);

        /* SIMD dot product */
        int32_t dot = simd_dotprod_s8(q_input, w_unpacked, K);

        output[m] = (float)dot * combined_scale;
    }
}

void ternary_matmul_batched_simd(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int B,
    int M,
    int K)
{
    const int k_packed = K / WEIGHTS_PER_BYTE;

    /* Aligned buffers for SIMD */
    static int8_t q_input[1536] __attribute__((aligned(16)));
    static int8_t w_unpacked[1536] __attribute__((aligned(16)));

    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        /* Quantize this batch element */
        float in_scale = quantize_input_simd(q_input, in_b, K);
        float combined_scale = in_scale * weight_scale;

        for (int m = 0; m < M; m++) {
            /* Unpack weight row */
            unpack_ternary_row(w_unpacked, weights + m * k_packed, k_packed);

            /* SIMD dot product */
            int32_t dot = simd_dotprod_s8(q_input, w_unpacked, K);

            out_b[m] = (float)dot * combined_scale;
        }
    }
}

/* ============================================================
 * Pre-Unpacked SIMD MatMul: weights already in INT8 format
 *
 * Skips the per-row LUT unpack step — weights point directly
 * to pre-unpacked INT8 buffers in PSRAM.
 * ============================================================ */

void ternary_matmul_simd_preunpacked(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int M,
    int K)
{
    /* Aligned buffer for quantized input */
    static int8_t q_input[1536] __attribute__((aligned(16)));

    /* Quantize input once */
    float in_scale = quantize_input_simd(q_input, input, K);
    float combined_scale = in_scale * weight_scale;

    for (int m = 0; m < M; m++) {
        /* Direct SIMD dot product — no unpack needed */
        int32_t dot = simd_dotprod_s8(q_input, weights_int8 + m * K, K);
        output[m] = (float)dot * combined_scale;
    }
}

void ternary_matmul_batched_simd_preunpacked(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int B,
    int M,
    int K)
{
    /* Aligned buffer for quantized input */
    static int8_t q_input[1536] __attribute__((aligned(16)));

    for (int b = 0; b < B; b++) {
        const float *in_b = input + b * K;
        float *out_b = output + b * M;

        /* Quantize this batch element */
        float in_scale = quantize_input_simd(q_input, in_b, K);
        float combined_scale = in_scale * weight_scale;

        for (int m = 0; m < M; m++) {
            /* Direct SIMD dot product — no unpack needed */
            int32_t dot = simd_dotprod_s8(q_input, weights_int8 + m * K, K);
            out_b[m] = (float)dot * combined_scale;
        }
    }
}

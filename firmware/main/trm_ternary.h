/**
 * Ternary Matrix Multiplication for ESP32-S3
 * 
 * Port of the TL2 lookup-table strategy from AVX2 to scalar C.
 * 
 * Key insight: For ternary weights {-1, 0, +1}, the matrix-vector product
 * reduces to additions and subtractions — NO multiplications needed.
 * 
 * Weight format (ESP32 simple packing):
 *   4 weights per byte, 2 bits each:
 *     0b00 = -1
 *     0b01 =  0
 *     0b10 = +1
 *   Byte layout: [w3:w2:w1:w0] where each wN is 2 bits
 */

#ifndef TRM_TERNARY_H
#define TRM_TERNARY_H

#include <stdint.h>
#include <stddef.h>

/**
 * Ternary matrix-vector multiplication: output = W_ternary @ input * scale
 * 
 * @param output    Output vector [M], caller must zero-initialize
 * @param input     Input vector [K], float32
 * @param weights   Packed ternary weights [M * K / 4], 2-bit packed
 * @param scale     Per-tensor scale factor (float)
 * @param M         Number of output rows
 * @param K         Number of input columns (must be multiple of 4)
 */
void ternary_matmul(
    float *output,
    const float *input,
    const uint8_t *weights,
    float scale,
    int M,
    int K
);

/**
 * Ternary matrix-matrix multiplication (for batch > 1 or seq_len > 1):
 * output[b][m] = sum_k(W_ternary[m][k] * input[b][k]) * scale
 * 
 * @param output    Output [B * M], row-major
 * @param input     Input [B * K], row-major
 * @param weights   Packed ternary weights [M * K / 4]
 * @param scale     Per-tensor scale factor
 * @param B         Batch dimension
 * @param M         Output dimension
 * @param K         Input dimension (must be multiple of 4)
 */
void ternary_matmul_batched(
    float *output,
    const float *input,
    const uint8_t *weights,
    float scale,
    int B,
    int M,
    int K
);

/**
 * INT8-optimized ternary matmul (Opt 1+4):
 * Input is dynamically quantized to INT8, inner loop uses INT32 accumulation.
 * Same interface as ternary_matmul but ~2-3x faster.
 */
void ternary_matmul_int8(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int M,
    int K
);

void ternary_matmul_batched_int8(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int B,
    int M,
    int K
);

/**
 * SIMD-optimized ternary matmul (PIE vectorized):
 * Uses ESP32-S3 PIE instructions (16 INT8 MACs per cycle).
 * Unpacks weights on-the-fly via LUT, then SIMD dot product.
 */
void ternary_matmul_simd(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int M,
    int K
);

void ternary_matmul_batched_simd(
    float *output,
    const float *input,
    const uint8_t *weights,
    float weight_scale,
    int B,
    int M,
    int K
);

/**
 * Unpack an entire M×K ternary weight matrix from 2-bit packed to INT8.
 * Used for pre-unpacking at model load time.
 */
void unpack_ternary_matrix(int8_t *out, const uint8_t *packed, int M, int K);

/**
 * SIMD matmul with pre-unpacked INT8 weights (no per-row LUT unpack).
 * Weights must already be INT8 {-1,0,+1} and 16-byte aligned.
 */
void ternary_matmul_simd_preunpacked(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int M,
    int K
);

void ternary_matmul_batched_simd_preunpacked(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int B,
    int M,
    int K
);

/**
 * Float32 matmul from pre-unpacked INT8 weights (for timing comparison).
 * Same interface as ternary_matmul_batched_simd_preunpacked but uses
 * standard float32 multiply-accumulate instead of SIMD INT8 dot product.
 */
void float32_matmul_from_int8(
    float *output,
    const float *input,
    const int8_t *weights_int8,
    float weight_scale,
    int B,
    int M,
    int K
);

/**
 * Float32 matmul from 2-bit packed ternary weights.
 * Unpacks each weight on the fly and uses float32 multiply-accumulate.
 * Used when pre-unpacked INT8 weights are not available.
 */
void float32_matmul_from_packed(
    float *output,
    const float *input,
    const uint8_t *packed_weights,
    float weight_scale,
    int B,
    int M,
    int K
);

#endif /* TRM_TERNARY_H */

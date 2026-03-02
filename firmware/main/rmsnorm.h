/**
 * RMSNorm implementation for ESP32-S3
 * 
 * Weight-free:  RMSNorm(x) = x / sqrt(mean(x^2) + eps)
 * Weighted:     RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * norm_weight
 * 
 * The weight-free variant is used for the architectural Post-Norm.
 * The weighted variant is used for BitNet per-layer RMSNorm (learned norm_weight).
 */

#ifndef RMSNORM_H
#define RMSNORM_H

#include <stddef.h>

/**
 * In-place RMSNorm: x = x / sqrt(mean(x^2) + eps)
 * @param x     Vector to normalize [dim], modified in-place
 * @param dim   Vector dimension
 * @param eps   Epsilon for numerical stability
 */
void rmsnorm(float *x, int dim, float eps);

/**
 * RMSNorm with output: out = x / sqrt(mean(x^2) + eps)
 * @param out   Output vector [dim]
 * @param x     Input vector [dim]
 * @param dim   Vector dimension
 * @param eps   Epsilon for numerical stability
 */
void rmsnorm_out(float *out, const float *x, int dim, float eps);

/**
 * Weighted RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * norm_weight[i]
 * Used for BitNet per-layer normalization with learned scale parameters.
 * If norm_weight is NULL, behaves identically to rmsnorm_out().
 * 
 * @param out          Output vector [dim]
 * @param x            Input vector [dim]
 * @param norm_weight  Learned per-feature scale [dim], or NULL for unit weight
 * @param dim          Vector dimension
 * @param eps          Epsilon for numerical stability
 */
void rmsnorm_weighted(float *out, const float *x, const float *norm_weight, int dim, float eps);

#endif /* RMSNORM_H */

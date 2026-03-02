/**
 * RMSNorm implementation for ESP32-S3
 */

#include "rmsnorm.h"
#include <math.h>
#include <stddef.h>

void rmsnorm(float *x, int dim, float eps)
{
    /* Compute mean of squares */
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    /* Normalize in-place */
    for (int i = 0; i < dim; i++) {
        x[i] *= rms;
    }
}

void rmsnorm_out(float *out, const float *x, int dim, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * rms;
    }
}

void rmsnorm_weighted(float *out, const float *x, const float *norm_weight, int dim, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);

    if (norm_weight != NULL) {
        for (int i = 0; i < dim; i++) {
            out[i] = x[i] * rms * norm_weight[i];
        }
    } else {
        for (int i = 0; i < dim; i++) {
            out[i] = x[i] * rms;
        }
    }
}

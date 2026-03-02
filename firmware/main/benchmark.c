/**
 * Benchmarking harness for ESP32-S3
 */

#include "benchmark.h"
#include "esp_timer.h"
#include "esp_log.h"
#include <float.h>

static const char *TAG = "benchmark";

int64_t benchmark_time_us(void)
{
    return esp_timer_get_time();
}

void benchmark_print(const BenchmarkResult *result)
{
    ESP_LOGI(TAG, "=== %s ===", result->name);
    ESP_LOGI(TAG, "  Iterations: %d", result->iterations);
    ESP_LOGI(TAG, "  Total:      %lld us (%.2f ms)", result->total_us,
             (float)result->total_us / 1000.0f);
    ESP_LOGI(TAG, "  Average:    %.2f us (%.2f ms)", result->avg_us,
             result->avg_us / 1000.0f);
    ESP_LOGI(TAG, "  Min:        %.2f us", result->min_us);
    ESP_LOGI(TAG, "  Max:        %.2f us", result->max_us);
}

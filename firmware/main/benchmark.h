/**
 * Benchmarking harness for ESP32-S3
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>

/**
 * Benchmark result for a single test.
 */
typedef struct {
    const char *name;
    int iterations;
    int64_t total_us;    /* Total microseconds */
    float avg_us;        /* Average per iteration */
    float min_us;
    float max_us;
} BenchmarkResult;

/**
 * Get current timestamp in microseconds (monotonic).
 */
int64_t benchmark_time_us(void);

/**
 * Print a benchmark result.
 */
void benchmark_print(const BenchmarkResult *result);

#endif /* BENCHMARK_H */

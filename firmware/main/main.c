/**
 * TRM Inference on ESP32-S3 — Main Entry Point
 * 
 * Provides a UART-based menu for:
 *   1. Loading model from flash
 *   2. Running ternary matmul benchmark
 *   3. Running full inference benchmark
 *   4. Printing memory statistics
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"

#include "model_config.h"
#include "model_loader.h"
#include "trm_engine.h"
#include "benchmark.h"

static const char *TAG = "main";

static TRMModel model;
static bool model_loaded = false;

static void print_banner(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   TRM Ternary Inference — ESP32-S3       ║\n");
    printf("║   1.58-bit Quantized Model               ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_memory(void)
{
    printf("\n=== Memory Report ===\n");
    printf("  Internal SRAM free: %lu bytes\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    printf("  PSRAM free:         %lu bytes (%.2f MB)\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (float)heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / (1024 * 1024));
    printf("  PSRAM total:        %lu bytes (%.2f MB)\n",
           (unsigned long)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
           (float)heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / (1024 * 1024));
    printf("  Minimum free heap:  %lu bytes\n",
           (unsigned long)heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT));
    printf("\n");
}

static void print_menu(void)
{
    printf("\n--- Commands ---\n");
    printf("  [m] Print memory stats\n");
    printf("  [b] Benchmark ternary matmul kernel\n");
    printf("  [l] Load model from flash\n");
    printf("  [u] Pre-unpack weights to INT8 (trades PSRAM for speed)\n");
    printf("  [r] Run reasoning step benchmark\n");
    printf("  [f] Run full inference test (dummy input)\n");
    printf("  [F] Run full inference test (seq_len=81 dummy input)\n");
    printf("  [e] Enter serial evaluation mode\n");
    printf("  [D] Debug text eval (paste tokens, prints predictions)\n");
    printf("  [T] Text eval protocol (supports ACT steps)\n");
    printf("  [i] Model info\n");
    printf("  [h] Show this menu\n");
    printf("> ");
    fflush(stdout);
}

static void cmd_benchmark_kernel(void)
{
    printf("\nRunning ternary matmul benchmark (100 iterations)...\n");
    trm_benchmark_ternary_matmul(100);
    printf("Done.\n");
}

static void cmd_load_model(void)
{
    if (model_loaded) {
        printf("Model already loaded. Free first.\n");
        return;
    }
    printf("\nLoading model from flash...\n");
    int ret = model_load(&model, "model");
    if (ret == 0) {
        model_loaded = true;
        model_print_stats(&model);
    } else {
        printf("Failed to load model.\n");
    }
}

static void cmd_model_info(void)
{
    if (!model_loaded) {
        printf("No model loaded.\n");
        return;
    }
    model_print_stats(&model);
}

static void cmd_unpack_weights(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }
    if (model.weights_unpacked) {
        printf("Weights already pre-unpacked.\n");
        return;
    }
    printf("\nPre-unpacking ternary weights to INT8...\n");
    print_memory();
    int ret = model_unpack_weights(&model);
    if (ret == 0) {
        printf("\nWeights pre-unpacked to INT8 successfully.\n");
        print_memory();
    } else {
        printf("\nFailed — not enough PSRAM.\n");
    }
}

static void cmd_benchmark_inference(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }

    int seq_len = 16;  /* Typical puzzle input length */
    int scratch_bytes = trm_scratch_size(seq_len);

    printf("\nAllocating %.2f KB scratch buffer...\n", (float)scratch_bytes / 1024);
    float *scratch = (float *)heap_caps_malloc(scratch_bytes, MALLOC_CAP_SPIRAM);
    float *hidden = (float *)heap_caps_malloc(seq_len * HIDDEN_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);

    if (!scratch || !hidden) {
        printf("Failed to allocate inference buffers.\n");
        if (scratch) heap_caps_free(scratch);
        if (hidden) heap_caps_free(hidden);
        return;
    }

    /* Initialize hidden state with dummy data */
    for (int i = 0; i < seq_len * HIDDEN_SIZE; i++) {
        hidden[i] = (float)(i % 7 - 3) * 0.01f;
    }

    int iterations = 10;
    printf("Running reasoning step benchmark (%d iterations, seq_len=%d)...\n",
           iterations, seq_len);

    BenchmarkResult result;
    result.name = "TRM Reasoning Step (L_CYCLES x L_LAYERS)";
    result.iterations = iterations;
    result.min_us = 1e9f;
    result.max_us = 0.0f;

    /* Warmup */
    trm_reasoning_step(hidden, &model, seq_len, scratch);
    /* Yield to reset WDT */
    vTaskDelay(10 / portTICK_PERIOD_MS);

    int64_t start = benchmark_time_us();
    for (int i = 0; i < iterations; i++) {
        int64_t t0 = benchmark_time_us();
        trm_reasoning_step(hidden, &model, seq_len, scratch);
        int64_t t1 = benchmark_time_us();
        
        float elapsed = (float)(t1 - t0);
        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;

        /* Yield to reset WDT (outside timing block) */
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
    int64_t end = benchmark_time_us();

    result.total_us = end - start;
    result.avg_us = (float)result.total_us / (float)iterations;
    benchmark_print(&result);

    heap_caps_free(scratch);
    heap_caps_free(hidden);
}

/* ============== Full Inference Test ============== */

static void cmd_full_inference_test(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }

    /* Use a small dummy input for testing: seq_len=8, tokens all 1 */
    int seq_len = 8;
    uint8_t input_tokens[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t pred_tokens[8] = {0};

    int mem_needed = trm_full_inference_mem(seq_len);
    printf("\nFull inference test (seq_len=%d)\n", seq_len);
    printf("  Memory needed: %.2f KB\n", (float)mem_needed / 1024);
    printf("  Free PSRAM:    %lu bytes\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    float *scratch = (float *)heap_caps_malloc(mem_needed, MALLOC_CAP_SPIRAM);
    if (!scratch) {
        printf("Failed to allocate inference buffers.\n");
        return;
    }

    printf("  Running full inference (H_CYCLES=%d, L_CYCLES=%d, L_LAYERS=%d)...\n",
           H_CYCLES, L_CYCLES, L_LAYERS);

    int64_t elapsed_us = trm_full_inference(pred_tokens, input_tokens, seq_len, &model, scratch);

    printf("  Time: %.2f ms (%.2f s)\n",
           (float)elapsed_us / 1000.0f, (float)elapsed_us / 1000000.0f);
    printf("  Input:  [");
    for (int i = 0; i < seq_len; i++) printf("%s%d", i ? ", " : "", input_tokens[i]);
    printf("]\n");
    printf("  Output: [");
    for (int i = 0; i < seq_len; i++) printf("%s%d", i ? ", " : "", pred_tokens[i]);
    printf("]\n");

    heap_caps_free(scratch);
}

static void cmd_full_inference_test_81(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }

    /* Dummy Sudoku-length input: 81 tokens */
    const int seq_len = 81;
    uint8_t *input_tokens = (uint8_t *)malloc(seq_len);
    uint8_t *pred_tokens = (uint8_t *)malloc(seq_len);
    if (!input_tokens || !pred_tokens) {
        printf("Failed to allocate token buffers.\n");
        if (input_tokens) free(input_tokens);
        if (pred_tokens) free(pred_tokens);
        return;
    }

    /* Fill with a simple pattern (mostly blanks/zeros) */
    for (int i = 0; i < seq_len; i++) {
        input_tokens[i] = (i % 9 == 0) ? 5 : 0;
        pred_tokens[i] = 0;
    }

    int mem_needed = trm_full_inference_mem(seq_len);
    printf("\nFull inference test (seq_len=%d)\n", seq_len);
    printf("  Internal total_seq: %d (adds %d puzzle_emb positions)\n", seq_len + PUZZLE_EMB_LEN, PUZZLE_EMB_LEN);
    printf("  Memory needed: %.2f KB\n", (float)mem_needed / 1024);
    printf("  Free PSRAM:    %lu bytes\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    float *scratch = (float *)heap_caps_malloc(mem_needed, MALLOC_CAP_SPIRAM);
    if (!scratch) {
        printf("Failed to allocate inference buffers.\n");
        free(input_tokens);
        free(pred_tokens);
        return;
    }

    printf("  Running full inference (H_CYCLES=%d, L_CYCLES=%d, L_LAYERS=%d)...\n",
           H_CYCLES, L_CYCLES, L_LAYERS);

    int64_t elapsed_us = trm_full_inference(pred_tokens, input_tokens, seq_len, &model, scratch);

    printf("  Time: %.2f ms (%.2f s)\n",
           (float)elapsed_us / 1000.0f, (float)elapsed_us / 1000000.0f);

    heap_caps_free(scratch);
    free(input_tokens);
    free(pred_tokens);
}

/* ============== Serial Evaluation Protocol ============== */

static int read_line(char *buf, int max_len)
{
    int n = 0;
    while (n < max_len - 1) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (c == '\r') continue;
        if (c == '\n') break;
        buf[n++] = (char)c;
    }
    buf[n] = '\0';
    return n;
}

static int parse_ints_from_line(const char *line, int *out, int max_out)
{
    int count = 0;
    const char *p = line;
    while (*p && count < max_out) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[count++] = (int)v;
        p = end;
    }
    return count;
}

static void cmd_eval_text(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }

    printf("\n=== Debug Text Eval Mode ===\n");
    printf("This mode is for MANUAL testing via idf.py monitor.\n");
    printf("Enter seq_len as a number (0 to exit), then paste token IDs (0..%d).\n", VOCAB_SIZE - 1);
    printf("Example:\n");
    printf("  81\n");
    printf("  0 0 0 5 0 ... (81 integers total, spaces or commas ok)\n\n");
    fflush(stdout);

    char line[512];

    while (1) {
        printf("seq_len> ");
        fflush(stdout);
        read_line(line, sizeof(line));

        int tmp = 0;
        if (parse_ints_from_line(line, &tmp, 1) != 1) {
            printf("Please enter a number.\n");
            continue;
        }

        int seq_len = tmp;
        if (seq_len == 0) {
            printf("Exiting debug eval.\n");
            break;
        }
        if (seq_len < 0 || seq_len > MAX_SEQ_LEN) {
            printf("Invalid seq_len. Must be 1..%d\n", MAX_SEQ_LEN);
            continue;
        }

        uint8_t *input_tokens = (uint8_t *)malloc((size_t)seq_len);
        uint8_t *pred_tokens = (uint8_t *)malloc((size_t)seq_len);
        if (!input_tokens || !pred_tokens) {
            printf("ERR_OOM allocating token buffers.\n");
            if (input_tokens) free(input_tokens);
            if (pred_tokens) free(pred_tokens);
            continue;
        }

        printf("tokens> ");
        fflush(stdout);

        int filled = 0;
        while (filled < seq_len) {
            read_line(line, sizeof(line));
            int parsed[256];
            int n = parse_ints_from_line(line, parsed, 256);
            if (n == 0) {
                printf("Need %d more ints...\n", seq_len - filled);
                printf("tokens> ");
                fflush(stdout);
                continue;
            }
            for (int i = 0; i < n && filled < seq_len; i++) {
                int v = parsed[i];
                if (v < 0) v = 0;
                if (v >= VOCAB_SIZE) v = 0;
                input_tokens[filled++] = (uint8_t)v;
            }
            if (filled < seq_len) {
                printf("... %d/%d\n", filled, seq_len);
                printf("tokens> ");
                fflush(stdout);
            }
        }

        int mem_needed = trm_full_inference_mem(seq_len);
        printf("Running inference (seq_len=%d, internal=%d). Need %.2f KB scratch.\n",
               seq_len, seq_len + PUZZLE_EMB_LEN, (float)mem_needed / 1024.0f);
        fflush(stdout);

        float *scratch = (float *)heap_caps_malloc(mem_needed, MALLOC_CAP_SPIRAM);
        if (!scratch) {
            printf("ERR_OOM allocating scratch in PSRAM.\n");
            free(input_tokens);
            free(pred_tokens);
            continue;
        }

        int64_t elapsed_us = trm_full_inference(pred_tokens, input_tokens, seq_len, &model, scratch);
        uint32_t time_ms = (uint32_t)(elapsed_us / 1000);

        printf("TIME_MS %lu\n", (unsigned long)time_ms);
        printf("PRED ");
        for (int i = 0; i < seq_len; i++) {
            printf("%s%u", i ? " " : "", (unsigned)pred_tokens[i]);
        }
        printf("\n");
        fflush(stdout);

        heap_caps_free(scratch);
        free(input_tokens);
        free(pred_tokens);
    }
}

/* Text protocol for Python (line-based, robust framing).
 * Host flow:
 *   - send 'T'
 *   - wait for "READYTXT"
 *   - for each puzzle:
 *       send: "<seq_len> [act_steps]\\n"   (act_steps defaults to 1 if omitted)
 *             "<t0> <t1> ... <t{seq_len-1}>\\n"  (may span multiple lines)
 *       recv: "TIME_MS <ms>\\n"
 *             "STEPS_USED <n>\\n"
 *             "PRED <p0> <p1> ... <p{seq_len-1}>\\n"
 *   - send: "0\\n" to exit
 */
static void cmd_eval_text_proto(void)
{
    if (!model_loaded) {
        printf("ERR not_loaded\n");
        fflush(stdout);
        return;
    }

    printf("READYTXT\n");
    fflush(stdout);

    char line[512];
    while (1) {
        /* Read seq_len [act_steps] line */
        int nline = read_line(line, sizeof(line));
        (void)nline;

        int header_vals[2] = {0, 1};  /* defaults: seq_len=0, act_steps=1 */
        int n_header = parse_ints_from_line(line, header_vals, 2);
        if (n_header < 1) {
            /* ignore garbage lines */
            continue;
        }

        int seq_len = header_vals[0];
        int act_steps = (n_header >= 2) ? header_vals[1] : 1;

        if (seq_len == 0) {
            printf("EVAL_DONE\n");
            fflush(stdout);
            break;
        }
        if (seq_len < 0) seq_len = 0;
        if (seq_len > MAX_SEQ_LEN) seq_len = MAX_SEQ_LEN;
        if (act_steps < 1) act_steps = 1;
        if (act_steps > HALT_MAX_STEPS) act_steps = HALT_MAX_STEPS;

        uint8_t *input_tokens = (uint8_t *)malloc((size_t)seq_len);
        uint8_t *pred_tokens = (uint8_t *)malloc((size_t)seq_len);
        if (!input_tokens || !pred_tokens) {
            printf("ERR oom_tokens\n");
            fflush(stdout);
            if (input_tokens) free(input_tokens);
            if (pred_tokens) free(pred_tokens);
            continue;
        }

        /* Read token ints until we have seq_len */
        int filled = 0;
        while (filled < seq_len) {
            read_line(line, sizeof(line));
            int parsed[256];
            int k = parse_ints_from_line(line, parsed, 256);
            if (k <= 0) continue;
            for (int i = 0; i < k && filled < seq_len; i++) {
                int v = parsed[i];
                if (v < 0) v = 0;
                if (v >= VOCAB_SIZE) v = 0;
                input_tokens[filled++] = (uint8_t)v;
            }
        }

        int mem_needed = trm_full_inference_mem(seq_len);
        float *scratch = (float *)heap_caps_malloc(mem_needed, MALLOC_CAP_SPIRAM);
        if (!scratch) {
            printf("ERR oom_scratch\n");
            fflush(stdout);
            free(input_tokens);
            free(pred_tokens);
            continue;
        }

        int64_t elapsed_us;
        int steps_used = 1;

        if (act_steps <= 1) {
            elapsed_us = trm_full_inference(pred_tokens, input_tokens, seq_len, &model, scratch);
        } else {
            elapsed_us = trm_full_inference_act(pred_tokens, input_tokens, seq_len,
                                                &model, scratch, act_steps, &steps_used);
        }
        uint32_t time_ms = (uint32_t)(elapsed_us / 1000);

        printf("TIME_MS %lu\n", (unsigned long)time_ms);
        printf("STEPS_USED %d\n", steps_used);
        printf("PRED");
        for (int i = 0; i < seq_len; i++) {
            printf(" %u", (unsigned)pred_tokens[i]);
        }
        printf("\n");
        fflush(stdout);

        heap_caps_free(scratch);
        free(input_tokens);
        free(pred_tokens);
    }
}

static void cmd_eval_serial(void)
{
    if (!model_loaded) {
        printf("Load model first with [l].\n");
        return;
    }

    printf("READY\n");
    fflush(stdout);

    while (1) {
        /* Read seq_len (uint16_t little-endian) via raw bytes */
        uint8_t len_buf[2];
        int b0 = EOF, b1 = EOF;

        /* Wait for first byte */
        while ((b0 = getchar()) == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        len_buf[0] = (uint8_t)b0;

        /* Wait for second byte */
        while ((b1 = getchar()) == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        len_buf[1] = (uint8_t)b1;

        uint16_t seq_len = (uint16_t)len_buf[0] | ((uint16_t)len_buf[1] << 8);

        /* Exit on seq_len == 0 */
        if (seq_len == 0) {
            printf("EVAL_DONE\n");
            fflush(stdout);
            break;
        }

        /* Clamp to MAX_SEQ_LEN */
        if (seq_len > MAX_SEQ_LEN) {
            seq_len = MAX_SEQ_LEN;
        }

        /* Read input tokens */
        uint8_t *input_tokens = (uint8_t *)malloc(seq_len);
        if (!input_tokens) {
            printf("ERR_OOM\n");
            fflush(stdout);
            continue;
        }

        for (int i = 0; i < seq_len; i++) {
            int c;
            while ((c = getchar()) == EOF) {
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            input_tokens[i] = (uint8_t)c;
        }

        /* Allocate inference buffers */
        int mem_needed = trm_full_inference_mem(seq_len);
        float *scratch = (float *)heap_caps_malloc(mem_needed, MALLOC_CAP_SPIRAM);
        uint8_t *pred_tokens = (uint8_t *)malloc(seq_len);

        if (!scratch || !pred_tokens) {
            printf("ERR_OOM\n");
            fflush(stdout);
            free(input_tokens);
            if (scratch) heap_caps_free(scratch);
            if (pred_tokens) free(pred_tokens);
            continue;
        }

        /* Run full inference */
        int64_t elapsed_us = trm_full_inference(pred_tokens, input_tokens, seq_len, &model, scratch);
        uint32_t time_ms = (uint32_t)(elapsed_us / 1000);

        /* Send response: [uint16_t seq_len] [pred_tokens...] [uint32_t time_ms] */
        uint8_t resp_len[2] = {(uint8_t)(seq_len & 0xFF), (uint8_t)((seq_len >> 8) & 0xFF)};
        fwrite(resp_len, 1, 2, stdout);
        fwrite(pred_tokens, 1, seq_len, stdout);
        uint8_t time_buf[4] = {
            (uint8_t)(time_ms & 0xFF),
            (uint8_t)((time_ms >> 8) & 0xFF),
            (uint8_t)((time_ms >> 16) & 0xFF),
            (uint8_t)((time_ms >> 24) & 0xFF)
        };
        fwrite(time_buf, 1, 4, stdout);
        fflush(stdout);

        /* Cleanup */
        free(input_tokens);
        free(pred_tokens);
        heap_caps_free(scratch);
    }
}

void app_main(void)
{
    print_banner();

    /* Print initial chip info */
    ESP_LOGI(TAG, "ESP32-S3 TRM Inference Engine");
    ESP_LOGI(TAG, "Model config: hidden=%d, heads=%d, layers=%d, inter=%d",
             HIDDEN_SIZE, NUM_HEADS, L_LAYERS, MLP_INTER);
    ESP_LOGI(TAG, "Ternary packing: %d weights/byte", WEIGHTS_PER_BYTE);

    print_memory();
    print_menu();

    /* Simple UART command loop */
    char cmd;
    while (1) {
        int c = getchar();
        if (c == EOF) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        cmd = (char)c;

        switch (cmd) {
            case 'm': print_memory(); break;
            case 'b': cmd_benchmark_kernel(); break;
            case 'l': cmd_load_model(); break;
            case 'u': cmd_unpack_weights(); break;
            case 'r': cmd_benchmark_inference(); break;
            case 'f': cmd_full_inference_test(); break;
            case 'F': cmd_full_inference_test_81(); break;
            case 'e': case 'E': cmd_eval_serial(); break;
            case 'D': cmd_eval_text(); break;
            case 'T': cmd_eval_text_proto(); break;
            case 'i': cmd_model_info(); break;
            case 'h': print_menu(); break;
            case '\n': case '\r': break;
            default:
                printf("Unknown command: '%c'. Press 'h' for help.\n", cmd);
                break;
        }

        if (cmd != '\n' && cmd != '\r') {
            print_menu();
        }
    }
}

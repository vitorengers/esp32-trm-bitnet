/**
 * Model Weight Loader for ESP32-S3
 * 
 * Reads a flat binary file from the SPIFFS data partition and
 * populates the TRMModel structure with pointers into PSRAM.
 * 
 * Binary file format (written by export_ternary.py):
 *   - Header: magic (4B) + version (4B) + vocab_size (4B) + num_layers (4B)
 *   - For each tensor: size_bytes (4B) + scale (4B) + data (size_bytes)
 *   - Tensors appear in the order defined in export_ternary.py
 */

#include "model_loader.h"
#include "trm_ternary.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_heap_caps.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static const char *TAG = "model_loader";

#define MODEL_MAGIC 0x54524D31  /* "TRM1" */
#define MODEL_VERSION_V1 1
#define MODEL_VERSION_V2 2
#define MODEL_VERSION_V3 3
#define MODEL_VERSION_V4 4

/* Helper: read bytes from file, allocate in PSRAM */
static void *read_tensor_psram(FILE *f, uint32_t *out_size, float *out_scale)
{
    uint32_t size;
    float scale;

    if (fread(&size, 4, 1, f) != 1) return NULL;
    if (fread(&scale, 4, 1, f) != 1) return NULL;

    void *data = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    if (!data) {
        ESP_LOGE(TAG, "Failed to allocate %lu bytes in PSRAM", (unsigned long)size);
        return NULL;
    }

    if (fread(data, 1, size, f) != size) {
        heap_caps_free(data);
        return NULL;
    }

    if (out_size) *out_size = size;
    if (out_scale) *out_scale = scale;
    return data;
}

/* Helper: allocate aligned INT8 buffer and unpack a ternary weight matrix */
static int8_t *alloc_and_unpack(const uint8_t *packed, int M, int K, size_t *out_bytes)
{
    size_t nbytes = (size_t)M * (size_t)K;
    /* 16-byte alignment required for SIMD ee.vld.128.ip */
    int8_t *buf = (int8_t *)heap_caps_aligned_alloc(16, nbytes, MALLOC_CAP_SPIRAM);
    if (!buf) return NULL;

    unpack_ternary_matrix(buf, packed, M, K);
    if (out_bytes) *out_bytes = nbytes;
    return buf;
}

int model_load(TRMModel *model, const char *partition_label)
{
    memset(model, 0, sizeof(TRMModel));

    /* Mount SPIFFS */
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/model",
        .partition_label = partition_label,
        .max_files = 2,
        .format_if_mount_failed = false,
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount SPIFFS: %s", esp_err_to_name(ret));
        return -1;
    }

    /* Open model file */
    FILE *f = fopen("/model/trm_ternary.bin", "rb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open model file");
        esp_vfs_spiffs_unregister(partition_label);
        return -1;
    }

    /* Read header */
    uint32_t magic, version, vocab_size, num_layers;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&vocab_size, 4, 1, f);
    fread(&num_layers, 4, 1, f);

    if (magic != MODEL_MAGIC || (version != MODEL_VERSION_V1 && version != MODEL_VERSION_V2 && version != MODEL_VERSION_V3 && version != MODEL_VERSION_V4)) {
        ESP_LOGE(TAG, "Invalid model file: magic=0x%08lx version=%lu",
                 (unsigned long)magic, (unsigned long)version);
        fclose(f);
        esp_vfs_spiffs_unregister(partition_label);
        return -1;
    }

    model->vocab_size = (int)vocab_size;
    ESP_LOGI(TAG, "Loading TRM model v%lu: vocab=%lu, layers=%lu",
             (unsigned long)version, (unsigned long)vocab_size, (unsigned long)num_layers);

    size_t total = 0;

    /* Read embedding */
    uint32_t sz;
    model->embed_tokens = (int8_t *)read_tensor_psram(f, &sz, &model->embed_scale);
    if (!model->embed_tokens) goto fail;
    total += sz;
    ESP_LOGI(TAG, "  embed_tokens: %lu bytes", (unsigned long)sz);

    /* Read puzzle_emb (v3+) */
    if (version >= MODEL_VERSION_V3) {
        float dummy_scale;
        model->puzzle_emb = (float *)read_tensor_psram(f, &sz, &dummy_scale);
        if (!model->puzzle_emb) goto fail;
        total += sz;
        ESP_LOGI(TAG, "  puzzle_emb: %lu bytes", (unsigned long)sz);
    } else {
        /* v1/v2 fallback: no puzzle_emb, allocate zeros */
        ESP_LOGW(TAG, "Model v%lu: no puzzle_emb — using zeros", (unsigned long)version);
        model->puzzle_emb = (float *)heap_caps_calloc(
            PUZZLE_EMB_LEN * HIDDEN_SIZE, sizeof(float), MALLOC_CAP_SPIRAM);
        if (!model->puzzle_emb) goto fail;
        total += PUZZLE_EMB_LEN * HIDDEN_SIZE * sizeof(float);
    }

    /* Read transformer blocks */
    for (int i = 0; i < L_LAYERS && i < (int)num_layers; i++) {
        if (version >= MODEL_VERSION_V3) {
            /* v3: norm_weight before each ternary projection */
            float dummy_scale;

            /* QKV norm_weight */
            model->blocks[i].qkv_norm_weight = (float *)read_tensor_psram(f, &sz, &dummy_scale);
            if (!model->blocks[i].qkv_norm_weight) goto fail;
            total += sz;

            /* QKV projection */
            model->blocks[i].attn.qkv_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].attn.qkv_scale);
            if (!model->blocks[i].attn.qkv_weights) goto fail;
            total += sz;

            /* O norm_weight */
            model->blocks[i].o_norm_weight = (float *)read_tensor_psram(f, &sz, &dummy_scale);
            if (!model->blocks[i].o_norm_weight) goto fail;
            total += sz;

            /* Output projection */
            model->blocks[i].attn.o_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].attn.o_scale);
            if (!model->blocks[i].attn.o_weights) goto fail;
            total += sz;

            /* gate_up norm_weight */
            model->blocks[i].gate_up_norm_weight = (float *)read_tensor_psram(f, &sz, &dummy_scale);
            if (!model->blocks[i].gate_up_norm_weight) goto fail;
            total += sz;

            /* Gate+Up projection */
            model->blocks[i].gate_up_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].gate_up_scale);
            if (!model->blocks[i].gate_up_weights) goto fail;
            total += sz;

            /* down norm_weight */
            model->blocks[i].down_norm_weight = (float *)read_tensor_psram(f, &sz, &dummy_scale);
            if (!model->blocks[i].down_norm_weight) goto fail;
            total += sz;

            /* Down projection */
            model->blocks[i].down_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].down_scale);
            if (!model->blocks[i].down_weights) goto fail;
            total += sz;
        } else {
            /* v1/v2: no norm_weight — NULL (fallback to unit weight in rmsnorm_weighted) */
            model->blocks[i].qkv_norm_weight = NULL;
            model->blocks[i].o_norm_weight = NULL;
            model->blocks[i].gate_up_norm_weight = NULL;
            model->blocks[i].down_norm_weight = NULL;

            /* QKV projection */
            model->blocks[i].attn.qkv_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].attn.qkv_scale);
            if (!model->blocks[i].attn.qkv_weights) goto fail;
            total += sz;

            /* Output projection */
            model->blocks[i].attn.o_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].attn.o_scale);
            if (!model->blocks[i].attn.o_weights) goto fail;
            total += sz;

            /* Gate+Up projection */
            model->blocks[i].gate_up_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].gate_up_scale);
            if (!model->blocks[i].gate_up_weights) goto fail;
            total += sz;

            /* Down projection */
            model->blocks[i].down_weights = (uint8_t *)read_tensor_psram(f, &sz, &model->blocks[i].down_scale);
            if (!model->blocks[i].down_weights) goto fail;
            total += sz;
        }

        ESP_LOGI(TAG, "  block[%d]: loaded (norm_weight=%s)", i,
                 model->blocks[i].qkv_norm_weight ? "YES" : "NO");
    }

    /* Read LM head */
    model->lm_head = (int8_t *)read_tensor_psram(f, &sz, &model->lm_head_scale);
    if (!model->lm_head) goto fail;
    total += sz;
    ESP_LOGI(TAG, "  lm_head: %lu bytes", (unsigned long)sz);

    /* Read H_init and L_init (version 2+) */
    if (version >= MODEL_VERSION_V2) {
        float dummy_scale;
        model->h_init = (float *)read_tensor_psram(f, &sz, &dummy_scale);
        if (!model->h_init) goto fail;
        total += sz;
        ESP_LOGI(TAG, "  h_init: %lu bytes", (unsigned long)sz);

        model->l_init = (float *)read_tensor_psram(f, &sz, &dummy_scale);
        if (!model->l_init) goto fail;
        total += sz;
        ESP_LOGI(TAG, "  l_init: %lu bytes", (unsigned long)sz);
    } else {
        /* v1 fallback: allocate and zero-initialize */
        ESP_LOGW(TAG, "Model v1: no H_init/L_init — using zeros");
        model->h_init = (float *)heap_caps_calloc(HIDDEN_SIZE, sizeof(float), MALLOC_CAP_SPIRAM);
        model->l_init = (float *)heap_caps_calloc(HIDDEN_SIZE, sizeof(float), MALLOC_CAP_SPIRAM);
        if (!model->h_init || !model->l_init) goto fail;
        total += 2 * HIDDEN_SIZE * sizeof(float);
    }

    /* Read Q-head weight and bias (v4+) */
    if (version >= MODEL_VERSION_V4) {
        float dummy_scale;
        model->q_head_weight = (float *)read_tensor_psram(f, &sz, &dummy_scale);
        if (!model->q_head_weight) goto fail;
        total += sz;
        ESP_LOGI(TAG, "  q_head_weight: %lu bytes", (unsigned long)sz);

        model->q_head_bias = (float *)read_tensor_psram(f, &sz, &dummy_scale);
        if (!model->q_head_bias) goto fail;
        total += sz;
        ESP_LOGI(TAG, "  q_head_bias: %lu bytes", (unsigned long)sz);
    } else {
        ESP_LOGW(TAG, "Model v%lu: no q_head — ACT halting disabled", (unsigned long)version);
        model->q_head_weight = NULL;
        model->q_head_bias = NULL;
    }

    model->total_allocated = total;
    model->weights_unpacked = false;
    model->unpacked_allocated = 0;

    /* Initialize RoPE */
    rope_init(&model->rope);

    fclose(f);
    ESP_LOGI(TAG, "Model loaded: %.2f MB in PSRAM", (float)total / (1024 * 1024));
    return 0;

fail:
    ESP_LOGE(TAG, "Failed to load model");
    fclose(f);
    model_free(model);
    esp_vfs_spiffs_unregister(partition_label);
    return -1;
}

int model_unpack_weights(TRMModel *model)
{
    if (model->weights_unpacked) {
        ESP_LOGW(TAG, "Weights already unpacked");
        return 0;
    }

    ESP_LOGI(TAG, "Pre-unpacking ternary weights to INT8 (replace-in-place)...");
    ESP_LOGI(TAG, "  Free PSRAM before: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    size_t total_unpacked = 0;
    size_t packed_freed = 0;

    for (int i = 0; i < L_LAYERS; i++) {
        size_t nbytes;

        /* QKV: [QKV_OUT_SIZE, HIDDEN_SIZE] */
        model->blocks[i].attn.qkv_unpacked = alloc_and_unpack(
            model->blocks[i].attn.qkv_weights, QKV_OUT_SIZE, HIDDEN_SIZE, &nbytes);
        if (!model->blocks[i].attn.qkv_unpacked) goto fail;
        total_unpacked += nbytes;
        /* Free packed — no longer needed */
        packed_freed += QKV_OUT_SIZE * HIDDEN_SIZE / WEIGHTS_PER_BYTE;
        heap_caps_free((void *)model->blocks[i].attn.qkv_weights);
        model->blocks[i].attn.qkv_weights = NULL;

        /* O: [HIDDEN_SIZE, ATTN_OUT_SIZE] */
        model->blocks[i].attn.o_unpacked = alloc_and_unpack(
            model->blocks[i].attn.o_weights, HIDDEN_SIZE, ATTN_OUT_SIZE, &nbytes);
        if (!model->blocks[i].attn.o_unpacked) goto fail;
        total_unpacked += nbytes;
        packed_freed += HIDDEN_SIZE * ATTN_OUT_SIZE / WEIGHTS_PER_BYTE;
        heap_caps_free((void *)model->blocks[i].attn.o_weights);
        model->blocks[i].attn.o_weights = NULL;

        /* gate_up: [GATE_UP_SIZE, HIDDEN_SIZE] */
        model->blocks[i].gate_up_unpacked = alloc_and_unpack(
            model->blocks[i].gate_up_weights, GATE_UP_SIZE, HIDDEN_SIZE, &nbytes);
        if (!model->blocks[i].gate_up_unpacked) goto fail;
        total_unpacked += nbytes;
        packed_freed += GATE_UP_SIZE * HIDDEN_SIZE / WEIGHTS_PER_BYTE;
        heap_caps_free(model->blocks[i].gate_up_weights);
        model->blocks[i].gate_up_weights = NULL;

        /* down: [HIDDEN_SIZE, MLP_INTER] */
        model->blocks[i].down_unpacked = alloc_and_unpack(
            model->blocks[i].down_weights, HIDDEN_SIZE, MLP_INTER, &nbytes);
        if (!model->blocks[i].down_unpacked) goto fail;
        total_unpacked += nbytes;
        packed_freed += HIDDEN_SIZE * MLP_INTER / WEIGHTS_PER_BYTE;
        heap_caps_free(model->blocks[i].down_weights);
        model->blocks[i].down_weights = NULL;

        ESP_LOGI(TAG, "  block[%d]: unpacked (freed %.2f KB packed)",
                 i, (float)packed_freed / 1024);
    }

    model->weights_unpacked = true;
    model->unpacked_allocated = total_unpacked;

    ESP_LOGI(TAG, "Unpacked %.2f MB, freed %.2f MB packed",
             (float)total_unpacked / (1024 * 1024),
             (float)packed_freed / (1024 * 1024));
    ESP_LOGI(TAG, "  Net PSRAM increase: %.2f MB",
             (float)(total_unpacked - packed_freed) / (1024 * 1024));
    ESP_LOGI(TAG, "  Free PSRAM after: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    return 0;

fail:
    ESP_LOGE(TAG, "Failed to allocate unpacked weights — out of PSRAM");
    /* Note: cannot roll back — some packed weights already freed.
     * Model is in a partially unpacked state. Reload needed. */
    return -1;
}

void model_free_unpacked(TRMModel *model)
{
    for (int i = 0; i < L_LAYERS; i++) {
        if (model->blocks[i].attn.qkv_unpacked) {
            heap_caps_free(model->blocks[i].attn.qkv_unpacked);
            model->blocks[i].attn.qkv_unpacked = NULL;
        }
        if (model->blocks[i].attn.o_unpacked) {
            heap_caps_free(model->blocks[i].attn.o_unpacked);
            model->blocks[i].attn.o_unpacked = NULL;
        }
        if (model->blocks[i].gate_up_unpacked) {
            heap_caps_free(model->blocks[i].gate_up_unpacked);
            model->blocks[i].gate_up_unpacked = NULL;
        }
        if (model->blocks[i].down_unpacked) {
            heap_caps_free(model->blocks[i].down_unpacked);
            model->blocks[i].down_unpacked = NULL;
        }
    }
    model->weights_unpacked = false;
    model->unpacked_allocated = 0;
}

void model_free(TRMModel *model)
{
    model_free_unpacked(model);
    if (model->embed_tokens) heap_caps_free(model->embed_tokens);
    if (model->puzzle_emb) heap_caps_free(model->puzzle_emb);
    for (int i = 0; i < L_LAYERS; i++) {
        if (model->blocks[i].attn.qkv_weights) heap_caps_free((void *)model->blocks[i].attn.qkv_weights);
        if (model->blocks[i].attn.o_weights) heap_caps_free((void *)model->blocks[i].attn.o_weights);
        if (model->blocks[i].gate_up_weights) heap_caps_free(model->blocks[i].gate_up_weights);
        if (model->blocks[i].down_weights) heap_caps_free(model->blocks[i].down_weights);
        if (model->blocks[i].qkv_norm_weight) heap_caps_free(model->blocks[i].qkv_norm_weight);
        if (model->blocks[i].o_norm_weight) heap_caps_free(model->blocks[i].o_norm_weight);
        if (model->blocks[i].gate_up_norm_weight) heap_caps_free(model->blocks[i].gate_up_norm_weight);
        if (model->blocks[i].down_norm_weight) heap_caps_free(model->blocks[i].down_norm_weight);
    }
    if (model->lm_head) heap_caps_free(model->lm_head);
    if (model->h_init) heap_caps_free(model->h_init);
    if (model->l_init) heap_caps_free(model->l_init);
    if (model->q_head_weight) heap_caps_free(model->q_head_weight);
    if (model->q_head_bias) heap_caps_free(model->q_head_bias);
    rope_free(&model->rope);
    memset(model, 0, sizeof(TRMModel));
}

void model_print_stats(const TRMModel *model)
{
    ESP_LOGI(TAG, "=== Model Stats ===");
    ESP_LOGI(TAG, "  Total PSRAM (packed): %.2f MB", (float)model->total_allocated / (1024 * 1024));
    if (model->weights_unpacked) {
        ESP_LOGI(TAG, "  Unpacked INT8: %.2f MB", (float)model->unpacked_allocated / (1024 * 1024));
    }
    ESP_LOGI(TAG, "  Weights unpacked: %s", model->weights_unpacked ? "YES" : "NO");
    ESP_LOGI(TAG, "  Vocab size: %d", model->vocab_size);
    ESP_LOGI(TAG, "  Free PSRAM: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "  Free internal: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

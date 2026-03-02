idf_component_register(
    SRCS
        "main.c"
        "trm_engine.c"
        "trm_ternary.c"
        "trm_ternary_simd.c"
        "simd_dotprod.S"
        "attention.c"
        "rmsnorm.c"
        "benchmark.c"
        "model_loader.c"
    INCLUDE_DIRS "."
    REQUIRES
        spiffs
        esp_timer
        heap
        esp_psram
)

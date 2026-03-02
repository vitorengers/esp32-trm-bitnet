# ESP-IDF project CMakeLists.txt for TRM Ternary Inference
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(trm_inference)

# Activation quantization: roundf() (default) or truncation toward zero
# Build with: idf.py build -DUSE_TRUNC=1
if(DEFINED USE_TRUNC AND USE_TRUNC)
    idf_build_set_property(COMPILE_DEFINITIONS "USE_TRUNC=1" APPEND)
    message(STATUS "INT8 activation quantization: TRUNCATION (int)(x)")
else()
    message(STATUS "INT8 activation quantization: ROUNDING roundf(x)")
endif()

# Matmul kernel: SIMD ternary (default) or float32 multiply-accumulate
# Build with: idf.py build -DUSE_FLOAT32_MATMUL=1
if(DEFINED USE_FLOAT32_MATMUL AND USE_FLOAT32_MATMUL)
    idf_build_set_property(COMPILE_DEFINITIONS "USE_FLOAT32_MATMUL=1" APPEND)
    message(STATUS "Matmul kernel: FLOAT32 multiply-accumulate (dequantized INT8)")
else()
    message(STATUS "Matmul kernel: SIMD ternary INT8 dot product")
endif()

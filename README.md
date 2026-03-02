# ESP32-S3 TRM Inference Engine

C firmware and evaluation tooling for running **Tiny Recursive Reasoning Models (TRM)** with **BitNet 1.58-bit ternary quantization** on ESP32-S3 microcontrollers. Implements the TRM-Att architecture with ternary weight inference using Xtensa PIE SIMD instructions.

**Target hardware:** ESP32-S3 DevKitC-1 with 16 MB Flash and 8 MB PSRAM (N16R8).

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Building the Firmware](#building-the-firmware)
- [Exporting a Model](#exporting-a-model)
- [Flashing the Device](#flashing-the-device)
- [Running Evaluations](#running-evaluations)
- [Build Variants](#build-variants)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project deploys a quantized TRM model for Sudoku solving on the ESP32-S3:

- **Model:** TRM-Att with BitNet 1.58-bit (ternary) quantization, 2 transformer layers, hidden size 512
- **Inference modes:** Fixed-step (1 or 16 ACT steps) or haltable (adaptive computation)
- **Quantization:** Weights in `{-1, 0, +1}`; activations quantized to INT8 via `round()` or `trunc()`
- **Kernel:** Hand-written Xtensa assembly SIMD dot product for ternary matmul

Typical inference time: ~4–6 minutes per Sudoku puzzle (81 positions, 16 ACT steps).

---

## Prerequisites

### Hardware

- **ESP32-S3 DevKitC-1** (N16R8: 16 MB Flash, 8 MB PSRAM)
- USB cable

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| **ESP-IDF** | v5.3.2 | Build system and firmware |
| **Python** | 3.9+ | Scripts, export, evaluation |
| **pyserial** | — | Serial communication for evaluation |

### Python Dependencies

```bash
pip install pyserial
```

For PyTorch baseline evaluation and model export (optional, requires trained checkpoint):

```bash
pip install torch pyyaml numpy
```

---

## Repository Structure

```
esp32_trm/
├── firmware/           # ESP-IDF project
│   ├── main/           # C source: inference engine, attention, matmul, model loader
│   ├── model_data/     # Export output: trm_ternary.bin (created by export script)
│   ├── flash_all.sh    # Flash firmware + model to device
│   ├── partitions.csv  # Custom partition table (app + model SPIFFS)
│   └── sdkconfig.defaults
├── export/             # PyTorch → binary conversion
│   └── export_ternary.py
├── scripts/            # Evaluation and utilities
│   ├── evaluate_serial.py      # ESP32 serial evaluation (collect predictions)
│   ├── evaluate_pytorch_baseline.py  # PC reference evaluation
│   ├── select_test_subset.py   # Create test subset from dataset
│   ├── split_subset_into_n.py  # Split subset for parallel evaluation
│   └── split_remaining_into_4.py
├── results/            # Evaluation results (JSON)
└── README.md
```

---

## Quick Start

If you already have a built firmware and exported model:

```bash
# 1. Enter firmware directory
cd esp32_trm/firmware

# 2. Source ESP-IDF
source ~/esp/esp-idf-v5.3.2/export.sh

# 3. Build
idf.py build

# 4. Generate SPIFFS image (model must be in model_data/trm_ternary.bin)
python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 model_data build/model_spiffs.bin

# 5. Flash (default port: /dev/ttyACM0)
bash flash_all.sh /dev/ttyACM0

# 6. Open serial monitor
idf.py -p /dev/ttyACM0 monitor
```

In the monitor: press `l` to load the model, then `T` for text evaluation mode, or use the Python script for automated evaluation.

---

## Detailed Setup

### 1. Install ESP-IDF v5.3.2

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install -y git wget flex bison gperf python3-venv cmake ninja-build \
    ccache libffi-dev libssl-dev dfu-util libusb-1.0-0

# Clone and install
mkdir -p ~/esp && cd ~/esp
git clone -b v5.3.2 --recursive --depth 1 https://github.com/espressif/esp-idf.git esp-idf-v5.3.2
cd esp-idf-v5.3.2
./install.sh esp32s3

# Verify
source export.sh
idf.py --version
```

### 2. WSL2 / USB Access (Linux)

If using WSL2, the ESP32 is connected via Windows. Use **usbipd** to attach it:

**On Windows (PowerShell as Administrator):**
```powershell
# Install
winget install usbipd

# List devices
usbipd list

# Attach to WSL (replace <BUSID> with your ESP32's bus ID)
usbipd attach --wsl --busid <BUSID> --auto-attach
```

`--auto-attach` is important: the ESP32 re-enumerates on reset, and this keeps the connection.

**In WSL:** After attaching, the device appears as `/dev/ttyACM0` (or similar).

### 3. Enter Download Mode

To flash the board:

1. Hold the **BOOT** button
2. Press and release **RST**
3. Release **BOOT**
4. Wait ~3 seconds for USB to reconnect (with `--auto-attach`)

---

## Building the Firmware

```bash
cd esp32_trm/firmware
source ~/esp/esp-idf-v5.3.2/export.sh
idf.py set-target esp32s3   # Only needed once
idf.py build
```

**Build variants** (see [Build Variants](#build-variants)):

```bash
# Round activation quantization (default)
idf.py build

# Truncation activation quantization
idf.py build -DUSE_TRUNC=1

# Float32 matmul (dequantized, slower, for debugging)
idf.py build -DUSE_FLOAT32_MATMUL=1
```

---

## Exporting a Model

You need a PyTorch checkpoint from the [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) training pipeline (with BitNet quantization). The export script converts it to the flat binary format the firmware expects.

**Recommended workspace layout:**

```
trm-bitnet/                    # Or your project root
├── TinyRecursiveModels/       # With trained checkpoints
│   ├── checkpoints/
│   │   └── Sudoku-extreme-1k-aug-1000-ACT-torch/
│   │       └── bitnet_round_v3/step_58590/
│   └── data/sudoku-extreme-1k-aug-1000/test/
└── esp32_trm/                 # This repo
```

**Export command** (run from project root):

```bash
python esp32_trm/export/export_ternary.py \
    --checkpoint TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/bitnet_round_v3/step_58590 \
    --output esp32_trm/firmware/model_data/trm_ternary.bin \
    --verify
```

**Dependencies:** `torch`, `numpy`, `pyyaml`. The script infers model dimensions from the checkpoint.

**Important:** Use the checkpoint that matches your firmware’s quantization:

- **Round firmware** (default): checkpoint trained with `BITNET_USE_TRUNC=0`
- **Trunc firmware** (`-DUSE_TRUNC=1`): checkpoint trained with `BITNET_USE_TRUNC=1`

---

## Flashing the Device

Before flashing, generate the SPIFFS image from the model binary:

```bash
cd esp32_trm/firmware
source ~/esp/esp-idf-v5.3.2/export.sh

# Create SPIFFS image (0xCF0000 = partition size from partitions.csv)
python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 model_data build/model_spiffs.bin

# Flash everything
bash flash_all.sh /dev/ttyACM0
```

Or manually:

```bash
python -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0 build/bootloader/bootloader.bin \
    0x8000 build/partition_table/partition-table.bin \
    0x10000 build/trm_inference.bin \
    0x310000 build/model_spiffs.bin
```

**Flashing only the model** (after firmware is already flashed):

```bash
python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 model_data build/model_spiffs.bin
python -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 write_flash 0x310000 build/model_spiffs.bin
```

---

## Running Evaluations

All scripts assume you run them from the **`esp32_trm`** directory.

### 1. Create a Test Subset

Requires TinyRecursiveModels data:

```bash
cd esp32_trm
python scripts/select_test_subset.py \
    --data-dir ../TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000/test \
    --output test_subset.json \
    --num-puzzles 200
```

*(Or use a pre-generated `test_subset.json` if available.)*

### 2. ESP32 Serial Evaluation

**Close the ESP-IDF monitor** before running; only one process can use the serial port.

```bash
cd esp32_trm
python scripts/evaluate_serial.py \
    --port /dev/ttyACM0 \
    --test-subset test_subset.json \
    --output results/results_esp32.json
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `/dev/ttyACM0` | Serial port |
| `--test-subset` | `test_subset.json` | Input puzzle subset |
| `--output` | `results/results_esp32.json` | Output results file |
| `--resume-output` | — | Resume from existing results file |
| `--timeout` | 1200 | Per-puzzle timeout (seconds) |
| `--auto-load` | — | Send `l` to load model if ESP32 resets on connect |
| `--protocol` | `binary` | `binary` or `text` (text supports ACT steps) |
| `--act-steps` | 1 | ACT steps when using `--protocol text` |

**Resuming a partial run:**

```bash
python scripts/evaluate_serial.py \
    --port /dev/ttyACM0 \
    --test-subset test_subset.json \
    --output results/results_esp32.json \
    --resume-output results/results_esp32.json \
    --timeout 6000
```

### 3. PyTorch Baseline Evaluation (PC Reference)

Runs the same model in PyTorch on the host for comparison. Requires **TinyRecursiveModels** (with the TRM + BitNet code). The script looks for `TinyRecursiveModels` inside `esp32_trm/`. If you use the full `trm-bitnet` layout with both repos as siblings, create a symlink:

```bash
cd esp32_trm
ln -s ../TinyRecursiveModels TinyRecursiveModels
```

Then run:

```bash
cd esp32_trm
python scripts/evaluate_pytorch_baseline.py \
    --checkpoint ../TinyRecursiveModels/checkpoints/.../step_58590 \
    --config ../TinyRecursiveModels/checkpoints/.../all_config.yaml \
    --test-subset test_subset.json \
    --output-dir results \
    --act-steps 16 \
    --haltable
```

### 4. Splitting for Parallel Evaluation

To run on multiple boards in parallel:

```bash
# Split a subset into N chunks
python scripts/split_subset_into_n.py \
    --subset test_subset.json \
    --num-chunks 4 \
    --out-dir . \
    --out-prefix test_subset_chunk_

# Split remaining puzzles after partial runs
python scripts/split_remaining_into_4.py \
    --subset-a test_subset_split_a.json \
    --subset-b test_subset_split_b.json \
    --results-a results/results_esp32_a.json \
    --results-b results/results_esp32_b.json \
    --out-dir . \
    --out-prefix test_subset_remaining_
```

---

## Build Variants

| Variant | CMake flag | Activation quantization | Use case |
|---------|------------|--------------------------|----------|
| Default | (none) | `roundf(x)` | Round-trained BitNet checkpoint |
| Trunc | `-DUSE_TRUNC=1` | `(int)(x)` truncation | Trunc-trained checkpoint |
| Float32 matmul | `-DUSE_FLOAT32_MATMUL=1` | — | Debug / baseline (slower) |

Example:

```bash
idf.py build -DUSE_TRUNC=1
```

---

## Troubleshooting

### "No such file or directory: build/model_spiffs.bin"

Generate the SPIFFS image before flashing:

```bash
cd esp32_trm/firmware
python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 model_data build/model_spiffs.bin
```

### "Did not receive READY from ESP32"

- Ensure the monitor is closed (`idf.py monitor` holds the serial port).
- Use `--auto-load` so the script sends `l` to load the model if the device resets.
- Check the correct serial port: `ls /dev/ttyACM*`.

### USB device not found in WSL2

- Run `usbipd attach --wsl --busid <BUSID> --auto-attach` on Windows.
- `--auto-attach` keeps the connection across ESP32 resets.

### Model load fails on device

- Verify `trm_ternary.bin` exists in `firmware/model_data/`.
- Check that the exported model matches the firmware (round vs trunc).
- Ensure the SPIFFS image was regenerated after exporting a new model.

### Serial permission denied

```bash
sudo usermod -aG dialout $USER
# Log out and back in
```

---

## License

See repository for license information. This project builds on the [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) architecture and [BitNet](https://github.com/microsoft/BitNet) quantization.

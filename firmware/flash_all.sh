#!/bin/bash
# Flash TRM firmware + model data to ESP32-S3
# 
# Usage: bash flash_all.sh [PORT]
#        Default port: /dev/ttyACM0
#
# Prerequisites:
#   1. Run on Windows PowerShell (Admin): usbipd attach --wsl --busid <BUSID> --auto-attach
#   2. Put board in download mode: hold BOOT, press RST, release BOOT
#   3. Wait ~3 seconds for USB to reconnect in WSL

set -e

PORT="${1:-/dev/ttyACM0}"
FIRMWARE_DIR="$(dirname "$0")"

source ~/esp/esp-idf-v5.3.2/export.sh 2>/dev/null

echo "==================================="
echo " TRM Inference — Full Flash"
echo "==================================="
echo ""
echo "Port:     $PORT"
echo "Firmware: $FIRMWARE_DIR/build/trm_inference.bin"
echo "Model:    $FIRMWARE_DIR/build/model_spiffs.bin"
echo ""

# Flash firmware (bootloader + partition table + app)
echo ">>> Flashing firmware..."
python -m esptool --chip esp32s3 -p "$PORT" -b 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0 "$FIRMWARE_DIR/build/bootloader/bootloader.bin" \
    0x8000 "$FIRMWARE_DIR/build/partition_table/partition-table.bin" \
    0x10000 "$FIRMWARE_DIR/build/trm_inference.bin" \
    0x310000 "$FIRMWARE_DIR/build/model_spiffs.bin"

echo ""
echo ">>> Flash complete! All partitions written."
echo ">>> Press RST on the board to start the firmware."
echo ">>> Then open monitor: idf.py -p $PORT monitor"

# SSD-MobileNetV2 for GAP8 (AI-Deck) — Full Pipeline

This repository contains a full training → quantization → deployment pipeline for running SSD-MobileNetV2 on the Bitcraze AI-Deck (GAP8) using:
- PyTorch Training
- ONNX Export
- NEMO Quantization
- DORY Code Generation
- GAP8 Firmware Deployment

## 🚀 Features

- SSD-MobileNetV2 with adjustable width multiplier
- Fully GAP8-compatible architecture
- Int8 quantization using NEMO
- DORY auto-tiling for fast GAP8 execution
- End-to-end bash script for reproducibility

## 🔁 Full Export & Cleanup Pipeline (Train → NEMO → DORY)

The full path from PyTorch to GAP8 is:

1. `train.py`  
   - Train SSD-MobileNetV2 (e.g. with `width_mult=0.1`, `160×160` input).

2. `export_nemo_quant.py`  
   - Uses NEMO (pulp-platform) to create a **quantized ONNX**:
   - Default: `export/ssd_mbv2_quant.onnx`.

3. `onnx-simplifier` (`onnxsim`)  
   - Simplifies the fused graph:
   - Input: `ssd_mbv2_fused.onnx`  
   - Output: `ssd_mbv2_simplified.onnx`.

4. `network_generate.py` (DORY, in the DORY repo)  
   - Consumes `ssd_mbv2_dory.onnx` and emits GAP8 code for AI-Deck.


## How to run

1. install python 3.8 and linux env (wsl or macOS)

2. run coco.sh

3. run train.py for 30 epochs

4. run run_all.sh
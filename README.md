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
   - Train SSD-MobileNetV2 (e.g. with `width_mult=0.25`, `320×320` input).

2. `export_float_raw.py`  
   - Builds an `SSDMobileNetV2Raw` wrapper and saves a clean checkpoint:
   - Default: `training/person_ssd_pytorch/ssd_mbv2_raw.pth`.

3. `export_nemo_quant.py`  
   - Uses NEMO (pulp-platform) to create a **quantized ONNX**:
   - Default: `export/ssd_mbv2_quant.onnx`.

4. `export/fuse_conv_bn.py`  
   - Fuses Conv+BatchNorm pairs inside the ONNX graph:
   - Input: `ssd_mbv2_quant.onnx`  
   - Output: `ssd_mbv2_fused.onnx`.

5. `onnx-simplifier` (`onnxsim`)  
   - Simplifies the fused graph:
   - Input: `ssd_mbv2_fused.onnx`  
   - Output: `ssd_mbv2_simplified.onnx`.

6. Strip unsupported / ugly ops **after** simplification:
   - `export/strip_transpose.py` → removes `Transpose` chains.  
   - `export/strip_min.py` → removes `Min` with constant clamp.  
   - `export/strip_fake_quant.py` → collapses fake quantization patterns.  
   - Final output: `export/ssd_mbv2_dory.onnx`.

7. `network_generate.py` (DORY, in the DORY repo)  
   - Consumes `ssd_mbv2_dory.onnx` and emits GAP8 code for AI-Deck.

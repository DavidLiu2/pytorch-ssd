#!/bin/bash
set -e

########################################
# PATHS / NAMES (edit if you change them)
########################################

# Checkpoint produced by export_float_raw.py (wrapper SSDMobileNetV2Raw weights)
RAW_CKPT="training/person_ssd_pytorch/ssd_mbv2_raw.pth"

# ONNX files through the pipeline
QUANT_ONNX="export/ssd_mbv2_quant.onnx"
FUSED_ONNX="export/ssd_mbv2_fused.onnx"
SIMPLIFIED_ONNX="export/ssd_mbv2_simplified.onnx"
NO_TRANSPOSE_ONNX="export/ssd_mbv2_notranspose.onnx"
NO_MIN_ONNX="export/ssd_mbv2_nomin.onnx"
FINAL_DORY_ONNX="export/ssd_mbv2_dory.onnx"

# DORY paths (adjust to your actual layout)
DORY_ROOT="../dory"
DORY_CONFIG="../dory_examples/config_files/config_person_ssd.json"

########################################
# 1. TRAIN (PyTorch)
########################################
echo "=== [1/7] Training SSD-MobileNetV2 ==="
# Uncomment to run training (skip if you have a trained model)
# python3 train.py

# NOTE:
# - Make sure train.py uses width_mult=0.25 (or your chosen value).
# - After training, ensure export_float_raw.py points to the checkpoint you like
#   (edit its --ckpt default), OR pass --ckpt explicitly below.

########################################
# 2. SAVE RAW WRAPPER CHECKPOINT FOR NEMO
########################################
echo "=== [2/7] Building raw wrapper checkpoint for NEMO ==="
# Uses defaults: --ckpt (training epoch) and --out (ssd_mbv2_raw.pth)
python3 export_float_raw.py

########################################
# 3. NEMO QUANTIZATION → QUANT ONNX
########################################
echo "=== [3/7] Exporting NEMO-quantized ONNX ==="
# Important: make sure export_nemo_quant.py default --width-mult matches training
python3 export_nemo_quant.py \
  --ckpt "${RAW_CKPT}" \
  --out "${QUANT_ONNX}"

########################################
# 4. FUSE CONV + BN (ON QUANT ONNX)
########################################
echo "=== [4/7] Fusing Conv + BatchNorm in ONNX graph ==="
python3 export/fuse_conv_bn.py \
  "${QUANT_ONNX}" \
  "${FUSED_ONNX}"

########################################
# 5. ONNX SIMPLIFIER (onnx-sim) BEFORE STRIPPING
########################################
echo "=== [5/7] Simplifying ONNX with onnx-simplifier ==="
python3 -m onnxsim \
  "${FUSED_ONNX}" \
  "${SIMPLIFIED_ONNX}"

########################################
# 6. STRIP UNSUPPORTED / UGLY OPS (AFTER onnx-sim)
########################################

echo "=== [6a] Stripping Transpose nodes ==="
python3 export/strip_transpose.py \
  "${SIMPLIFIED_ONNX}" \
  "${NO_TRANSPOSE_ONNX}"

echo "=== [6b] Stripping Min nodes (clamp-like) ==="
python3 export/strip_min.py \
  "${NO_TRANSPOSE_ONNX}" \
  "${NO_MIN_ONNX}"

echo "=== [6c] Stripping fake-quantization chains ==="
python3 export/strip_fake_quant.py \
  "${NO_MIN_ONNX}" \
  "${FINAL_DORY_ONNX}"

echo "Final DORY-ready ONNX: ${FINAL_DORY_ONNX}"

########################################
# 7. DORY CODE GENERATION
########################################
echo "=== [7/7] Running DORY network_generate ==="
(
  cd "${DORY_ROOT}"
  python3 network_generate.py \
    NEMO \
    PULP.GAP8 \
    "${DORY_CONFIG}" \
    --verbose 3
)

echo "============================================="
echo " DONE: SSD-MBV2 quantized & cleaned for GAP8 "
echo "============================================="

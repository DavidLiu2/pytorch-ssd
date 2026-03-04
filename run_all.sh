#!/bin/bash
exec > log.txt 2>&1
set -euo pipefail

########################################
# PROJECT & ENV PATHS
########################################

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DORY_ENV_DIR="${PROJECT_ROOT}/../doryenv"
NEMO_ENV_DIR="${PROJECT_ROOT}/../nemoenv"

DORY_REQ="${PROJECT_ROOT}/requirements_doryenv.txt"
NEMO_REQ="${PROJECT_ROOT}/requirements_nemoenv.txt"

########################################
# QUANT/SIM CONFIG
########################################

CKPT="${CKPT:-training/person_ssd_pytorch/ssd_mbv2_epoch_030.pth}"
OUT_ONNX="${OUT_ONNX:-export/ssd_mbv2_nemo_id.onnx}"
SIM_ONNX="${SIM_ONNX:-export/ssd_mbv2_nemo_id_sim.onnx}"
STAGE="${STAGE:-id}"
STAGE_REPORT="${STAGE_REPORT:-export/ssd_mbv2_final_stage.txt}"
STRICT_STAGE="${STRICT_STAGE:-0}"
BITS="${BITS:-8}"
EPS_IN="${EPS_IN:-$(python3 -c "print(1/255)")}"
CALIB_DIR="${CALIB_DIR:-data/rep_images}"
CALIB_BATCHES="${CALIB_BATCHES:-128}"
MEAN="${MEAN:-0.5,0.5,0.5}"
STD="${STD:-0.5,0.5,0.5}"

########################################
# HELPERS
########################################

ensure_venv() {
  local env_dir="$1"
  local req_file="$2"

  if [ ! -d "$env_dir" ]; then
    echo "=== Creating virtualenv at ${env_dir} ==="
    python3 -m venv "$env_dir"
    activate_venv "$env_dir"
    pip install --upgrade pip
    if [ -f "$req_file" ]; then
      pip install -r "$req_file"
    else
      echo "WARNING: requirements file not found: $req_file"
    fi
    deactivate
  fi
}

activate_venv() {
  local env_dir="$1"
  if [ -f "$env_dir/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$env_dir/bin/activate"
  elif [ -f "$env_dir/Scripts/activate" ]; then
    # shellcheck disable=SC1090
    source "$env_dir/Scripts/activate"
  else
    echo "ERROR: could not find activate script in ${env_dir}"
    exit 1
  fi
}

ensure_module_or_sync_requirements() {
  local module_name="$1"
  local req_file="$2"
  local py_bin="$3"
  if ! "$py_bin" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "=== Module '${module_name}' missing in active env; syncing ${req_file} ==="
    if [ -f "$req_file" ]; then
      "$py_bin" -m pip install -r "$req_file"
    else
      echo "ERROR: requirements file not found: $req_file"
      exit 1
    fi
  fi
}

########################################
# ENSURE ENVS EXIST
########################################

ensure_venv "$DORY_ENV_DIR" "$DORY_REQ"
ensure_venv "$NEMO_ENV_DIR" "$NEMO_REQ"

########################################
# 1. TRAIN (whatever env you started in, usually doryenv)
########################################

echo "=== [1/3] Training SSD-MobileNetV2 ==="
# python3 train.py

########################################
# 2. NEMO QUANT EXPORT (in nemoenv)
########################################

echo "=== [2/3] Exporting NEMO-quantized ONNX (using nemoenv) ==="

ORIG_VENV="${VIRTUAL_ENV:-}"   # remember where we started

if [ "${VIRTUAL_ENV:-}" != "$NEMO_ENV_DIR" ]; then
  activate_venv "$NEMO_ENV_DIR"
  echo "activated nemoenv: $NEMO_ENV_DIR"
fi

NEMO_PY="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
if [ -z "${NEMO_PY}" ]; then
  NEMO_PY="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi

if [ -z "${NEMO_PY}" ]; then
  echo "ERROR: expected nemoenv python/pip at:"
  echo "  $NEMO_ENV_DIR/bin/python3 or $NEMO_ENV_DIR/Scripts/python.exe"
  echo "  $NEMO_ENV_DIR/bin/pip or $NEMO_ENV_DIR/Scripts/pip.exe"
  exit 1
fi

NEMO_PREFIX="$("$NEMO_PY" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [[ "${NEMO_PREFIX}" != *"nemoenv"* ]]; then
  echo "WARNING: active python is not nemoenv; continuing with current interpreter:"
  echo "  python: ${NEMO_PY}"
  echo "  prefix: ${NEMO_PREFIX}"
  echo "If you want strict nemoenv isolation, run this script from the same shell family that created nemoenv."
fi

echo "nemo python: $("$NEMO_PY" --version)"
echo "nemo pip: $("$NEMO_PY" -m pip --version)"

# Ensure critical packages exist in nemoenv.
ensure_module_or_sync_requirements "torch" "$NEMO_REQ" "$NEMO_PY"
ensure_module_or_sync_requirements "nemo" "$NEMO_REQ" "$NEMO_PY"
ensure_module_or_sync_requirements "onnxsim" "$NEMO_REQ" "$NEMO_PY"

NEMO_CMD=(
  export_nemo_quant.py
  --ckpt "${CKPT}"
  --out "${OUT_ONNX}"
  --stage "${STAGE}"
  --stage-report "${STAGE_REPORT}"
  --bits "${BITS}"
  --eps-in "${EPS_IN}"
  --calib-batches "${CALIB_BATCHES}"
  --strict-stage
)

if [ "${STRICT_STAGE}" = "1" ]; then
  NEMO_CMD+=(--strict-stage)
fi

if [ -n "${CALIB_DIR}" ]; then
  NEMO_CMD+=(--calib-dir "${CALIB_DIR}")
fi

# If training used mean/std normalization, set both MEAN and STD.
if [ -n "${MEAN}" ] || [ -n "${STD}" ]; then
  if [ -z "${MEAN}" ] || [ -z "${STD}" ]; then
    echo "ERROR: set both MEAN and STD, or neither."
    exit 1
  fi
  NEMO_CMD+=(--mean "${MEAN}" --std "${STD}")
fi

CUDA_VISIBLE_DEVICES="" "$NEMO_PY" "${NEMO_CMD[@]}"

########################################
# 3. ONNX SIMPLIFIER
########################################

FINAL_STAGE="unknown"
if [ -f "${STAGE_REPORT}" ]; then
  FINAL_STAGE="$(tr -d '\r\n' < "${STAGE_REPORT}")"
fi

echo "=== [3/3] Simplifying ONNX with onnx-simplifier ==="
if [ "${FINAL_STAGE^^}" = "FQ" ]; then
  echo "Skipping onnxsim because final stage is FQ (tutorial flow expects QD/ID before simplification)."
  cp "${OUT_ONNX}" "${SIM_ONNX}"
else
  if ! "$NEMO_PY" -m onnxsim "${OUT_ONNX}" "${SIM_ONNX}" --skip-optimization; then
    echo "WARNING: onnxsim failed on ${OUT_ONNX}; using unsimplified ONNX as fallback."
    cp "${OUT_ONNX}" "${SIM_ONNX}"
  fi
fi

# Leave nemoenv
if [ "${VIRTUAL_ENV:-}" = "$NEMO_ENV_DIR" ]; then
  deactivate
fi

# Restore previous env (typically ../doryenv)
if [ -n "$ORIG_VENV" ] && { [ -f "$ORIG_VENV/bin/activate" ] || [ -f "$ORIG_VENV/Scripts/activate" ]; }; then
  activate_venv "$ORIG_VENV"
  echo "activated original venv: $ORIG_VENV"
elif [ -d "$DORY_ENV_DIR" ]; then
  activate_venv "$DORY_ENV_DIR"
  echo "activated original venv: $DORY_ENV_DIR"
fi

echo "============================================="
echo " DONE: NEMO export + ONNX simplification only "
echo "============================================="
echo "Python used: ${NEMO_PY}"
echo "Final quant stage: requested=${STAGE^^}, actual=${FINAL_STAGE^^}"
echo "Exported ONNX: ${OUT_ONNX}"
echo "Simplified ONNX: ${SIM_ONNX}"

#!/bin/bash
set -euo pipefail

########################################
# PROJECT & ENV PATHS
########################################

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${PROJECT_ROOT}"

DORY_ENV_DIR="${PROJECT_ROOT}/../doryenv"
NEMO_ENV_DIR="${PROJECT_ROOT}/../nemoenv"

DORY_REQ="${PROJECT_ROOT}/requirements_doryenv.txt"
NEMO_REQ="${PROJECT_ROOT}/requirements_nemoenv.txt"

########################################
# QUANT/SIM CONFIG
########################################

MODEL_TYPE="${MODEL_TYPE:-hybrid_follow}"
if [ "${MODEL_TYPE}" = "hybrid_follow" ]; then
  DEFAULT_CKPT="training/hybrid_follow/hybrid_follow_best_x.pth"
  DEFAULT_OUT_ONNX="export/hybrid_follow/hybrid_follow_quant.onnx"
  DEFAULT_SIM_ONNX="export/hybrid_follow/hybrid_follow_quant_sim.onnx"
  DEFAULT_STAGE="id"
  DEFAULT_STAGE_REPORT="export/hybrid_follow/hybrid_follow_final_stage.txt"
  DEFAULT_STRICT_STAGE="1"
  DEFAULT_INPUT_HEIGHT="128"
  DEFAULT_INPUT_WIDTH="128"
  DEFAULT_INPUT_CHANNELS="1"
  DEFAULT_CALIB_DIR=""
  DEFAULT_DORY_CONFIG_GEN="export/hybrid_follow/config_hybrid_follow_runtime.json"
  DEFAULT_DORY_ONNX="export/hybrid_follow/hybrid_follow_dory.onnx"
  DEFAULT_DORY_NO_AFFINE_ONNX="export/hybrid_follow/hybrid_follow_noaffine.onnx"
  DEFAULT_DORY_NO_TRANSPOSE_ONNX="export/hybrid_follow/hybrid_follow_notranspose.onnx"
  DEFAULT_DORY_NO_MIN_ONNX="export/hybrid_follow/hybrid_follow_nomin.onnx"
  DEFAULT_DORY_WEIGHTS_TXT_DIR="export/hybrid_follow/weights_txt"
  DEFAULT_DORY_ARTIFACT_MANIFEST="export/hybrid_follow/nemo_dory_artifacts.json"
  DEFAULT_COMPAT_PY_REPORT="export/hybrid_follow/model_compat_python.json"
  DEFAULT_COMPAT_ONNX_REPORT="export/hybrid_follow/model_compat_onnx.json"
  DEFAULT_DORY_APP_DIR="${PROJECT_ROOT}/application"
  DEFAULT_RUN_DORY="1"
  DEFAULT_SYNC_TO_CRAZYFLIE="0"
  DEFAULT_RUN_STAGE_DRIFT="1"
  DEFAULT_REAPPLY_GAP8_RAW_RESIDUAL_PATCHES="1"
  DEFAULT_RAW_RESIDUAL_PATCH_REPORT="export/hybrid_follow/gap8_raw_residual_patch_report.json"
  DEFAULT_STAGE_DRIFT_IMAGE="training/hybrid_follow/eval_epoch_015/top_fn/01_p0.0114_000000132408.jpg"
  DEFAULT_STAGE_DRIFT_OUTPUT_DIR="export/hybrid_follow/stage_drift/run_all"
  DEFAULT_STAGE_DRIFT_NEMO_STAGE="auto"
else
  DEFAULT_CKPT="training/person_ssd_pytorch/ssd_mbv2_epoch_030.pth"
  DEFAULT_OUT_ONNX="export/ssd_mbv2_nemo_id.onnx"
  DEFAULT_SIM_ONNX="export/ssd_mbv2_nemo_id_sim.onnx"
  DEFAULT_STAGE="id"
  DEFAULT_STAGE_REPORT="export/ssd_mbv2_final_stage.txt"
  DEFAULT_STRICT_STAGE="0"
  DEFAULT_INPUT_HEIGHT="160"
  DEFAULT_INPUT_WIDTH="160"
  DEFAULT_INPUT_CHANNELS="1"
  DEFAULT_CALIB_DIR="data/rep_images"
  DEFAULT_DORY_CONFIG_GEN="export/config_person_ssd_runtime.json"
  DEFAULT_DORY_ONNX="export/ssd_mbv2_dory.onnx"
  DEFAULT_DORY_NO_AFFINE_ONNX="export/ssd_mbv2_noaffine.onnx"
  DEFAULT_DORY_NO_TRANSPOSE_ONNX="export/ssd_mbv2_notranspose.onnx"
  DEFAULT_DORY_NO_MIN_ONNX="export/ssd_mbv2_nomin.onnx"
  DEFAULT_DORY_WEIGHTS_TXT_DIR="export/weights_txt"
  DEFAULT_DORY_ARTIFACT_MANIFEST="export/nemo_dory_artifacts.json"
  DEFAULT_COMPAT_PY_REPORT="export/model_compat_python.json"
  DEFAULT_COMPAT_ONNX_REPORT="export/model_compat_onnx.json"
  DEFAULT_DORY_APP_DIR="${PROJECT_ROOT}/application"
  DEFAULT_RUN_DORY="1"
  DEFAULT_SYNC_TO_CRAZYFLIE="0"
  DEFAULT_RUN_STAGE_DRIFT="0"
  DEFAULT_REAPPLY_GAP8_RAW_RESIDUAL_PATCHES="0"
  DEFAULT_RAW_RESIDUAL_PATCH_REPORT="export/gap8_raw_residual_patch_report.json"
  DEFAULT_STAGE_DRIFT_IMAGE=""
  DEFAULT_STAGE_DRIFT_OUTPUT_DIR="export/stage_drift/run_all"
  DEFAULT_STAGE_DRIFT_NEMO_STAGE="skip"
fi

CKPT="${CKPT:-${DEFAULT_CKPT}}"
OUT_ONNX="${OUT_ONNX:-${DEFAULT_OUT_ONNX}}"
SIM_ONNX="${SIM_ONNX:-${DEFAULT_SIM_ONNX}}"
STAGE="${STAGE:-${DEFAULT_STAGE}}"
STAGE_REPORT="${STAGE_REPORT:-${DEFAULT_STAGE_REPORT}}"
STRICT_STAGE="${STRICT_STAGE:-${DEFAULT_STRICT_STAGE}}"
BITS="${BITS:-8}"
EPS_IN="${EPS_IN:-$(python3 -c "print(1/255)")}"
INPUT_HEIGHT="${INPUT_HEIGHT:-${DEFAULT_INPUT_HEIGHT}}"
INPUT_WIDTH="${INPUT_WIDTH:-${DEFAULT_INPUT_WIDTH}}"
INPUT_CHANNELS="${INPUT_CHANNELS:-${DEFAULT_INPUT_CHANNELS}}"
CALIB_DIR="${CALIB_DIR:-${DEFAULT_CALIB_DIR}}"
CALIB_BATCHES="${CALIB_BATCHES:-128}"
COMPAT_CALIB_BATCHES="${COMPAT_CALIB_BATCHES:-8}"
RUN_DORY="${RUN_DORY:-${DEFAULT_RUN_DORY}}"
RUN_COMPAT_CHECKS="${RUN_COMPAT_CHECKS:-1}"
SYNC_TO_CRAZYFLIE="${SYNC_TO_CRAZYFLIE:-${DEFAULT_SYNC_TO_CRAZYFLIE}}"
GENERATE_DORY_ARTIFACTS="${GENERATE_DORY_ARTIFACTS:-1}"
STRICT_DORY_ARTIFACTS="${STRICT_DORY_ARTIFACTS:-0}"
RUN_STAGE_DRIFT="${RUN_STAGE_DRIFT:-${DEFAULT_RUN_STAGE_DRIFT}}"
REAPPLY_GAP8_RAW_RESIDUAL_PATCHES="${REAPPLY_GAP8_RAW_RESIDUAL_PATCHES:-${DEFAULT_REAPPLY_GAP8_RAW_RESIDUAL_PATCHES}}"
RAW_RESIDUAL_PATCH_REPORT="${RAW_RESIDUAL_PATCH_REPORT:-${DEFAULT_RAW_RESIDUAL_PATCH_REPORT}}"
STAGE_DRIFT_IMAGE="${STAGE_DRIFT_IMAGE:-${DEFAULT_STAGE_DRIFT_IMAGE}}"
STAGE_DRIFT_OUTPUT_DIR="${STAGE_DRIFT_OUTPUT_DIR:-${DEFAULT_STAGE_DRIFT_OUTPUT_DIR}}"
STAGE_DRIFT_NEMO_STAGE="${STAGE_DRIFT_NEMO_STAGE:-${DEFAULT_STAGE_DRIFT_NEMO_STAGE}}"
STAGE_DRIFT_GOLDEN="${STAGE_DRIFT_GOLDEN:-}"
STAGE_DRIFT_GVSOC_JSON="${STAGE_DRIFT_GVSOC_JSON:-}"
STAGE_DRIFT_CALIB_BATCHES="${STAGE_DRIFT_CALIB_BATCHES:-${COMPAT_CALIB_BATCHES}}"
if [ "${MODEL_TYPE}" = "hybrid_follow" ]; then
  DEFAULT_MEAN=""
  DEFAULT_STD=""
elif [ "${INPUT_CHANNELS}" = "1" ]; then
  DEFAULT_MEAN="0.5"
  DEFAULT_STD="0.5"
else
  DEFAULT_MEAN="0.5,0.5,0.5"
  DEFAULT_STD="0.5,0.5,0.5"
fi
MEAN="${MEAN:-${DEFAULT_MEAN}}"
STD="${STD:-${DEFAULT_STD}}"
CALIB_SEED="${CALIB_SEED:-0}"

########################################
# DORY CONFIG
########################################

DORY_ROOT="${DORY_ROOT:-${PROJECT_ROOT}/../dory}"
DORY_CONFIG_TEMPLATE="${DORY_CONFIG_TEMPLATE:-${PROJECT_ROOT}/../dory_examples/config_files/config_person_ssd.json}"
DORY_CONFIG_GEN="${DORY_CONFIG_GEN:-${DEFAULT_DORY_CONFIG_GEN}}"
DORY_ONNX="${DORY_ONNX:-${DEFAULT_DORY_ONNX}}"
DORY_NO_AFFINE_ONNX="${DORY_NO_AFFINE_ONNX:-${DEFAULT_DORY_NO_AFFINE_ONNX}}"
DORY_NO_TRANSPOSE_ONNX="${DORY_NO_TRANSPOSE_ONNX:-${DEFAULT_DORY_NO_TRANSPOSE_ONNX}}"
DORY_NO_MIN_ONNX="${DORY_NO_MIN_ONNX:-${DEFAULT_DORY_NO_MIN_ONNX}}"
DORY_FRONTEND="${DORY_FRONTEND:-NEMO}"
DORY_TARGET="${DORY_TARGET:-PULP.GAP8}"
DORY_APP_DIR="${DORY_APP_DIR:-${DEFAULT_DORY_APP_DIR}}"
DORY_PREFIX="${DORY_PREFIX:-}"
DORY_WEIGHTS_TXT_DIR="${DORY_WEIGHTS_TXT_DIR:-${DEFAULT_DORY_WEIGHTS_TXT_DIR}}"
DORY_ARTIFACT_MANIFEST="${DORY_ARTIFACT_MANIFEST:-${DEFAULT_DORY_ARTIFACT_MANIFEST}}"
COMPAT_PY_REPORT="${COMPAT_PY_REPORT:-${DEFAULT_COMPAT_PY_REPORT}}"
COMPAT_ONNX_REPORT="${COMPAT_ONNX_REPORT:-${DEFAULT_COMPAT_ONNX_REPORT}}"
CRAZYFLIE_APP_DIR="${CRAZYFLIE_APP_DIR:-${PROJECT_ROOT}/../crazyflie_ssd}"

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

ensure_nemo_requirement_sync() {
  local req_file="$1"
  local py_bin="$2"

  if [ ! -f "$req_file" ]; then
    echo "ERROR: requirements file not found: $req_file"
    exit 1
  fi

  local req_line
  req_line="$(grep -E '^[[:space:]]*pytorch-nemo[[:space:]]*@' "$req_file" | head -n1 || true)"
  if [ -z "$req_line" ]; then
    return
  fi

  local req_url
  req_url="${req_line#*@ }"
  if ! "$py_bin" -c "import inspect; import nemo.transf.deploy as d; import sys; sys.exit(0 if 'if eps_in_new is None:' in inspect.getsource(d._set_eps_in_pact) else 1)" >/dev/null 2>&1; then
    echo "=== pytorch-nemo missing expected deploy fix; syncing ${req_file} ==="
    "$py_bin" -m pip install --upgrade --force-reinstall --no-deps "$req_url"
  fi
}

abspath_with_python() {
  local py_bin="$1"
  local path_value="$2"
  "$py_bin" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$path_value"
}

resolve_existing_path() {
  local path_value="$1"
  if [ -f "$path_value" ]; then
    printf '%s\n' "$path_value"
    return 0
  fi

  local rooted="${PROJECT_ROOT}/${path_value}"
  if [ -f "$rooted" ]; then
    printf '%s\n' "$rooted"
    return 0
  fi

  return 1
}

select_hybrid_follow_ckpt() {
  local found_path

  if found_path="$(resolve_existing_path "${CKPT}")"; then
    printf '%s\n' "$found_path"
    return 0
  fi

  # Export should default to the checkpoint with the best horizontal control
  # error. Visibility-only checkpoints can still be monitored manually via CKPT,
  # but they should not be selected automatically for deployment.
  local best_x_ckpt="${PROJECT_ROOT}/training/hybrid_follow/hybrid_follow_best_x.pth"
  local best_follow_score_ckpt="${PROJECT_ROOT}/training/hybrid_follow/hybrid_follow_best_follow_score.pth"
  local best_total_loss_ckpt="${PROJECT_ROOT}/training/hybrid_follow/hybrid_follow_best_total_loss.pth"
  if [ -f "${best_x_ckpt}" ]; then
    printf '%s\n' "$best_x_ckpt"
    return 0
  fi
  if [ -f "${best_follow_score_ckpt}" ]; then
    printf '%s\n' "$best_follow_score_ckpt"
    return 0
  fi
  if [ -f "${best_total_loss_ckpt}" ]; then
    printf '%s\n' "$best_total_loss_ckpt"
    return 0
  fi

  local latest_train_ckpt
  latest_train_ckpt="$(ls -1 "${PROJECT_ROOT}/training/hybrid_follow"/hybrid_follow_epoch_*.pth 2>/dev/null | tail -n1 || true)"
  if [ -n "${latest_train_ckpt}" ] && [ -f "${latest_train_ckpt}" ]; then
    printf '%s\n' "$latest_train_ckpt"
    return 0
  fi

  local tmp_ckpt="${PROJECT_ROOT}/export/tmp_hybrid_ckpt.pth"
  if [ -f "${tmp_ckpt}" ]; then
    printf '%s\n' "$tmp_ckpt"
    return 0
  fi

  return 1
}

bootstrap_hybrid_follow_ckpt() {
  local py_bin="$1"
  local bootstrap_rel="export/hybrid_follow/bootstrap_hybrid_follow_random.pth"
  local bootstrap_abs="${PROJECT_ROOT}/${bootstrap_rel}"

  mkdir -p "$(dirname "${bootstrap_abs}")"
  BOOTSTRAP_CKPT_PATH="${bootstrap_abs}" "$py_bin" - <<'PY'
from pathlib import Path
import os

import torch

from models.hybrid_follow_net import HybridFollowNet

out_path = Path(os.environ["BOOTSTRAP_CKPT_PATH"])
model = HybridFollowNet(input_channels=1, image_size=(128, 128))
torch.save(
    {
        "model_type": "hybrid_follow",
        "height": 128,
        "width": 128,
        "input_channels": 1,
        "state_dict": model.state_dict(),
    },
    out_path,
)
print(out_path)
PY
  printf '%s\n' "${bootstrap_abs}"
}

dir_has_image_files() {
  local dir_path="$1"
  [ -d "${dir_path}" ] || return 1
  find "${dir_path}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .
}

select_hybrid_follow_calib_dir() {
  local candidate

  for candidate in \
    "${PROJECT_ROOT}/data/coco/images/val2017" \
    "${PROJECT_ROOT}/data/coco/images/train2017" \
    "${PROJECT_ROOT}/data/rep_images"
  do
    if dir_has_image_files "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if [ -d "${PROJECT_ROOT}/training/hybrid_follow" ]; then
    for candidate in $(find "${PROJECT_ROOT}/training/hybrid_follow" -maxdepth 1 -type d -name 'eval_epoch_*' | sort -r); do
      if dir_has_image_files "${candidate}" && find "${candidate}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  fi

  return 1
}

sync_dir_clean() {
  local src_dir="$1"
  local dst_dir="$2"
  local label="$3"

  if [ ! -d "${src_dir}" ]; then
    echo "ERROR: missing ${label} source directory: ${src_dir}"
    exit 1
  fi

  mkdir -p "${dst_dir}"
  find "${dst_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  cp -a "${src_dir}/." "${dst_dir}/"
  echo "[run_all] Synced ${label}: ${src_dir} -> ${dst_dir}"
}

sync_file() {
  local src_file="$1"
  local dst_file="$2"
  local label="$3"

  if [ ! -f "${src_file}" ]; then
    echo "ERROR: missing ${label} source file: ${src_file}"
    exit 1
  fi

  mkdir -p "$(dirname "${dst_file}")"
  cp "${src_file}" "${dst_file}"
  echo "[run_all] Synced ${label}: ${src_file} -> ${dst_file}"
}

sync_crazyflie_bundle() {
  local generated_dir="$1"
  local crazyflie_root="$2"
  local crazyflie_generated_dir="${crazyflie_root}/generated"

  if [ ! -d "${crazyflie_root}" ]; then
    echo "WARNING: crazyflie app directory not found; skipping sync: ${crazyflie_root}"
    return 0
  fi

  if [ "${generated_dir}" != "${crazyflie_generated_dir}" ]; then
    sync_dir_clean "${generated_dir}" "${crazyflie_generated_dir}" "crazyflie generated bundle"
  fi

  sync_dir_clean "${crazyflie_generated_dir}/hex" "${crazyflie_root}/hex" "crazyflie readfs hex"
  sync_file "${crazyflie_generated_dir}/vars.mk" "${crazyflie_root}/vars.mk" "crazyflie vars.mk"
}

########################################
# ENSURE ENVS EXIST
########################################

ensure_venv "$DORY_ENV_DIR" "$DORY_REQ"
ensure_venv "$NEMO_ENV_DIR" "$NEMO_REQ"

########################################
# 1. TRAIN (whatever env you started in, usually doryenv)
########################################

echo "=== [1/4] Training (${MODEL_TYPE}) ==="
# python3 train.py

########################################
# 2. NEMO QUANT EXPORT (in nemoenv)
########################################

echo "=== [2/4] Exporting ONNX (using nemoenv) ==="

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
ensure_nemo_requirement_sync "$NEMO_REQ" "$NEMO_PY"
ensure_module_or_sync_requirements "onnxsim" "$NEMO_REQ" "$NEMO_PY"

if [ "${MODEL_TYPE}" = "hybrid_follow" ]; then
  if HYBRID_CKPT_PATH="$(select_hybrid_follow_ckpt)"; then
    CKPT="${HYBRID_CKPT_PATH}"
    echo "[run_all] Using hybrid_follow checkpoint: ${CKPT}"
  else
    echo "[run_all] No hybrid_follow checkpoint found; creating bootstrap random-init checkpoint for export smoke test."
    CKPT="$(bootstrap_hybrid_follow_ckpt "$NEMO_PY")"
    echo "[run_all] Bootstrap checkpoint created: ${CKPT}"
  fi

  if [ -z "${CALIB_DIR}" ]; then
    if HYBRID_CALIB_DIR="$(select_hybrid_follow_calib_dir)"; then
      CALIB_DIR="${HYBRID_CALIB_DIR}"
      echo "[run_all] Using hybrid_follow calibration dir: ${CALIB_DIR}"
    else
      echo "[run_all] No hybrid_follow calibration images found; export will fall back to random calibration."
    fi
  fi
fi

if [ "${RUN_COMPAT_CHECKS}" = "1" ]; then
  echo "=== [preflight] Checking model compatibility (PyTorch) ==="
  COMPAT_CMD=(
    "${PROJECT_ROOT}/export/check_model_compatibility.py"
    --mode python
    --model-type "${MODEL_TYPE}"
    --ckpt "${CKPT}"
    --height "${INPUT_HEIGHT}"
    --width "${INPUT_WIDTH}"
    --input-channels "${INPUT_CHANNELS}"
    --stage "${STAGE}"
    --bits "${BITS}"
    --eps-in "${EPS_IN}"
    --calib-batches "${CALIB_BATCHES}"
    --compat-calib-batches "${COMPAT_CALIB_BATCHES}"
    --calib-seed "${CALIB_SEED}"
    --report-json "${COMPAT_PY_REPORT}"
    --fail-on-errors
  )

  if [ -n "${CALIB_DIR}" ]; then
    COMPAT_CMD+=(--calib-dir "${CALIB_DIR}")
  fi

  if [ -n "${MEAN}" ] || [ -n "${STD}" ]; then
    if [ -z "${MEAN}" ] || [ -z "${STD}" ]; then
      echo "ERROR: set both MEAN and STD, or neither."
      exit 1
    fi
    COMPAT_CMD+=(--mean "${MEAN}" --std "${STD}")
  fi

  CUDA_VISIBLE_DEVICES="" "$NEMO_PY" "${COMPAT_CMD[@]}"
fi

NEMO_CMD=(
  export_nemo_quant.py
  --model-type "${MODEL_TYPE}"
  --ckpt "${CKPT}"
  --out "${OUT_ONNX}"
  --height "${INPUT_HEIGHT}"
  --width "${INPUT_WIDTH}"
  --input-channels "${INPUT_CHANNELS}"
  --stage "${STAGE}"
  --stage-report "${STAGE_REPORT}"
  --bits "${BITS}"
  --eps-in "${EPS_IN}"
  --calib-batches "${CALIB_BATCHES}"
  --calib-seed "${CALIB_SEED}"
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

echo "=== [3/4] Simplifying ONNX with onnx-simplifier ==="
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

if [ "${RUN_DORY}" != "1" ]; then
  echo "=== DORY/codegen skipped (RUN_DORY=${RUN_DORY}) ==="
  if [ "${MODEL_TYPE}" = "hybrid_follow" ]; then
    echo "[run_all] hybrid_follow uses quantized ID export with a single 3-output head for DORY."
    echo "[run_all] Re-run with RUN_DORY=1 to regenerate the local pytorch_ssd/application deployment artifacts."
  fi
  echo "[run_all] Final stage: ${FINAL_STAGE^^}"
  echo "[run_all] Exported ONNX: ${OUT_ONNX}"
  echo "[run_all] Simplified ONNX: ${SIM_ONNX}"
  exit 0
fi

########################################
# 4. DORY + network_generate (in doryenv)
########################################

echo "=== [4/4] Preparing DORY input and running network_generate ==="

if [ ! -f "${SIM_ONNX}" ]; then
  echo "ERROR: simplified ONNX not found: ${SIM_ONNX}"
  exit 1
fi

if [ "${FINAL_STAGE^^}" != "ID" ]; then
  echo "ERROR: DORY stage requires ID ONNX, but final exported stage is: ${FINAL_STAGE^^}"
  echo "Set STAGE=id and re-run (current requested stage: ${STAGE^^})."
  exit 1
fi

if [ ! -d "${DORY_ROOT}" ] || [ ! -f "${DORY_ROOT}/network_generate.py" ]; then
  echo "ERROR: DORY repo not found or missing network_generate.py at: ${DORY_ROOT}"
  exit 1
fi

if [ ! -f "${DORY_CONFIG_TEMPLATE}" ]; then
  echo "ERROR: DORY config template not found: ${DORY_CONFIG_TEMPLATE}"
  exit 1
fi

if [ "${VIRTUAL_ENV:-}" != "$DORY_ENV_DIR" ]; then
  activate_venv "$DORY_ENV_DIR"
  echo "activated doryenv: $DORY_ENV_DIR"
fi

DORY_PY="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
if [ -z "${DORY_PY}" ]; then
  DORY_PY="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi

if [ -z "${DORY_PY}" ]; then
  echo "ERROR: expected doryenv python/pip at:"
  echo "  $DORY_ENV_DIR/bin/python3 or $DORY_ENV_DIR/Scripts/python.exe"
  echo "  $DORY_ENV_DIR/bin/pip or $DORY_ENV_DIR/Scripts/pip.exe"
  exit 1
fi

echo "dory python: $("$DORY_PY" --version)"
echo "dory pip: $("$DORY_PY" -m pip --version)"

# Ensure DORY runtime deps exist in doryenv.
ensure_module_or_sync_requirements "onnx" "$DORY_REQ" "$DORY_PY"
ensure_module_or_sync_requirements "onnxruntime" "$DORY_REQ" "$DORY_PY"

if [ ! -f "${PROJECT_ROOT}/export/strip_affine_mul_add.py" ]; then
  echo "ERROR: missing ONNX cleanup script: ${PROJECT_ROOT}/export/strip_affine_mul_add.py"
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/export/strip_transpose.py" ]; then
  echo "ERROR: missing ONNX cleanup script: ${PROJECT_ROOT}/export/strip_transpose.py"
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/export/strip_min.py" ]; then
  echo "ERROR: missing ONNX cleanup script: ${PROJECT_ROOT}/export/strip_min.py"
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/export/strip_fake_quant.py" ]; then
  echo "ERROR: missing ONNX cleanup script: ${PROJECT_ROOT}/export/strip_fake_quant.py"
  exit 1
fi

mkdir -p "$(dirname "${DORY_ONNX}")"

echo "[run_all] Cleaning ID ONNX for DORY frontend compatibility..."
"$DORY_PY" "${PROJECT_ROOT}/export/strip_affine_mul_add.py" "${SIM_ONNX}" "${DORY_NO_AFFINE_ONNX}"
"$DORY_PY" "${PROJECT_ROOT}/export/strip_transpose.py" "${DORY_NO_AFFINE_ONNX}" "${DORY_NO_TRANSPOSE_ONNX}"
"$DORY_PY" "${PROJECT_ROOT}/export/strip_min.py" "${DORY_NO_TRANSPOSE_ONNX}" "${DORY_NO_MIN_ONNX}"
"$DORY_PY" "${PROJECT_ROOT}/export/strip_fake_quant.py" "${DORY_NO_MIN_ONNX}" "${DORY_ONNX}"

if [ "${RUN_COMPAT_CHECKS}" = "1" ]; then
  echo "[run_all] Checking model compatibility (ONNX) ..."
  "$DORY_PY" "${PROJECT_ROOT}/export/check_model_compatibility.py" \
    --mode onnx \
    --onnx "${OUT_ONNX}" \
    --dory-onnx "${DORY_ONNX}" \
    --report-json "${COMPAT_ONNX_REPORT}" \
    --fail-on-errors
fi

DORY_ONNX_ABS="$(abspath_with_python "$DORY_PY" "${DORY_ONNX}")"
DORY_CONFIG_TEMPLATE_ABS="$(abspath_with_python "$DORY_PY" "${DORY_CONFIG_TEMPLATE}")"
DORY_CONFIG_GEN_ABS="$(abspath_with_python "$DORY_PY" "${DORY_CONFIG_GEN}")"
DORY_APP_DIR_ABS="$(abspath_with_python "$DORY_PY" "${DORY_APP_DIR}")"
DORY_WEIGHTS_TXT_DIR_ABS="$(abspath_with_python "$DORY_PY" "${DORY_WEIGHTS_TXT_DIR}")"
DORY_ARTIFACT_MANIFEST_ABS="$(abspath_with_python "$DORY_PY" "${DORY_ARTIFACT_MANIFEST}")"
CRAZYFLIE_APP_DIR_ABS="$(abspath_with_python "$DORY_PY" "${CRAZYFLIE_APP_DIR}")"
RAW_RESIDUAL_PATCH_REPORT_ABS="$(abspath_with_python "$DORY_PY" "${RAW_RESIDUAL_PATCH_REPORT}")"

mkdir -p "$(dirname "${DORY_CONFIG_GEN}")"
mkdir -p "${DORY_APP_DIR}"

DORY_TEMPLATE_PATH="${DORY_CONFIG_TEMPLATE_ABS}" \
DORY_CONFIG_OUT="${DORY_CONFIG_GEN_ABS}" \
DORY_ONNX_PATH="${DORY_ONNX_ABS}" \
"$DORY_PY" - <<'PY'
import json
import os

template_path = os.environ["DORY_TEMPLATE_PATH"]
config_out = os.environ["DORY_CONFIG_OUT"]
onnx_path = os.environ["DORY_ONNX_PATH"]

with open(template_path, "r", encoding="utf-8") as f:
    config = json.load(f)

config["onnx_file"] = onnx_path

os.makedirs(os.path.dirname(config_out), exist_ok=True)
with open(config_out, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

print(f"[run_all] DORY config written: {config_out}")
print(f"[run_all] DORY ONNX set to: {onnx_path}")
PY

if [ ! -f "${PROJECT_ROOT}/export/generate_dory_io_artifacts.py" ]; then
  echo "ERROR: missing artifact generation script: ${PROJECT_ROOT}/export/generate_dory_io_artifacts.py"
  exit 1
fi

mkdir -p "${DORY_WEIGHTS_TXT_DIR}"
mkdir -p "$(dirname "${DORY_ARTIFACT_MANIFEST}")"

echo "[run_all] Generating DORY input/output, golden activations, and weight txt files..."
if [ "${GENERATE_DORY_ARTIFACTS}" = "1" ]; then
  if ! "$DORY_PY" "${PROJECT_ROOT}/export/generate_dory_io_artifacts.py" \
    --onnx "${DORY_ONNX_ABS}" \
    --config "${DORY_CONFIG_GEN_ABS}" \
    --frontend "${DORY_FRONTEND}" \
    --target "${DORY_TARGET}" \
    --prefix "${DORY_PREFIX}" \
    --fallback-height "${INPUT_HEIGHT}" \
    --fallback-width "${INPUT_WIDTH}" \
    --weights-dir "${DORY_WEIGHTS_TXT_DIR_ABS}" \
    --manifest "${DORY_ARTIFACT_MANIFEST_ABS}"; then
    if [ "${STRICT_DORY_ARTIFACTS}" = "1" ]; then
      echo "ERROR: DORY artifact generation failed and STRICT_DORY_ARTIFACTS=1."
      exit 1
    fi
    echo "[run_all] WARNING: DORY artifact generation failed; continuing to network_generate."
  fi
else
  echo "[run_all] Skipping DORY artifact generation (GENERATE_DORY_ARTIFACTS=${GENERATE_DORY_ARTIFACTS})."
fi

DORY_CMD=(
  network_generate.py
  "${DORY_FRONTEND}"
  "${DORY_TARGET}"
  "${DORY_CONFIG_GEN_ABS}"
  --app_dir "${DORY_APP_DIR_ABS}"
)

if [ -n "${DORY_PREFIX}" ]; then
  DORY_CMD+=(--prefix "${DORY_PREFIX}")
fi

(
  cd "${DORY_ROOT}" || exit 1
  "$DORY_PY" -u "${DORY_CMD[@]}"
)

DORY_NETWORK_C="${DORY_APP_DIR_ABS}/src/network.c"
DORY_NETWORK_H="${DORY_APP_DIR_ABS}/inc/network.h"
if [ ! -f "${DORY_NETWORK_C}" ] || [ ! -f "${DORY_NETWORK_H}" ]; then
  echo "ERROR: DORY did not generate expected network files:"
  echo "  missing: ${DORY_NETWORK_C} and/or ${DORY_NETWORK_H}"
  echo "Hint: lower \"code reserved space\" in ${DORY_CONFIG_TEMPLATE} (100000 works for current SSD)."
  exit 1
fi

if [ "${REAPPLY_GAP8_RAW_RESIDUAL_PATCHES}" = "1" ]; then
  RAW_PATCH_TOOL="${PROJECT_ROOT}/tools/reapply_gap8_raw_residual_patches.py"
  if [ ! -f "${RAW_PATCH_TOOL}" ]; then
    echo "ERROR: missing raw-residual patch tool: ${RAW_PATCH_TOOL}"
    exit 1
  fi
  echo "[run_all] Reapplying hybrid_follow GAP8 raw-residual patch set ..."
  "$DORY_PY" "${RAW_PATCH_TOOL}" \
    --application-dir "${DORY_APP_DIR_ABS}" \
    --json-out "${RAW_RESIDUAL_PATCH_REPORT_ABS}"
fi

if [ "${SYNC_TO_CRAZYFLIE}" = "1" ]; then
  sync_crazyflie_bundle "${DORY_APP_DIR_ABS}" "${CRAZYFLIE_APP_DIR_ABS}"
fi

STAGE_DRIFT_SUMMARY=""
if [ "${RUN_STAGE_DRIFT}" = "1" ] && [ "${MODEL_TYPE}" = "hybrid_follow" ]; then
  echo "[run_all] Running stage-drift comparison ..."
  STAGE_DRIFT_CMD=(
    "${NEMO_PY}"
    "${PROJECT_ROOT}/export/compare_hybrid_follow_stages.py"
    --image "${STAGE_DRIFT_IMAGE}"
    --ckpt "${CKPT}"
    --onnx "${DORY_ONNX}"
    --output-dir "${STAGE_DRIFT_OUTPUT_DIR}"
    --overwrite
    --nemo-stage "${STAGE_DRIFT_NEMO_STAGE}"
    --nemo-bits "${BITS}"
    --nemo-eps-in "${EPS_IN}"
    --nemo-calib-batches "${STAGE_DRIFT_CALIB_BATCHES}"
  )
  if [ -n "${CALIB_DIR}" ]; then
    STAGE_DRIFT_CMD+=(--nemo-calib-dir "${CALIB_DIR}")
  fi
  if [ -n "${STAGE_DRIFT_GOLDEN}" ]; then
    STAGE_DRIFT_CMD+=(--golden "${STAGE_DRIFT_GOLDEN}")
  fi
  if [ -n "${STAGE_DRIFT_GVSOC_JSON}" ]; then
    STAGE_DRIFT_CMD+=(--gvsoc-json "${STAGE_DRIFT_GVSOC_JSON}")
  fi
  if ! "${STAGE_DRIFT_CMD[@]}"; then
    echo "[run_all] WARNING: stage-drift comparison failed; export artifacts are still available."
  else
    STAGE_DRIFT_SUMMARY="${STAGE_DRIFT_OUTPUT_DIR}/summary.md"
  fi
fi

echo "======================================================="
echo " DONE: NEMO export + ONNX simplification + DORY export "
echo "======================================================="
echo "Python used: ${NEMO_PY}"
echo "Final quant stage: requested=${STAGE^^}, actual=${FINAL_STAGE^^}"
echo "Exported ONNX: ${OUT_ONNX}"
echo "Simplified ONNX: ${SIM_ONNX}"
echo "DORY ONNX: ${DORY_ONNX}"
echo "DORY config: ${DORY_CONFIG_GEN_ABS}"
echo "DORY application dir: ${DORY_APP_DIR_ABS}"
echo "DORY weight txt dir: ${DORY_WEIGHTS_TXT_DIR_ABS}"
echo "DORY artifact manifest: ${DORY_ARTIFACT_MANIFEST_ABS}"
if [ "${REAPPLY_GAP8_RAW_RESIDUAL_PATCHES}" = "1" ]; then
  echo "Raw residual patch report: ${RAW_RESIDUAL_PATCH_REPORT_ABS}"
fi
if [ -n "${STAGE_DRIFT_SUMMARY}" ]; then
  echo "Stage drift summary: ${STAGE_DRIFT_SUMMARY}"
fi
if [ "${SYNC_TO_CRAZYFLIE}" = "1" ]; then
  echo "crazyflie sync dir: ${CRAZYFLIE_APP_DIR_ABS}"
fi

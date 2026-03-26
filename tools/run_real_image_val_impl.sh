#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_DIR}"

select_python() {
  if [ -n "${REAL_IMAGE_PYTHON:-}" ]; then
    printf '%s\n' "${REAL_IMAGE_PYTHON}"
    return 0
  fi
  if [ -x "${PROJECT_DIR}/../doryenv/bin/python3" ]; then
    printf '%s\n' "${PROJECT_DIR}/../doryenv/bin/python3"
    return 0
  fi
  printf '%s\n' "python3"
}

PYTHON_BIN="$(select_python)"

MODE="validate"
FORWARD_ARGS=()
while (($#)); do
  case "$1" in
    --overlay-only)
      MODE="overlay"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${MODE}" = "overlay" ]; then
  "${PYTHON_BIN}" export/visualize_hybrid_follow_batch.py "${FORWARD_ARGS[@]}"
  exit 0
fi

STAGE_DRIFT="${STAGE_DRIFT:-1}"
STAGE_DRIFT_ARGS=()
if [ "${STAGE_DRIFT}" = "1" ]; then
  STAGE_DRIFT_ARGS+=(--stage-drift)
  if [ -n "${STAGE_DRIFT_CKPT:-}" ]; then
    STAGE_DRIFT_ARGS+=(--stage-drift-ckpt "${STAGE_DRIFT_CKPT}")
  fi
  if [ -n "${STAGE_DRIFT_PYTHON:-}" ]; then
    STAGE_DRIFT_ARGS+=(--stage-drift-python "${STAGE_DRIFT_PYTHON}")
  fi
  if [ -n "${STAGE_DRIFT_NEMO_STAGE:-}" ]; then
    STAGE_DRIFT_ARGS+=(--stage-drift-nemo-stage "${STAGE_DRIFT_NEMO_STAGE}")
  fi
else
  STAGE_DRIFT_ARGS+=(--no-stage-drift)
fi

"${PYTHON_BIN}" export/validate_hybrid_follow_real_images.py "${STAGE_DRIFT_ARGS[@]}" "${FORWARD_ARGS[@]}"

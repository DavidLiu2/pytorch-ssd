#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if command -v wsl.exe >/dev/null 2>&1; then
      SCRIPT_DIR_WIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null || true)"
      if [[ -n "${SCRIPT_DIR_WIN:-}" ]]; then
        SCRIPT_PATH_WIN="${SCRIPT_DIR_WIN}\\$(basename "${BASH_SOURCE[0]}")"
        SCRIPT_PATH_WSL="$(wsl.exe wslpath -a "$SCRIPT_PATH_WIN" | tr -d '\r')"
        WSL_ENV_VARS=(
          CONTAINER_NAME
          AIDECK_IMAGE
          PLATFORM
          EXTRA_MAKE_ARGS
          RUN_AFTER_BUILD
          DETACH_RUN
          RUN_IO
          RUN_EXTRA_MAKE_ARGS
          RUN_LOG_NAME
          RUN_PID_FILE
          CONTAINER_TERM
          USE_VALIDATION_MAIN
          VERIFY_AFTER_RUN
          EXPECTED_TENSOR_LABEL
          EXPECTED_TENSOR_COUNT
          RUN_STAGE_DRIFT_DEBUG
          HOST_REPO_ROOT
          HOST_APP_DIR
          HOST_INPUT_HEX
          HOST_VALIDATION_MAIN
          HOST_EXPECTED_OUTPUT
          HOST_RUN_LOG_COPY
          HOST_FINAL_TENSOR_JSON
          HOST_STAGE_DRIFT_IMAGE
          HOST_STAGE_DRIFT_CKPT
          HOST_STAGE_DRIFT_ONNX
          HOST_STAGE_DRIFT_OUTPUT_DIR
          COMPARE_SCRIPT
          STAGE_DRIFT_SCRIPT
          STAGE_DRIFT_PYTHON
          STAGE_DRIFT_NEMO_STAGE
          AUTO_REFRESH_APP
          MODEL_SENTINEL
          MODEL_MANIFEST
          FALLBACK_APP_DIR
          REFRESH_SCRIPT
        )
        wsl_env_prefix=""
        for var_name in "${WSL_ENV_VARS[@]}"; do
          if [[ -n "${!var_name+x}" ]]; then
            printf -v assignment "%s=%q " "$var_name" "${!var_name}"
            wsl_env_prefix+="$assignment"
          fi
        done
        wsl_args=""
        for arg in "$@"; do
          printf -v escaped_arg "%q " "$arg"
          wsl_args+="$escaped_arg"
        done
        exec wsl.exe bash -lc "${wsl_env_prefix}${SCRIPT_PATH_WSL} ${wsl_args}"
      fi
    fi
    ;;
esac

# AI-Deck Docker bring-up + GVSOC validation for the current generated app.
#
# Usage:
#   bash pytorch_ssd/run_aideck_val.sh
#
# Optional env vars:
#   CONTAINER_NAME=aideck
#   AIDECK_IMAGE=bitcraze/aideck
#   HOST_REPO_ROOT=/abs/path/to/DroneRS
#   HOST_APP_DIR=/abs/path/to/generated/app
#   HOST_INPUT_HEX=/abs/path/to/inputs.hex
#   HOST_VALIDATION_MAIN=/abs/path/to/validation_main.c
#   HOST_EXPECTED_OUTPUT=/abs/path/to/output.txt
#   HOST_RUN_LOG_COPY=/abs/path/to/output/run_gvsoc.log
#   COMPARE_SCRIPT=/abs/path/to/compare_gap8_final_tensor.py
#   PLATFORM=gvsoc            # gvsoc (default) or board
#   EXTRA_MAKE_ARGS="..."     # extra args appended to make clean all

CONTAINER_NAME="${CONTAINER_NAME:-aideck}"
AIDECK_IMAGE="${AIDECK_IMAGE:-bitcraze/aideck}"
PLATFORM="${PLATFORM:-gvsoc}"
EXTRA_MAKE_ARGS="${EXTRA_MAKE_ARGS:-}"
RUN_AFTER_BUILD="${RUN_AFTER_BUILD:-1}"
DETACH_RUN="${DETACH_RUN:-0}"
RUN_IO="${RUN_IO:-host}"
RUN_EXTRA_MAKE_ARGS="${RUN_EXTRA_MAKE_ARGS:-}"
RUN_LOG_NAME="${RUN_LOG_NAME:-run_${PLATFORM}.log}"
RUN_PID_FILE="${RUN_PID_FILE:-gvsoc_run.pid}"
CONTAINER_TERM="${CONTAINER_TERM:-dumb}"
USE_VALIDATION_MAIN="${USE_VALIDATION_MAIN:-1}"
VERIFY_AFTER_RUN="${VERIFY_AFTER_RUN:-1}"
EXPECTED_TENSOR_LABEL="${EXPECTED_TENSOR_LABEL:-final}"
EXPECTED_TENSOR_COUNT="${EXPECTED_TENSOR_COUNT:-3}"
RUN_STAGE_DRIFT_DEBUG="${RUN_STAGE_DRIFT_DEBUG:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

HOST_APP_DIR="${HOST_APP_DIR:-$HOST_REPO_ROOT/pytorch_ssd/application}"
HOST_INPUT_HEX="${HOST_INPUT_HEX:-}"
HOST_VALIDATION_MAIN="${HOST_VALIDATION_MAIN:-$HOST_REPO_ROOT/pytorch_ssd/aideck_val_main_hybrid.c}"
HOST_EXPECTED_OUTPUT="${HOST_EXPECTED_OUTPUT:-$HOST_REPO_ROOT/pytorch_ssd/export/hybrid_follow/output.txt}"
HOST_RUN_LOG_COPY="${HOST_RUN_LOG_COPY:-}"
HOST_FINAL_TENSOR_JSON="${HOST_FINAL_TENSOR_JSON:-}"
HOST_STAGE_DRIFT_IMAGE="${HOST_STAGE_DRIFT_IMAGE:-}"
HOST_STAGE_DRIFT_CKPT="${HOST_STAGE_DRIFT_CKPT:-$HOST_REPO_ROOT/pytorch_ssd/training/hybrid_follow/hybrid_follow_best_follow_score.pth}"
HOST_STAGE_DRIFT_ONNX="${HOST_STAGE_DRIFT_ONNX:-$HOST_REPO_ROOT/pytorch_ssd/export/hybrid_follow/hybrid_follow_dory.onnx}"
HOST_STAGE_DRIFT_OUTPUT_DIR="${HOST_STAGE_DRIFT_OUTPUT_DIR:-}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-$HOST_REPO_ROOT/pytorch_ssd/export/compare_gap8_final_tensor.py}"
STAGE_DRIFT_SCRIPT="${STAGE_DRIFT_SCRIPT:-$HOST_REPO_ROOT/pytorch_ssd/export/compare_hybrid_follow_stages.py}"
STAGE_DRIFT_PYTHON="${STAGE_DRIFT_PYTHON:-}"
STAGE_DRIFT_NEMO_STAGE="${STAGE_DRIFT_NEMO_STAGE:-auto}"
AUTO_REFRESH_APP="${AUTO_REFRESH_APP:-1}"
MODEL_SENTINEL="${MODEL_SENTINEL:-$HOST_REPO_ROOT/pytorch_ssd/export/hybrid_follow/hybrid_follow_nomin.onnx}"
MODEL_MANIFEST="${MODEL_MANIFEST:-$HOST_REPO_ROOT/pytorch_ssd/export/hybrid_follow/nemo_dory_artifacts.json}"
FALLBACK_APP_DIR="${FALLBACK_APP_DIR:-}"
REFRESH_SCRIPT="${REFRESH_SCRIPT:-$HOST_REPO_ROOT/pytorch_ssd/run_all.sh}"
HOST_BUILD_DIR="$HOST_REPO_ROOT/aideck-gap8-examples/examples/other/dory_examples/application/BUILD/GAP8_V2/GCC_RISCV_PULPOS"
CONTAINER_APP_DIR="/module/aideck-gap8-examples/examples/other/dory_examples/application"
CONTAINER_APP_PARENT="/module/aideck-gap8-examples/examples/other/dory_examples"
CONTAINER_BUILD_DIR="$CONTAINER_APP_DIR/BUILD/GAP8_V2/GCC_RISCV_PULPOS"
HOST_RUN_LOG="$HOST_BUILD_DIR/$RUN_LOG_NAME"

container_path_from_host() {
  local host_path="$1"
  if [[ "$host_path" != "$HOST_REPO_ROOT" && "$host_path" != "$HOST_REPO_ROOT/"* ]]; then
    echo "ERROR: path must live under repo root so the Docker bind mount can see it: $host_path"
    exit 1
  fi
  local rel_path="${host_path#"$HOST_REPO_ROOT"/}"
  printf '/module/%s\n' "$rel_path"
}

app_dir_has_generated_model() {
  local app_dir="$1"
  [[ -d "$app_dir" ]] &&
  [[ -f "$app_dir/src/network.c" ]] &&
  [[ -f "$app_dir/inc/network.h" ]] &&
  [[ -f "$app_dir/hex/inputs.hex" ]] &&
  [[ -f "$app_dir/vars.mk" ]]
}

model_export_is_available() {
  [[ -f "$MODEL_SENTINEL" ]] && [[ -f "$MODEL_MANIFEST" ]]
}

select_stage_drift_python() {
  if [[ -n "$STAGE_DRIFT_PYTHON" ]]; then
    printf '%s\n' "$STAGE_DRIFT_PYTHON"
    return 0
  fi
  if [[ -x "$HOST_REPO_ROOT/nemoenv/bin/python3" ]]; then
    printf '%s\n' "$HOST_REPO_ROOT/nemoenv/bin/python3"
    return 0
  fi
  if [[ -f "$HOST_REPO_ROOT/nemoenv/Scripts/python.exe" ]]; then
    printf '%s\n' "$HOST_REPO_ROOT/nemoenv/Scripts/python.exe"
    return 0
  fi
  printf '%s\n' "python3"
}

write_final_tensor_json() {
  local log_path="$1"
  local json_path="$2"
  local tensor_label="$3"
  local tensor_count="$4"

  mkdir -p "$(dirname "$json_path")"
  python3 - "$log_path" "$json_path" "$tensor_label" "$tensor_count" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
label = sys.argv[3]
expected_count = int(sys.argv[4])

begin_re = re.compile(r"^FINAL_TENSOR_I32_BEGIN\s+(\w+)\s+count=(\d+)$")
line_re = re.compile(r"^FINAL_TENSOR_I32\s+(\w+)(.*)$")
end_re = re.compile(r"^FINAL_TENSOR_I32_END\s+(\w+)$")

values = []
count = None
collecting = False
for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = raw_line.strip()
    begin_match = begin_re.match(line)
    if begin_match and begin_match.group(1) == label:
        count = int(begin_match.group(2))
        values = []
        collecting = True
        continue
    end_match = end_re.match(line)
    if end_match and end_match.group(1) == label:
        collecting = False
        continue
    line_match = line_re.match(line)
    if line_match and line_match.group(1) == label and collecting:
        payload = line_match.group(2).strip()
        if payload:
            values.extend(int(token) for token in payload.split())

if count is None:
    raise RuntimeError(f"No FINAL_TENSOR_I32 block found for label '{label}' in {log_path}")
if count != expected_count:
    raise RuntimeError(f"Expected count={expected_count} for '{label}', got {count}")
if len(values) != count:
    raise RuntimeError(f"Tensor '{label}' dump size mismatch: expected {count}, got {len(values)}")

json_path.write_text(
    json.dumps({"label": label, "count": count, "values": values}, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

app_dir_is_fresh() {
  local app_dir="$1"
  app_dir_has_generated_model "$app_dir" || return 1
  model_export_is_available || return 1
  [[ ! "$app_dir/src/network.c" -ot "$MODEL_SENTINEL" ]] || return 1
  [[ ! "$app_dir/src/network.c" -ot "$MODEL_MANIFEST" ]] || return 1
  return 0
}

copy_app_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "$(dirname "$dst_dir")"
  rm -rf "$dst_dir"
  cp -a "$src_dir" "$dst_dir"
}

refresh_generated_app() {
  if app_dir_is_fresh "$HOST_APP_DIR"; then
    echo "[model] Using fresh generated app at $HOST_APP_DIR"
    return 0
  fi

  if [[ -n "$FALLBACK_APP_DIR" && "$FALLBACK_APP_DIR" != "$HOST_APP_DIR" ]] && app_dir_is_fresh "$FALLBACK_APP_DIR"; then
    echo "[model] Copying fresh generated app from $FALLBACK_APP_DIR to $HOST_APP_DIR"
    copy_app_dir "$FALLBACK_APP_DIR" "$HOST_APP_DIR"
    return 0
  fi

  if [[ "$AUTO_REFRESH_APP" != "1" ]]; then
    echo "ERROR: fresh generated app not found and AUTO_REFRESH_APP=0"
    exit 1
  fi

  if [[ ! -f "$REFRESH_SCRIPT" ]]; then
    echo "ERROR: refresh script not found: $REFRESH_SCRIPT"
    exit 1
  fi

  echo "[model] Fresh generated app not found; rerunning $REFRESH_SCRIPT ..."
  (
    cd "$HOST_REPO_ROOT/pytorch_ssd"
    bash "$REFRESH_SCRIPT"
  )

  if ! app_dir_is_fresh "$HOST_APP_DIR"; then
    echo "ERROR: expected fresh generated app under $HOST_APP_DIR after rerunning run_all.sh"
    exit 1
  fi

  echo "[model] Refreshed generated app at $HOST_APP_DIR"
}

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH."
  exit 1
fi

refresh_generated_app

if [[ ! -d "$HOST_APP_DIR" ]]; then
  echo "ERROR: app directory not found: $HOST_APP_DIR"
  exit 1
fi

if [[ "$USE_VALIDATION_MAIN" == "1" && ! -f "$HOST_VALIDATION_MAIN" ]]; then
  echo "ERROR: validation main not found: $HOST_VALIDATION_MAIN"
  exit 1
fi

if [[ -n "$HOST_INPUT_HEX" && ! -f "$HOST_INPUT_HEX" ]]; then
  echo "ERROR: input hex override not found: $HOST_INPUT_HEX"
  exit 1
fi

if [[ "$VERIFY_AFTER_RUN" == "1" ]]; then
  if [[ ! -f "$HOST_EXPECTED_OUTPUT" ]]; then
    echo "ERROR: expected output file not found: $HOST_EXPECTED_OUTPUT"
    exit 1
  fi
  if [[ ! -f "$COMPARE_SCRIPT" ]]; then
    echo "ERROR: compare script not found: $COMPARE_SCRIPT"
    exit 1
  fi
fi

CONTAINER_APP_SRC_DIR="$(container_path_from_host "$HOST_APP_DIR")"
CONTAINER_INPUT_HEX=""
if [[ -n "$HOST_INPUT_HEX" ]]; then
  CONTAINER_INPUT_HEX="$(container_path_from_host "$HOST_INPUT_HEX")"
fi
CONTAINER_VALIDATION_MAIN=""
if [[ "$USE_VALIDATION_MAIN" == "1" ]]; then
  CONTAINER_VALIDATION_MAIN="$(container_path_from_host "$HOST_VALIDATION_MAIN")"
fi

echo "[1/5] Ensure container '$CONTAINER_NAME' exists and is running..."
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker start "$CONTAINER_NAME" >/dev/null
  fi
else
  docker run -d --name "$CONTAINER_NAME" -v "$HOST_REPO_ROOT:/module" "$AIDECK_IMAGE" tail -f /dev/null >/dev/null
fi

echo "[2/5] Sync app into container path..."
docker exec "$CONTAINER_NAME" bash -lc "
  set -e
  mkdir -p '$CONTAINER_APP_PARENT'
  rm -rf '$CONTAINER_APP_DIR'
  cp -a '$CONTAINER_APP_SRC_DIR' '$CONTAINER_APP_DIR'
  if [[ '$USE_VALIDATION_MAIN' == '1' ]]; then
    cp '$CONTAINER_VALIDATION_MAIN' '$CONTAINER_APP_DIR/src/main.c'
  fi
  if [[ -n '$CONTAINER_INPUT_HEX' ]]; then
    cp '$CONTAINER_INPUT_HEX' '$CONTAINER_APP_DIR/hex/inputs.hex'
  fi
  if grep -q 'unsigned int args\\[4\\];' '$CONTAINER_APP_DIR/src/network.c'; then
    perl -0pi -e 's/unsigned int args\\[4\\];/unsigned int args[5];/' '$CONTAINER_APP_DIR/src/network.c'
  fi
"

echo "[3/5] Apply out_mult alias patch where needed..."
docker exec "$CONTAINER_NAME" bash -lc "
  set -e
  cd '$CONTAINER_APP_DIR'
  for f in src/Convolution*.c; do
    if grep -q 'out_mult_in' \"\$f\" && grep -q 'out_shift_in' \"\$f\" && grep -q 'uint16_t out_shift = out_shift_in;' \"\$f\" && ! grep -q 'uint16_t out_mult = out_mult_in;' \"\$f\"; then
      perl -0777 -i -pe 's/\\n(\\s*)uint16_t out_shift = out_shift_in;/\\n\$1uint16_t out_mult = out_mult_in;\\n\$1uint16_t out_shift = out_shift_in;/g' \"\$f\"
    fi
  done
"

echo "[4/5] Show toolchain environment..."
docker exec "$CONTAINER_NAME" bash -lc "
  set -e
  export TERM='$CONTAINER_TERM'
  cd /gap_sdk
  source configs/ai_deck.sh
  which riscv32-unknown-elf-gcc
  which gapy
  env | grep -E 'GAP|PMSIS|RISCV|GNU' || true
"

echo "[5/5] Build application (platform=$PLATFORM)..."
docker exec "$CONTAINER_NAME" bash -lc "
  set -eo pipefail
  export TERM='$CONTAINER_TERM'
  cd /gap_sdk
  source configs/ai_deck.sh
  cd '$CONTAINER_APP_DIR'
  make help || true
  make clean all platform='$PLATFORM' $EXTRA_MAKE_ARGS 2>&1 | tee build_${PLATFORM}.log
"

if [[ "$DETACH_RUN" == "1" ]]; then
  RUN_AFTER_BUILD=1
  VERIFY_AFTER_RUN=0
fi

if [[ "$RUN_AFTER_BUILD" == "1" ]]; then
  run_cmd=(make run "platform=$PLATFORM" "io=$RUN_IO")
  if [[ -n "$RUN_EXTRA_MAKE_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=($RUN_EXTRA_MAKE_ARGS)
    run_cmd+=("${extra_args[@]}")
  fi
  printf -v run_cmd_str '%q ' "${run_cmd[@]}"

  echo "[run] Starting application (platform=$PLATFORM, io=$RUN_IO)..."
  docker exec "$CONTAINER_NAME" bash -lc "
    set -e
    mkdir -p '$CONTAINER_BUILD_DIR'
    rm -f \
      '$CONTAINER_BUILD_DIR/network_progress.txt' \
      '$CONTAINER_BUILD_DIR/bbox.bin' \
      '$CONTAINER_BUILD_DIR/cls.bin' \
      '$CONTAINER_BUILD_DIR/$RUN_LOG_NAME' \
      '$CONTAINER_BUILD_DIR/$RUN_PID_FILE'
  "
  if [[ "$DETACH_RUN" == "1" ]]; then
    docker exec "$CONTAINER_NAME" bash -lc "
      set -e
      export TERM='$CONTAINER_TERM'
      cd /gap_sdk
      source configs/ai_deck.sh
      cd '$CONTAINER_APP_DIR'
      mkdir -p '$CONTAINER_BUILD_DIR'
      nohup bash -lc \"$run_cmd_str\" > '$CONTAINER_BUILD_DIR/$RUN_LOG_NAME' 2>&1 < /dev/null &
      echo \$! > '$CONTAINER_BUILD_DIR/$RUN_PID_FILE'
    "
    echo "[run] Detached."
    echo "Host build dir: $HOST_BUILD_DIR"
    echo "Host progress file: $HOST_BUILD_DIR/network_progress.txt"
    echo "Host bbox dump: $HOST_BUILD_DIR/bbox.bin"
    echo "Host cls dump: $HOST_BUILD_DIR/cls.bin"
    echo "Host run log: $HOST_BUILD_DIR/$RUN_LOG_NAME"
    echo "Host pid file: $HOST_BUILD_DIR/$RUN_PID_FILE"
  else
    docker exec "$CONTAINER_NAME" bash -lc "
      set -e
      export TERM='$CONTAINER_TERM'
      cd /gap_sdk
      source configs/ai_deck.sh
      cd '$CONTAINER_APP_DIR'
      mkdir -p '$CONTAINER_BUILD_DIR'
      bash -lc \"$run_cmd_str\" 2>&1 | tee '$CONTAINER_BUILD_DIR/$RUN_LOG_NAME'
    "
    if [[ -n "$HOST_RUN_LOG_COPY" ]]; then
      mkdir -p "$(dirname "$HOST_RUN_LOG_COPY")"
      cp "$HOST_RUN_LOG" "$HOST_RUN_LOG_COPY"
    fi
    FINAL_TENSOR_JSON_PATH="${HOST_FINAL_TENSOR_JSON:-}"
    if [[ -z "$FINAL_TENSOR_JSON_PATH" && "$RUN_STAGE_DRIFT_DEBUG" == "1" && -n "$HOST_STAGE_DRIFT_OUTPUT_DIR" ]]; then
      FINAL_TENSOR_JSON_PATH="$HOST_STAGE_DRIFT_OUTPUT_DIR/gvsoc_final_tensor.json"
    fi
    if [[ -n "$FINAL_TENSOR_JSON_PATH" ]]; then
      write_final_tensor_json "$HOST_RUN_LOG" "$FINAL_TENSOR_JSON_PATH" "$EXPECTED_TENSOR_LABEL" "$EXPECTED_TENSOR_COUNT"
    fi
    if [[ "$VERIFY_AFTER_RUN" == "1" ]]; then
      echo "[verify] Comparing GVSOC final tensor against $HOST_EXPECTED_OUTPUT..."
      python3 "$COMPARE_SCRIPT" \
        --gvsoc-log "$HOST_RUN_LOG" \
        --expected-output "$HOST_EXPECTED_OUTPUT" \
        --label "$EXPECTED_TENSOR_LABEL" \
        --count "$EXPECTED_TENSOR_COUNT"
    fi
    if [[ "$RUN_STAGE_DRIFT_DEBUG" == "1" ]]; then
      if [[ -z "$HOST_STAGE_DRIFT_IMAGE" || -z "$HOST_STAGE_DRIFT_OUTPUT_DIR" ]]; then
        echo "[stage-drift] Skipping because HOST_STAGE_DRIFT_IMAGE or HOST_STAGE_DRIFT_OUTPUT_DIR was not set."
      elif [[ ! -f "$STAGE_DRIFT_SCRIPT" ]]; then
        echo "[stage-drift] WARNING: script not found: $STAGE_DRIFT_SCRIPT"
      else
        STAGE_DRIFT_PY_BIN="$(select_stage_drift_python)"
        STAGE_DRIFT_CMD=(
          "$STAGE_DRIFT_PY_BIN"
          "$STAGE_DRIFT_SCRIPT"
          --image "$HOST_STAGE_DRIFT_IMAGE"
          --ckpt "$HOST_STAGE_DRIFT_CKPT"
          --onnx "$HOST_STAGE_DRIFT_ONNX"
          --golden "$HOST_EXPECTED_OUTPUT"
          --output-dir "$HOST_STAGE_DRIFT_OUTPUT_DIR"
          --overwrite
          --nemo-stage "$STAGE_DRIFT_NEMO_STAGE"
        )
        if [[ -n "$FINAL_TENSOR_JSON_PATH" && -f "$FINAL_TENSOR_JSON_PATH" ]]; then
          STAGE_DRIFT_CMD+=(--gvsoc-json "$FINAL_TENSOR_JSON_PATH")
        fi
        if ! "${STAGE_DRIFT_CMD[@]}"; then
          echo "[stage-drift] WARNING: stage-drift comparison failed; continuing."
        fi
      fi
    fi
  fi
fi

echo
echo "Done."
echo "Host app source: $HOST_APP_DIR"
if [[ -n "$HOST_INPUT_HEX" ]]; then
  echo "Host input hex override: $HOST_INPUT_HEX"
fi
echo "Container app path: $CONTAINER_APP_DIR"
echo "Build log: $CONTAINER_APP_DIR/build_${PLATFORM}.log"
echo "Host build dir: $HOST_BUILD_DIR"
echo "Host run log: $HOST_RUN_LOG"
if [[ -n "$HOST_RUN_LOG_COPY" ]]; then
  echo "Copied run log: $HOST_RUN_LOG_COPY"
fi
if [[ "$VERIFY_AFTER_RUN" == "1" && "$DETACH_RUN" != "1" ]]; then
  echo "Expected output: $HOST_EXPECTED_OUTPUT"
fi
if [[ -n "$HOST_FINAL_TENSOR_JSON" ]]; then
  echo "Final tensor JSON: $HOST_FINAL_TENSOR_JSON"
fi
if [[ "$RUN_STAGE_DRIFT_DEBUG" == "1" && -n "$HOST_STAGE_DRIFT_OUTPUT_DIR" ]]; then
  echo "Stage drift output dir: $HOST_STAGE_DRIFT_OUTPUT_DIR"
fi

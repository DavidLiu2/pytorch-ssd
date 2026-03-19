#!/usr/bin/env bash
set -euo pipefail

# AI-Deck Docker bring-up + build from scratch.
#
# Usage:
#   bash pytorch_ssd/run_aideck_from_scratch.sh
#
# Optional env vars:
#   CONTAINER_NAME=aideck
#   AIDECK_IMAGE=bitcraze/aideck
#   HOST_REPO_ROOT=/abs/path/to/DroneRS
#   PLATFORM=gvsoc            # gvsoc (default) or board
#   EXTRA_MAKE_ARGS="..."     # extra args appended to make clean all

CONTAINER_NAME="${CONTAINER_NAME:-aideck}"
AIDECK_IMAGE="${AIDECK_IMAGE:-bitcraze/aideck}"
PLATFORM="${PLATFORM:-gvsoc}"
EXTRA_MAKE_ARGS="${EXTRA_MAKE_ARGS:-}"
RUN_AFTER_BUILD="${RUN_AFTER_BUILD:-0}"
DETACH_RUN="${DETACH_RUN:-0}"
RUN_IO="${RUN_IO:-host}"
RUN_EXTRA_MAKE_ARGS="${RUN_EXTRA_MAKE_ARGS:-}"
RUN_LOG_NAME="${RUN_LOG_NAME:-run_${PLATFORM}.log}"
RUN_PID_FILE="${RUN_PID_FILE:-gvsoc_run.pid}"
CONTAINER_TERM="${CONTAINER_TERM:-dumb}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

HOST_APP_DIR="$HOST_REPO_ROOT/pytorch_ssd/application"
HOST_BUILD_DIR="$HOST_REPO_ROOT/aideck-gap8-examples/examples/other/dory_examples/application/BUILD/GAP8_V2/GCC_RISCV_PULPOS"
CONTAINER_APP_DIR="/module/aideck-gap8-examples/examples/other/dory_examples/application"
CONTAINER_APP_PARENT="/module/aideck-gap8-examples/examples/other/dory_examples"
CONTAINER_BUILD_DIR="$CONTAINER_APP_DIR/BUILD/GAP8_V2/GCC_RISCV_PULPOS"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH."
  exit 1
fi

if [[ ! -d "$HOST_APP_DIR" ]]; then
  echo "ERROR: app directory not found: $HOST_APP_DIR"
  exit 1
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
docker exec "$CONTAINER_NAME" bash -lc "set -e; mkdir -p '$CONTAINER_APP_PARENT'; rm -rf '$CONTAINER_APP_DIR'; cp -a '/module/pytorch_ssd/application' '$CONTAINER_APP_PARENT/'"

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
  fi
fi

echo
echo "Done."
echo "Container app path: $CONTAINER_APP_DIR"
echo "Build log: $CONTAINER_APP_DIR/build_${PLATFORM}.log"
echo "Host build dir: $HOST_BUILD_DIR"
echo "Progress file (expected): $HOST_BUILD_DIR/network_progress.txt"

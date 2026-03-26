#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOOLS_DIR="${SCRIPT_DIR}/tools"
AIDECK_IMPL="${TOOLS_DIR}/run_aideck_val_impl.sh"
REAL_IMPL="${TOOLS_DIR}/run_real_image_val_impl.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/export/evaluate_hybrid_follow_application.py"

show_usage() {
  cat <<'EOF'
Usage:
  ./run_hybrid_follow_val.sh [full] [evaluate args...]
  ./run_hybrid_follow_val.sh aideck [run_aideck_val args...]
  ./run_hybrid_follow_val.sh real [run_real_image_val args...]
  ./run_hybrid_follow_val.sh overlay [run_real_image_val args...]
  ./run_hybrid_follow_val.sh compare [evaluate args...]

Modes:
  full
    Canonical hybrid-follow validation flow. Runs:
      1. staged AI-Deck validation
      2. checkpoint-vs-application evaluation
    The evaluation step already runs the real-image validation internally.
    This is the default when no mode is provided.
  aideck
    Run only the staged single-sample AI-Deck Docker validation.
  real
    Run only the batch real-image validation.
  overlay
    Regenerate real-image overlays without rerunning GVSOC.
  compare
    Run only the checkpoint-vs-application evaluation report.

Notes:
  - `run_aideck_val.sh` and `run_real_image_val.sh` are now compatibility wrappers.
  - Set `COMPARE_PYTHON=/path/to/python` if compare/full should use a specific interpreter.
EOF
}

select_compare_python() {
  if [[ -n "${COMPARE_PYTHON:-}" ]]; then
    printf '%s\n' "${COMPARE_PYTHON}"
    return 0
  fi
  if [[ -n "${HYBRID_FOLLOW_COMPARE_PYTHON:-}" ]]; then
    printf '%s\n' "${HYBRID_FOLLOW_COMPARE_PYTHON}"
    return 0
  fi

  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      if [[ -f "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe"
        return 0
      fi
      ;;
    *)
      if [[ -x "${SCRIPT_DIR}/../nemoenv/bin/python3" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/../nemoenv/bin/python3"
        return 0
      fi
      if [[ -f "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe"
        return 0
      fi
      ;;
  esac

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi

  echo "ERROR: no Python interpreter found for compare mode." >&2
  echo "Set COMPARE_PYTHON=/path/to/python and try again." >&2
  exit 1
}

run_compare() {
  local python_bin
  local has_run_script=0
  local args=("$@")

  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--run-script" ]]; then
      has_run_script=1
      break
    fi
  done

  if [[ "${has_run_script}" -eq 0 ]]; then
    args=(--run-script "${REAL_IMPL}" "${args[@]}")
  fi

  python_bin="$(select_compare_python)"
  "${python_bin}" "${EVAL_SCRIPT}" "${args[@]}"
}

MODE="full"
if (($# > 0)); then
  case "$1" in
    help|-h|--help|full|aideck|real|overlay|compare)
      MODE="$1"
      shift
      ;;
    *)
      MODE="full"
      ;;
  esac
fi

case "${MODE}" in
  help|-h|--help)
    show_usage
    ;;
  aideck)
    exec "${AIDECK_IMPL}" "$@"
    ;;
  real)
    exec "${REAL_IMPL}" "$@"
    ;;
  overlay)
    exec "${REAL_IMPL}" --overlay-only "$@"
    ;;
  compare)
    run_compare "$@"
    ;;
  full)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      show_usage
      exit 0
    fi

    aideck_rc=0
    set +e
    "${AIDECK_IMPL}"
    aideck_rc=$?
    set -e
    if [[ "${aideck_rc}" -ne 0 ]]; then
      echo "[warn] Staged AI-Deck validation failed with exit code ${aideck_rc}; continuing into checkpoint-vs-application evaluation." >&2
    fi

    compare_rc=0
    set +e
    run_compare "$@"
    compare_rc=$?
    set -e

    if [[ "${aideck_rc}" -ne 0 || "${compare_rc}" -ne 0 ]]; then
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}'." >&2
    echo >&2
    show_usage >&2
    exit 1
    ;;
esac

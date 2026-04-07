#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

select_python() {
  if [[ -n "${PLAIN_FOLLOW_PYTHON:-}" ]]; then
    printf '%s\n' "${PLAIN_FOLLOW_PYTHON}"
    return 0
  fi
  if [[ -x "${SCRIPT_DIR}/../nemoenv/bin/python3" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../nemoenv/bin/python3"
    return 0
  fi
  if [[ -x "${SCRIPT_DIR}/../nemoenv/bin/python" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../nemoenv/bin/python"
    return 0
  fi
  if [[ -f "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../nemoenv/Scripts/python.exe"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi

  echo "ERROR: no Python interpreter found for run_plain_follow.sh" >&2
  exit 1
}

PYTHON_BIN="$(select_python)"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/export/run_plain_follow_release.py" "$@"

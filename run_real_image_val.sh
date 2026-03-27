#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "${SCRIPT_DIR}/tools/run_real_image_val_impl.sh" "$@"

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_nemo_quant_analysis import *  # noqa: F401,F403
from export_nemo_quant_cli import ExportRequest, build_arg_parser, main
from export_nemo_quant_core import *  # noqa: F401,F403
from export_nemo_quant_presets import *  # noqa: F401,F403
from export_nemo_quant_scopes import (  # noqa: F401
    clone_model_and_run_with_integer_add_scale_selection,
    integer_add_scale_selection_scope,
    patch_integer_add_scale_selection,
    restore_integer_add_scale_selection,
    run_with_integer_add_scale_selection,
)
from export_nemo_quant_sweeps import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())

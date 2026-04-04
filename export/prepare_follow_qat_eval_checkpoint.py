#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.follow_model_factory import (  # noqa: E402
    build_follow_model,
    checkpoint_state_dict,
    follow_model_kwargs_from_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a QAT follow-model checkpoint down to the float-model state_dict keys so "
            "the standard checkpoint loaders and evaluators can consume it."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Source follow-model checkpoint.")
    parser.add_argument("--output", required=True, help="Destination checkpoint path.")
    parser.add_argument("--report-json", default=None, help="Optional filter report output path.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing float-model keys after filtering instead of failing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = None if args.report_json is None else Path(args.report_json).expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if report_path is not None and report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {report_path}")

    payload = torch.load(ckpt_path, map_location=torch.device("cpu"))
    payload_dict = payload if isinstance(payload, dict) else {"state_dict": payload}
    source_state = checkpoint_state_dict(payload_dict)
    if not isinstance(source_state, dict):
        raise TypeError("Checkpoint payload is not a dict-like state_dict.")

    model_kwargs = follow_model_kwargs_from_metadata(payload_dict)
    model = build_follow_model(**model_kwargs).cpu().eval()
    target_state = model.state_dict()

    filtered_state: dict[str, Any] = {}
    missing_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for key, target_value in target_state.items():
        if key not in source_state:
            missing_keys.append(key)
            continue
        source_value = source_state[key]
        if hasattr(source_value, "shape") and hasattr(target_value, "shape"):
            if tuple(source_value.shape) != tuple(target_value.shape):
                shape_mismatches.append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
                missing_keys.append(key)
                continue
        filtered_state[key] = source_value

    unexpected_keys = sorted(key for key in source_state.keys() if key not in target_state)
    report = {
        "source_checkpoint": str(ckpt_path),
        "output_checkpoint": str(output_path),
        "model_kwargs": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in model_kwargs.items()
        },
        "source_key_count": int(len(source_state)),
        "target_key_count": int(len(target_state)),
        "filtered_key_count": int(len(filtered_state)),
        "missing_key_count": int(len(missing_keys)),
        "unexpected_key_count": int(len(unexpected_keys)),
        "shape_mismatch_count": int(len(shape_mismatches)),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
    }

    if missing_keys and not args.allow_missing:
        raise RuntimeError(
            "Filtered checkpoint is missing float-model keys: "
            f"{missing_keys[:20]} (total={len(missing_keys)})"
        )

    output_payload = dict(payload_dict)
    output_payload["state_dict"] = filtered_state
    output_payload["source_qat_checkpoint"] = str(ckpt_path)
    output_payload["qat_eval_filter_report"] = report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    if report_path is not None:
        write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

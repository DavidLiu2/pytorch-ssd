#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if the promoted hybrid_follow exporter patch regresses against "
            "the stored reference metrics."
        )
    )
    parser.add_argument("--comparison-summary", required=True, help="summary.json from compare_hybrid_follow_export_runs.py")
    parser.add_argument("--reference-json", required=True, help="Pinned regression reference JSON.")
    parser.add_argument("--output-json", default=None, help="Optional path for the regression report JSON.")
    return parser.parse_args()


def read_json(path_value: str | Path) -> Any:
    return json.loads(Path(path_value).read_text(encoding="utf-8"))


def get_dataset_mean_score(summary: dict[str, Any], dataset_name: str) -> float:
    dataset = (summary.get("datasets") or {}).get(dataset_name) or {}
    patched = (dataset.get("patched") or {}).get("application") or {}
    value = patched.get("mean_score")
    if value is None:
        raise KeyError(f"Missing patched application mean score for dataset: {dataset_name}")
    return float(value)


def get_warn_count(summary: dict[str, Any]) -> int:
    stage_loc = ((summary.get("stage_localization") or {}).get("patched") or {})
    export_fidelity = stage_loc.get("export_fidelity") or {}
    value = export_fidelity.get("id_to_onnx_threshold_warn_count")
    if value is None:
        raise KeyError("Missing patched stage-localization id_to_onnx_threshold_warn_count")
    return int(value)


def main() -> int:
    args = parse_args()
    summary = read_json(args.comparison_summary)
    reference = read_json(args.reference_json)
    tolerance = float(reference.get("tolerance", 0.0))

    checks = []
    failures = []

    for dataset_name, limit in (reference.get("application_mean_score_limits") or {}).items():
        current = get_dataset_mean_score(summary, dataset_name)
        threshold = float(limit)
        passed = current <= (threshold + tolerance)
        checks.append(
            {
                "type": "application_mean_score",
                "dataset": dataset_name,
                "current": current,
                "threshold": threshold,
                "tolerance": tolerance,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"{dataset_name} mean score regressed: current={current:.6f} threshold={threshold:.6f}"
            )

    max_warn_count = int(reference.get("max_id_to_onnx_warn_count", 0))
    current_warn_count = get_warn_count(summary)
    warn_passed = current_warn_count <= max_warn_count
    checks.append(
        {
            "type": "id_to_onnx_warn_count",
            "current": current_warn_count,
            "threshold": max_warn_count,
            "passed": warn_passed,
        }
    )
    if not warn_passed:
        failures.append(
            f"id_to_onnx warn count regressed: current={current_warn_count} threshold={max_warn_count}"
        )

    report = {
        "comparison_summary": str(Path(args.comparison_summary)),
        "reference_json": str(Path(args.reference_json)),
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }

    output_path = Path(args.output_json) if args.output_json else Path(args.comparison_summary).with_name("regression_check.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures:
        for failure in failures:
            print(f"REGRESSION: {failure}")
        return 1

    print(f"Regression check passed: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

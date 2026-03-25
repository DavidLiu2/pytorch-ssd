#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from visualize_hybrid_follow_prediction import (
    DEFAULT_ANN,
    DEFAULT_Q_SCALE,
    DEFAULT_VIS_THRESHOLD,
    generate_overlay_for_sample,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate prediction overlays for every sample directory inside a "
            "hybrid_follow real-image validation results folder."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Validation results directory containing per-sample folders.",
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANN),
        help="Optional COCO annotations json used to draw the largest person box.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sample directories to process.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any sample cannot be visualized.",
    )
    parser.add_argument(
        "--q-scale",
        type=float,
        default=DEFAULT_Q_SCALE,
        help="Quantization scale used to decode the 3-value hybrid_follow tensor.",
    )
    parser.add_argument(
        "--vis-threshold",
        type=float,
        default=DEFAULT_VIS_THRESHOLD,
        help="Visibility confidence threshold used to mark visible/not-visible.",
    )
    return parser.parse_args()


def discover_sample_dirs(results_dir: Path) -> list[Path]:
    sample_dirs = []
    for comparison_path in results_dir.rglob("comparison.json"):
        sample_dir = comparison_path.parent
        if (sample_dir / "gvsoc_final_tensor.json").is_file():
            sample_dirs.append(sample_dir)
    return sorted(set(sample_dirs))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "sample_dir",
        "status",
        "overlay_path",
        "image_path",
        "raw_values",
        "x_offset",
        "size_proxy",
        "visibility_confidence",
        "visible",
        "gt_x_offset",
        "gt_size_proxy",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_dir": row.get("sample_dir"),
                    "status": row.get("status"),
                    "overlay_path": row.get("overlay_path"),
                    "image_path": row.get("image_path"),
                    "raw_values": json.dumps(row.get("raw_values")),
                    "x_offset": row.get("x_offset"),
                    "size_proxy": row.get("size_proxy"),
                    "visibility_confidence": row.get("visibility_confidence"),
                    "visible": row.get("visible"),
                    "gt_x_offset": row.get("gt_x_offset"),
                    "gt_size_proxy": row.get("gt_size_proxy"),
                    "error": row.get("error", ""),
                }
            )


def main() -> int:
    args = parse_args()

    results_dir = resolve_repo_path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    sample_dirs = discover_sample_dirs(results_dir)
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under {results_dir}")

    rows: list[dict] = []
    failed = 0

    for index, sample_dir in enumerate(sample_dirs, start=1):
        try:
            result = generate_overlay_for_sample(
                sample_dir=sample_dir,
                annotations_path=args.annotations,
                q_scale=args.q_scale,
                vis_threshold=args.vis_threshold,
            )
            row = result.to_dict()
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)
            print(f"[{index}/{len(sample_dirs)}] OK   {sample_dir.name} -> {result.overlay_path}")
        except Exception as exc:
            failed += 1
            row = {
                "sample_dir": str(sample_dir),
                "status": "fail",
                "overlay_path": None,
                "image_path": None,
                "raw_values": None,
                "x_offset": None,
                "size_proxy": None,
                "visibility_confidence": None,
                "visible": None,
                "gt_x_offset": None,
                "gt_size_proxy": None,
                "error": str(exc),
            }
            rows.append(row)
            print(f"[{index}/{len(sample_dirs)}] FAIL {sample_dir.name}: {exc}")
            if args.strict:
                break

    summary_payload = {
        "results_dir": str(results_dir),
        "annotations": str(resolve_repo_path(args.annotations)),
        "count": len(rows),
        "failed": failed,
        "results": rows,
    }

    summary_json_path = results_dir / "overlay_summary.json"
    summary_csv_path = results_dir / "overlay_summary.csv"
    write_json(summary_json_path, summary_payload)
    write_csv(summary_csv_path, rows)

    print(f"Overlay summary JSON: {summary_json_path}")
    print(f"Overlay summary CSV:  {summary_csv_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

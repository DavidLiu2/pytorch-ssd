#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


TARGET_KEY_MAP = {
    "onnx": "onnx",
    "application": "gvsoc",
}

TRANSITION_ORDER = [
    "FP -> FQ",
    "FQ -> ID",
    "ID/ONNX export",
    "golden -> GVSOC runtime",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs patched hybrid_follow validation runs and summarize "
            "final drift, win/loss counts, worst regressions, and optional stage localization."
        )
    )
    parser.add_argument("--baseline-dir", required=True, help="Directory containing baseline dataset result dirs.")
    parser.add_argument("--patched-dir", required=True, help="Directory containing patched dataset result dirs.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.{md,json}.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--patched-label", default="patched")
    parser.add_argument("--baseline-stage-localization", default=None)
    parser.add_argument("--patched-stage-localization", default=None)
    return parser.parse_args()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_summary_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Summary not found: {path}")
    return path


def discover_dataset_summaries(root: Path) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.endswith("_stage_localization"):
            continue
        summary_path = child / "summary.json"
        if summary_path.is_file():
            datasets[child.name] = summary_path
    return datasets


def x_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def stage_metric_row(row: dict[str, Any], target_name: str) -> dict[str, Any]:
    target_key = TARGET_KEY_MAP[target_name]
    pytorch = (row.get("stage_outputs") or {}).get("fp") or (row.get("stage_outputs") or {}).get("pytorch") or {}
    target = (row.get("stage_outputs") or {}).get(target_key) or {}

    pytorch_x = float(pytorch["x_offset_raw"])
    pytorch_size = float(pytorch["size_proxy"])
    pytorch_vis = float(pytorch["visibility_logit"])
    target_x = float(target["x_offset_raw"])
    target_size = float(target["size_proxy"])
    target_vis = float(target["visibility_logit"])

    vis_conf_abs_diff = abs(sigmoid(pytorch_vis) - sigmoid(target_vis))
    score = abs(pytorch_x - target_x) + abs(pytorch_size - target_size) + vis_conf_abs_diff
    return {
        "image_name": row["image_name"],
        "score": float(score),
        "x_abs_diff": abs(pytorch_x - target_x),
        "size_abs_diff": abs(pytorch_size - target_size),
        "vis_logit_abs_diff": abs(pytorch_vis - target_vis),
        "vis_conf_abs_diff": float(vis_conf_abs_diff),
        "x_sign_flip": x_sign(pytorch_x) != x_sign(target_x),
        "vis_threshold_flip": (pytorch_vis >= 0.0) != (target_vis >= 0.0),
        "pytorch": {
            "x_offset": pytorch_x,
            "size_proxy": pytorch_size,
            "visibility_logit": pytorch_vis,
        },
        "target": {
            "x_offset": target_x,
            "size_proxy": target_size,
            "visibility_logit": target_vis,
        },
        "drift_onset": (row.get("drift_onset") or {}).get("message"),
        "stage_drift_dir": row.get("stage_drift_dir"),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_score": None,
            "median_score": None,
            "mean_abs_diff": {
                "x": None,
                "size": None,
                "vis_logit": None,
                "vis_conf": None,
            },
            "x_sign_flip_rate": None,
            "vis_threshold_flip_rate": None,
        }

    return {
        "count": len(rows),
        "mean_score": float(mean(row["score"] for row in rows)),
        "median_score": float(median(row["score"] for row in rows)),
        "mean_abs_diff": {
            "x": float(mean(row["x_abs_diff"] for row in rows)),
            "size": float(mean(row["size_abs_diff"] for row in rows)),
            "vis_logit": float(mean(row["vis_logit_abs_diff"] for row in rows)),
            "vis_conf": float(mean(row["vis_conf_abs_diff"] for row in rows)),
        },
        "x_sign_flip_rate": float(sum(1 for row in rows if row["x_sign_flip"])) / float(len(rows)),
        "vis_threshold_flip_rate": float(sum(1 for row in rows if row["vis_threshold_flip"])) / float(len(rows)),
    }


def compare_run_summaries(
    baseline_summary: dict[str, Any],
    patched_summary: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    baseline_rows = {
        row["image_name"]: row
        for row in baseline_summary.get("results", [])
    }
    patched_rows = {
        row["image_name"]: row
        for row in patched_summary.get("results", [])
    }
    common_names = sorted(set(baseline_rows) & set(patched_rows))

    dataset_report: dict[str, Any] = {
        "dataset": dataset_name,
        "baseline_images_dir": baseline_summary.get("images_dir"),
        "patched_images_dir": patched_summary.get("images_dir"),
        "baseline_count": len(baseline_rows),
        "patched_count": len(patched_rows),
        "common_count": len(common_names),
        "only_in_baseline": sorted(set(baseline_rows) - set(patched_rows)),
        "only_in_patched": sorted(set(patched_rows) - set(baseline_rows)),
        "baseline": {},
        "patched": {},
        "comparison": {},
    }

    for target_name in TARGET_KEY_MAP:
        baseline_metric_rows = [stage_metric_row(baseline_rows[name], target_name) for name in common_names]
        patched_metric_rows = [stage_metric_row(patched_rows[name], target_name) for name in common_names]
        baseline_aggregate = aggregate_metric_rows(baseline_metric_rows)
        patched_aggregate = aggregate_metric_rows(patched_metric_rows)

        per_image_deltas = []
        improved = 0
        worsened = 0
        tied = 0
        for baseline_row, patched_row in zip(baseline_metric_rows, patched_metric_rows):
            score_delta = float(patched_row["score"] - baseline_row["score"])
            if score_delta < -1e-12:
                improved += 1
            elif score_delta > 1e-12:
                worsened += 1
            else:
                tied += 1
            per_image_deltas.append(
                {
                    "image_name": baseline_row["image_name"],
                    "baseline_score": float(baseline_row["score"]),
                    "patched_score": float(patched_row["score"]),
                    "score_delta": score_delta,
                    "baseline_mean_abs_diff": {
                        "x": float(baseline_row["x_abs_diff"]),
                        "size": float(baseline_row["size_abs_diff"]),
                        "vis_logit": float(baseline_row["vis_logit_abs_diff"]),
                    },
                    "patched_mean_abs_diff": {
                        "x": float(patched_row["x_abs_diff"]),
                        "size": float(patched_row["size_abs_diff"]),
                        "vis_logit": float(patched_row["vis_logit_abs_diff"]),
                    },
                    "baseline_drift_onset": baseline_row.get("drift_onset"),
                    "patched_drift_onset": patched_row.get("drift_onset"),
                    "baseline_stage_drift_dir": baseline_row.get("stage_drift_dir"),
                    "patched_stage_drift_dir": patched_row.get("stage_drift_dir"),
                }
            )

        per_image_deltas.sort(key=lambda item: float(item["score_delta"]), reverse=True)
        dataset_report["baseline"][target_name] = baseline_aggregate
        dataset_report["patched"][target_name] = patched_aggregate
        dataset_report["comparison"][target_name] = {
            "improved_count": improved,
            "worsened_count": worsened,
            "tied_count": tied,
            "mean_score_delta": (
                None
                if baseline_aggregate["mean_score"] is None or patched_aggregate["mean_score"] is None
                else float(patched_aggregate["mean_score"] - baseline_aggregate["mean_score"])
            ),
            "worst_regressions": per_image_deltas[:5],
            "best_improvements": list(reversed(per_image_deltas[-5:])),
        }

    return dataset_report


def summarize_stage_localization(path_value: str | Path | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    payload = read_json(ensure_summary_path(path_value))
    transition_counts = payload.get("transition_counts") or {}
    earliest_nonzero = None
    for label in TRANSITION_ORDER:
        if int(transition_counts.get(label, 0) or 0) > 0:
            earliest_nonzero = label
            break
    ranked = sorted(
        (
            (label, int(count))
            for label, count in transition_counts.items()
            if label != "none"
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "summary_path": str(ensure_summary_path(path_value)),
        "hybrid_follow_export_preset": payload.get("hybrid_follow_export_preset"),
        "count": payload.get("count"),
        "transition_counts": transition_counts,
        "earliest_nonzero_transition": earliest_nonzero,
        "most_common_transition": ranked[0][0] if ranked and ranked[0][1] > 0 else None,
        "export_fidelity": payload.get("export_fidelity") or {},
    }


def metric_line(label: str, metrics: dict[str, Any]) -> str:
    if metrics["mean_score"] is None:
        return f"- {label}: unavailable"
    return (
        f"- {label}: mean_score={metrics['mean_score']:.6f} "
        f"x={metrics['mean_abs_diff']['x']:.6f} "
        f"size={metrics['mean_abs_diff']['size']:.6f} "
        f"vis_logit={metrics['mean_abs_diff']['vis_logit']:.6f} "
        f"x_sign_flip={metrics['x_sign_flip_rate']:.3f} "
        f"vis_flip={metrics['vis_threshold_flip_rate']:.3f}"
    )


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Follow Export Patch Comparison",
        "",
        f"- Baseline dir: `{summary['baseline_dir']}`",
        f"- Patched dir: `{summary['patched_dir']}`",
        f"- Baseline label: `{summary['baseline_label']}`",
        f"- Patched label: `{summary['patched_label']}`",
        "",
    ]

    for dataset_name, report in summary["datasets"].items():
        app_baseline = report["baseline"]["application"]
        app_patched = report["patched"]["application"]
        app_compare = report["comparison"]["application"]
        onnx_baseline = report["baseline"]["onnx"]
        onnx_patched = report["patched"]["onnx"]

        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- Common samples: `{report['common_count']}`",
                metric_line(f"{summary['baseline_label']} application", app_baseline),
                metric_line(f"{summary['patched_label']} application", app_patched),
                metric_line(f"{summary['baseline_label']} ONNX", onnx_baseline),
                metric_line(f"{summary['patched_label']} ONNX", onnx_patched),
                (
                    f"- Application wins/losses/ties vs baseline: "
                    f"`{app_compare['improved_count']}` / `{app_compare['worsened_count']}` / `{app_compare['tied_count']}`"
                ),
                f"- Application mean score delta: `{app_compare['mean_score_delta']:.6f}`",
                "",
                "| Worst regression | baseline score | patched score | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in app_compare["worst_regressions"]:
            lines.append(
                "| `{}` | `{:.6f}` | `{:.6f}` | `{:.6f}` |".format(
                    item["image_name"],
                    float(item["baseline_score"]),
                    float(item["patched_score"]),
                    float(item["score_delta"]),
                )
            )
        lines.append("")

    if summary.get("stage_localization"):
        lines.extend(["## Stage Localization", ""])
        for label in ("baseline", "patched"):
            payload = summary["stage_localization"].get(label)
            if payload is None:
                continue
            export_fidelity = payload.get("export_fidelity") or {}
            anti_collapse = payload.get("anti_collapse") or {}
            lines.extend(
                [
                    f"- {label}: preset=`{payload.get('hybrid_follow_export_preset')}` "
                    f"earliest_nonzero=`{payload.get('earliest_nonzero_transition')}` "
                    f"most_common=`{payload.get('most_common_transition')}` "
                    f"id_to_onnx_warn=`{export_fidelity.get('id_to_onnx_threshold_warn_count')}` "
                    f"golden_to_gvsoc_warn=`{export_fidelity.get('golden_to_gvsoc_threshold_warn_count')}`",
                ]
            )
            gvsoc_collapse = anti_collapse.get("gvsoc") or {}
            if gvsoc_collapse:
                lines.append(
                    f"  gvsoc anti-collapse: sign_flip=`{gvsoc_collapse.get('sign_flip_rate')}` "
                    f"corr=`{gvsoc_collapse.get('correlation')}` "
                    f"slope=`{gvsoc_collapse.get('slope')}` "
                    f"collapsed_fraction=`{gvsoc_collapse.get('collapsed_fraction')}`"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    patched_dir = Path(args.patched_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_datasets = discover_dataset_summaries(baseline_dir)
    patched_datasets = discover_dataset_summaries(patched_dir)
    common_dataset_names = sorted(set(baseline_datasets) & set(patched_datasets))
    if not common_dataset_names:
        raise RuntimeError(
            f"No common dataset summaries found between {baseline_dir} and {patched_dir}."
        )

    datasets = {}
    for dataset_name in common_dataset_names:
        datasets[dataset_name] = compare_run_summaries(
            read_json(baseline_datasets[dataset_name]),
            read_json(patched_datasets[dataset_name]),
            dataset_name=dataset_name,
        )

    summary = {
        "baseline_dir": str(baseline_dir),
        "patched_dir": str(patched_dir),
        "baseline_label": args.baseline_label,
        "patched_label": args.patched_label,
        "datasets": datasets,
        "stage_localization": {
            "baseline": summarize_stage_localization(args.baseline_stage_localization),
            "patched": summarize_stage_localization(args.patched_stage_localization),
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")

    print(f"Comparison summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

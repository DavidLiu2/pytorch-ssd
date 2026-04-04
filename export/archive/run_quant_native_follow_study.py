#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    (parent for parent in SCRIPT_DIR.parents if (parent / "models").is_dir() and (parent / "export").is_dir()),
    SCRIPT_DIR.parent,
)
EXPORT_DIR = PROJECT_DIR / "export"
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "quant_native_val"
DEFAULT_REP16_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
DEFAULT_HARD_CASE_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "5_microblock_add_only_patch"
    / "input_sets"
    / "hard_case_subset"
)
DEFAULT_BASELINE_SUMMARY = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "10_refactor_stage_localization_baseline_rep16"
    / "summary.json"
)
DEFAULT_BASELINE_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_follow_score.pth"

CANDIDATE_SPECS = {
    "plain_xbin9_size_scalar": {
        "model_type": "plain_follow",
        "follow_head_type": "xbin9_size_scalar",
    },
    "plain_xbin9_size_bucket4": {
        "model_type": "plain_follow",
        "follow_head_type": "xbin9_size_bucket4",
    },
    "plain_lcr3_residual_size_scalar": {
        "model_type": "plain_follow",
        "follow_head_type": "lcr3_residual_size_scalar",
    },
    "dronet_lite_xbin9_size_scalar": {
        "model_type": "dronet_lite_follow",
        "follow_head_type": "xbin9_size_scalar",
    },
    "dronet_lite_xbin9_size_bucket4": {
        "model_type": "dronet_lite_follow",
        "follow_head_type": "xbin9_size_bucket4",
    },
    "dronet_lite_lcr3_residual_size_scalar": {
        "model_type": "dronet_lite_follow",
        "follow_head_type": "lcr3_residual_size_scalar",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, quantize, export, and compare quant-native follow candidates under logs/quant_native_val."
        )
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(CANDIDATE_SPECS),
        choices=sorted(CANDIDATE_SPECS),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--baseline-ckpt", default=str(DEFAULT_BASELINE_CKPT))
    parser.add_argument("--train-python", default=str(REPO_ROOT / "trainenv" / "bin" / "python"))
    parser.add_argument("--eval-python", default=str(REPO_ROOT / "nemoenv" / "bin" / "python3"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=32)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {shlex.join(command)}. See {log_path}."
        )


def load_baseline_checkpoint(path: Path, python_bin: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    command = [
        str(python_bin),
        "-c",
        (
            "import json, torch, sys; "
            "obj=torch.load(sys.argv[1], map_location='cpu'); "
            "payload={k:v for k,v in obj.items() if k!='state_dict'} if isinstance(obj, dict) else {}; "
            "print(json.dumps(payload))"
        ),
        str(path),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_DIR), capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip() or "{}")


def candidate_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    rep16 = ((summary.get("datasets") or {}).get("rep16") or {}).get("onnx") or {}
    hard_case = ((summary.get("datasets") or {}).get("hard_case") or {}).get("onnx") or {}
    fidelity = (summary.get("quant_fidelity") or {}).get("float_to_onnx_bin_preservation") or {}
    pipeline = summary.get("pipeline_complexity") or {}
    compat_rank = {"pass": 0.0, "warn": 1.0, "fail": 2.0}.get(pipeline.get("compatibility_status"), 3.0)
    return (
        compat_rank,
        float(rep16.get("follow_score") or 1e9),
        float(hard_case.get("follow_score") or 1e9),
        -float(
            fidelity.get("x_bin_exact_match_rate")
            or fidelity.get("x_coarse_exact_match_rate")
            or 0.0
        ),
        float(summary.get("parameter_count") or 1e12),
    )


def summarize_candidates(
    results: dict[str, dict[str, Any]],
    *,
    baseline_ckpt: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(results.items(), key=lambda item: candidate_sort_key(item[1]))
    best_name, best_summary = ordered[0]
    return {
        "recommended_candidate": best_name,
        "recommended_summary_path": str(Path(best_summary["summary_path"]).resolve()) if "summary_path" in best_summary else None,
        "baseline_checkpoint": baseline_ckpt,
        "baseline_summary": baseline_summary,
        "ordered_candidates": [name for name, _ in ordered],
    }


def build_summary_markdown(
    aggregate: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> str:
    baseline_ckpt = aggregate.get("baseline_checkpoint") or {}
    baseline_summary = aggregate.get("baseline_summary") or {}
    lines = [
        "# Quant-Native Follow Restart",
        "",
        "## Old Family Baseline",
        f"- float follow_score: `{((baseline_ckpt.get('val_stats') or {}).get('follow_score'))}`",
        f"- float x_mae: `{((baseline_ckpt.get('val_stats') or {}).get('x_mae'))}`",
        f"- float size_mae: `{((baseline_ckpt.get('val_stats') or {}).get('size_mae'))}`",
        f"- rep16 dominant issue bucket: `{((baseline_summary.get('dominant_issue') or {}).get('dominant_bucket'))}`",
        f"- baseline onnx anti-collapse: `{(((baseline_summary.get('anti_collapse') or {}).get('onnx') or {}))}`",
        "",
        "## New Candidates",
    ]
    for name in aggregate["ordered_candidates"]:
        summary = results[name]
        rep16 = ((summary.get("datasets") or {}).get("rep16") or {}).get("onnx") or {}
        hard_case = ((summary.get("datasets") or {}).get("hard_case") or {}).get("onnx") or {}
        fidelity = (summary.get("quant_fidelity") or {}).get("float_to_onnx_bin_preservation") or {}
        pipeline = summary.get("pipeline_complexity") or {}
        lines.extend(
            [
                f"### {name}",
                f"- float follow_score: `{(summary.get('float_validation') or {}).get('follow_score')}`",
                f"- rep16 onnx follow_score: `{rep16.get('follow_score')}`",
                f"- hard-case onnx follow_score: `{hard_case.get('follow_score')}`",
                f"- float->onnx bin preservation: `{fidelity}`",
                f"- earliest bad boundary: `{((summary.get('quant_fidelity') or {}).get('earliest_bad_boundary') or {}).get('boundary_name')}`",
                f"- custom export patches required: `{pipeline.get('custom_export_patch_count')}`",
                f"- graph repair/head collapse/residual rescue: `{pipeline.get('graph_repair_needed')}` / `{pipeline.get('head_collapse_needed')}` / `{pipeline.get('residual_rescue_needed')}`",
                f"- compatibility status: `{pipeline.get('compatibility_status')}`",
                f"- summary: `{summary.get('summary_path')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            f"- next baseline: `{aggregate['recommended_candidate']}`",
            "- decision rule: prefer the smallest pipeline-safe model that holds float usefulness while materially improving quantized faithfulness and bin preservation.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = load_baseline_checkpoint(Path(args.baseline_ckpt), Path(args.train_python))
    baseline_summary = load_json(Path(args.baseline_summary)) if Path(args.baseline_summary).is_file() else {}

    results: dict[str, dict[str, Any]] = {}
    for candidate_name in args.candidates:
        spec = CANDIDATE_SPECS[candidate_name]
        candidate_dir = output_dir / candidate_name
        train_dir = candidate_dir / "train"
        eval_dir = candidate_dir / "eval"
        best_ckpt = train_dir / f"{spec['model_type']}_best_follow_score.pth"

        if not args.reuse_existing or not best_ckpt.is_file():
            train_command = [
                str(Path(args.train_python).expanduser()),
                str(PROJECT_DIR / "train.py"),
                "--model-type",
                spec["model_type"],
                "--follow-head-type",
                spec["follow_head_type"],
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--num_workers",
                str(args.num_workers),
                "--output_dir",
                str(train_dir.relative_to(PROJECT_DIR.parent)),
                "--max-train-batches",
                str(args.max_train_batches),
                "--max-val-batches",
                str(args.max_val_batches),
            ]
            run_logged(train_command, cwd=PROJECT_DIR, log_path=candidate_dir / "train.log")

        if not best_ckpt.is_file():
            raise FileNotFoundError(best_ckpt)

        if not args.reuse_existing or not (eval_dir / "summary.json").is_file():
            eval_command = [
                str(Path(args.eval_python).expanduser()),
                str(PROJECT_DIR / "export" / "evaluate_quant_native_follow.py"),
                "--ckpt",
                str(best_ckpt),
                "--output-dir",
                str(eval_dir),
                "--rep16-dir",
                str(Path(args.rep16_dir).expanduser().resolve()),
                "--hard-case-dir",
                str(Path(args.hard_case_dir).expanduser().resolve()),
                "--calib-dir",
                str(Path(args.rep16_dir).expanduser().resolve()),
                "--calib-batches",
                str(args.calib_batches),
                "--overwrite",
            ]
            run_logged(eval_command, cwd=PROJECT_DIR, log_path=candidate_dir / "eval.log")

        summary = load_json(eval_dir / "summary.json")
        summary["summary_path"] = str((eval_dir / "summary.md").resolve())
        results[candidate_name] = summary

    aggregate = summarize_candidates(
        results,
        baseline_ckpt=baseline_ckpt,
        baseline_summary=baseline_summary,
    )
    aggregate_path = output_dir / "comparison_summary.json"
    write_json(aggregate_path, aggregate)
    write_markdown(output_dir / "comparison_summary.md", build_summary_markdown(aggregate, results))


if __name__ == "__main__":
    main()

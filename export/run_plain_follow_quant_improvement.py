#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_CKPT = PROJECT_DIR / "training" / "plain_follow" / "plain_follow_best_follow_score.pth"
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
DEFAULT_VAL_IMAGES = PROJECT_DIR / "data" / "coco" / "images" / "val2017"
DEFAULT_VAL_ANN = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_TRAIN_IMAGES = PROJECT_DIR / "data" / "coco" / "images" / "train2017"
DEFAULT_TRAIN_ANN = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_train2017.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "plain_follow_quant_val" / "focused_qat_study"
DEFAULT_WINDOWS_PYTHON = Path("/mnt/c/Python313/python.exe")

PROMOTION_BASELINE = {
    "follow_score": 0.07207,
    "x_mae": 0.04390,
    "size_mae": 0.09390,
    "no_person_fp_rate": 0.1667,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the supported plain_follow quant-improvement loop: fixed PTQ baseline, "
            "deployment-matched calibration manifest, focused stem/stage1/output_head QAT, "
            "epoch-by-epoch quant evaluation, and rep16 overlay generation."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--follow-head-type", default="xbin9_size_bucket4")
    parser.add_argument(
        "--stem-mode",
        default="conv_bn_relu",
        choices=["conv_bn_relu", "delayed_relu"],
        help="Stem architecture used for the focused-QAT candidate. Baseline PTQ still evaluates the input checkpoint as-is.",
    )
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_VAL_ANN))
    parser.add_argument("--calib-image-dir", default=str(DEFAULT_VAL_IMAGES))
    parser.add_argument("--calib-annotations", default=str(DEFAULT_VAL_ANN))
    parser.add_argument("--calib-manifest", default=None)
    parser.add_argument("--calib-target-count", type=int, default=128)
    parser.add_argument("--train-image-dir", default=str(DEFAULT_TRAIN_IMAGES))
    parser.add_argument("--train-annotations", default=str(DEFAULT_TRAIN_ANN))
    parser.add_argument("--train-sample-manifest", default=None)
    parser.add_argument("--train-manifest-target-count", type=int, default=2048)
    parser.add_argument("--train-manifest-max-images", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--qat-bits", type=int, default=8)
    parser.add_argument("--qat-calib-batches", type=int, default=32)
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--python", default=None, help="Fallback interpreter for training and evaluation steps.")
    parser.add_argument("--train-python", default=None, help="Interpreter used for train.py.")
    parser.add_argument("--eval-python", default=None, help="Interpreter used for export/eval scripts.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_python(user_value: str | None, fallback_value: str | None = None) -> str:
    if user_value:
        return user_value
    if fallback_value:
        return fallback_value
    if DEFAULT_WINDOWS_PYTHON.is_file():
        return str(DEFAULT_WINDOWS_PYTHON)
    return sys.executable or "python3"


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_logged(command: list[str], *, log_path: Path, cwd: Path = PROJECT_DIR) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {format_command(command)}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {format_command(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: see {log_path}")


def build_manifest_if_needed(
    *,
    python_bin: str,
    output_path: Path,
    image_dir: Path,
    annotations: Path,
    follow_head_type: str,
    model_type: str,
    target_count: int,
    max_images: int,
    overwrite: bool,
    log_path: Path,
) -> Path:
    command = [
        python_bin,
        str(SCRIPT_DIR / "build_follow_calibration_manifest.py"),
        "--image-dir",
        str(image_dir),
        "--annotations",
        str(annotations),
        "--model-type",
        model_type,
        "--follow-head-type",
        follow_head_type,
        "--target-count",
        str(int(target_count)),
        "--output",
        str(output_path),
    ]
    if int(max_images) > 0:
        command.extend(["--max-images", str(int(max_images))])
    if overwrite:
        command.append("--overwrite")
    run_logged(command, log_path=log_path)
    return output_path


def evaluate_checkpoint(
    *,
    python_bin: str,
    ckpt_path: Path,
    output_dir: Path,
    candidate_name: str,
    rep16_dir: Path,
    hard_case_dir: Path,
    annotations: Path,
    calib_manifest: Path,
    calib_dir: Path,
    vis_thresh: float,
    overwrite: bool,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        python_bin,
        str(SCRIPT_DIR / "evaluate_quant_native_follow.py"),
        "--ckpt",
        str(ckpt_path),
        "--output-dir",
        str(output_dir),
        "--candidate-name",
        candidate_name,
        "--rep16-dir",
        str(rep16_dir),
        "--hard-case-dir",
        str(hard_case_dir),
        "--annotations",
        str(annotations),
        "--calib-dir",
        str(calib_dir),
        "--calib-manifest",
        str(calib_manifest),
        "--vis-thresh",
        str(float(vis_thresh)),
    ]
    if overwrite:
        command.append("--overwrite")
    run_logged(command, log_path=log_path)
    return read_json(output_dir / "summary.json")


def build_overlay_summary(
    *,
    python_bin: str,
    ckpt_path: Path,
    onnx_path: Path,
    output_dir: Path,
    rep16_dir: Path,
    annotations: Path,
    vis_thresh: float,
    overwrite: bool,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        python_bin,
        str(SCRIPT_DIR / "compare_quant_native_follow_rep16_overlays.py"),
        "--ckpt",
        str(ckpt_path),
        "--onnx",
        str(onnx_path),
        "--output-dir",
        str(output_dir),
        "--images-dir",
        str(rep16_dir),
        "--annotations",
        str(annotations),
        "--vis-thresh",
        str(float(vis_thresh)),
    ]
    if overwrite:
        command.append("--overwrite")
    run_logged(command, log_path=log_path)
    return read_json(output_dir / "comparison_summary.json")


def filter_qat_checkpoint(
    *,
    python_bin: str,
    ckpt_path: Path,
    output_path: Path,
    report_path: Path,
    overwrite: bool,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        python_bin,
        str(SCRIPT_DIR / "prepare_follow_qat_eval_checkpoint.py"),
        "--ckpt",
        str(ckpt_path),
        "--output",
        str(output_path),
        "--report-json",
        str(report_path),
    ]
    if overwrite:
        command.append("--overwrite")
    run_logged(command, log_path=log_path)
    return read_json(report_path)


def epoch_checkpoints(training_dir: Path, model_type: str) -> list[Path]:
    return sorted(training_dir.glob(f"{model_type}_epoch_*.pth"))


def rep16_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return dict((((summary.get("datasets") or {}).get("rep16") or {}).get("onnx")) or {})


def promotion_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "follow_score": float(metrics.get("follow_score") or 1e9) < PROMOTION_BASELINE["follow_score"],
        "x_mae": float(metrics.get("x_mae") or 1e9) <= PROMOTION_BASELINE["x_mae"],
        "size_mae": float(metrics.get("size_mae") or 1e9) <= PROMOTION_BASELINE["size_mae"],
        "no_person_fp_rate": float(metrics.get("no_person_fp_rate") or 1e9) <= PROMOTION_BASELINE["no_person_fp_rate"],
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "baseline_reference": dict(PROMOTION_BASELINE),
    }


def epoch_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = row["rep16_onnx"]
    return (
        float(metrics.get("follow_score") or 1e9),
        float(metrics.get("x_mae") or 1e9),
        float(metrics.get("size_mae") or 1e9),
        float(metrics.get("no_person_fp_rate") or 1e9),
        int(row["epoch"]),
    )


def build_summary_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]["rep16_onnx"]
    best_qat = summary["best_qat"]["rep16_onnx"]
    promotion = summary["best_qat"]["promotion_gate"]
    lines = [
        "# Plain Follow Focused QAT Study",
        "",
        f"- focused_qat_stem_mode: `{summary['artifacts']['qat_stem_mode']}`",
        "",
        "## Baseline PTQ",
        f"- follow_score: `{baseline.get('follow_score')}`",
        f"- x_mae: `{baseline.get('x_mae')}`",
        f"- size_mae: `{baseline.get('size_mae')}`",
        f"- no_person_fp_rate: `{baseline.get('no_person_fp_rate')}`",
        f"- activation_sensitivity: `{summary['baseline']['activation_sensitivity_report']}`",
        "",
        "## Best Focused QAT",
        f"- epoch: `{summary['best_qat'].get('epoch')}`",
        f"- follow_score: `{best_qat.get('follow_score')}`",
        f"- x_mae: `{best_qat.get('x_mae')}`",
        f"- size_mae: `{best_qat.get('size_mae')}`",
        f"- no_person_fp_rate: `{best_qat.get('no_person_fp_rate')}`",
        f"- promotion_gate_passed: `{promotion.get('passed')}`",
        f"- activation_sensitivity: `{summary['best_qat']['activation_sensitivity_report']}`",
        "",
        "## Epoch Table",
        "",
        "| Epoch | follow_score | x_mae | size_mae | no_person_fp_rate | filtered_ckpt |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.get("epoch_results") or []:
        metrics = row["rep16_onnx"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | `{}` |".format(
                row["epoch"],
                float(metrics.get("follow_score") or 0.0),
                float(metrics.get("x_mae") or 0.0),
                float(metrics.get("size_mae") or 0.0),
                float(metrics.get("no_person_fp_rate") or 0.0),
                row["filtered_eval_checkpoint"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- calibration_manifest: `{summary['artifacts']['calibration_manifest']}`",
            f"- train_sample_manifest: `{summary['artifacts']['train_sample_manifest']}`",
            f"- baseline_overlays: `{summary['baseline']['overlay_summary_path']}`",
            f"- best_qat_overlays: `{summary['best_qat']['overlay_summary_path']}`",
            f"- best_qat_eval_checkpoint: `{summary['artifacts']['best_qat_eval_checkpoint']}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    rep16_dir = Path(args.rep16_dir).expanduser().resolve()
    hard_case_dir = Path(args.hard_case_dir).expanduser().resolve()
    annotations = Path(args.annotations).expanduser().resolve()
    calib_image_dir = Path(args.calib_image_dir).expanduser().resolve()
    calib_annotations = Path(args.calib_annotations).expanduser().resolve()
    train_image_dir = Path(args.train_image_dir).expanduser().resolve()
    train_annotations = Path(args.train_annotations).expanduser().resolve()

    train_python = resolve_python(args.train_python, args.python)
    eval_python = resolve_python(args.eval_python, args.python)

    manifests_dir = output_dir / "manifests"
    logs_dir = output_dir / "logs"
    baseline_eval_dir = output_dir / "baseline_eval"
    baseline_overlay_dir = output_dir / "overlays" / "baseline"
    training_dir = output_dir / "training" / "focused_qat"
    filtered_dir = output_dir / "filtered_checkpoints"
    epoch_eval_root = output_dir / "qat_epoch_eval"
    best_overlay_dir = output_dir / "overlays" / "best_qat"

    calibration_manifest = (
        Path(args.calib_manifest).expanduser().resolve()
        if args.calib_manifest
        else manifests_dir / "calibration_manifest_val2017_top128.json"
    )
    if not args.calib_manifest:
        build_manifest_if_needed(
            python_bin=eval_python,
            output_path=calibration_manifest,
            image_dir=calib_image_dir,
            annotations=calib_annotations,
            follow_head_type=args.follow_head_type,
            model_type="plain_follow",
            target_count=int(args.calib_target_count),
            max_images=0,
            overwrite=True,
            log_path=logs_dir / "build_calibration_manifest.log",
        )

    train_sample_manifest = (
        Path(args.train_sample_manifest).expanduser().resolve()
        if args.train_sample_manifest
        else manifests_dir / "train_manifest_train2017_top2048.json"
    )
    if not args.train_sample_manifest:
        build_manifest_if_needed(
            python_bin=eval_python,
            output_path=train_sample_manifest,
            image_dir=train_image_dir,
            annotations=train_annotations,
            follow_head_type=args.follow_head_type,
            model_type="plain_follow",
            target_count=int(args.train_manifest_target_count),
            max_images=int(args.train_manifest_max_images),
            overwrite=True,
            log_path=logs_dir / "build_train_manifest.log",
        )

    baseline_summary = evaluate_checkpoint(
        python_bin=eval_python,
        ckpt_path=ckpt_path,
        output_dir=baseline_eval_dir,
        candidate_name="plain_follow_ptq_baseline",
        rep16_dir=rep16_dir,
        hard_case_dir=hard_case_dir,
        annotations=annotations,
        calib_manifest=calibration_manifest,
        calib_dir=calib_image_dir,
        vis_thresh=float(args.vis_thresh),
        overwrite=True,
        log_path=logs_dir / "baseline_eval.log",
    )
    baseline_overlay = build_overlay_summary(
        python_bin=eval_python,
        ckpt_path=ckpt_path,
        onnx_path=baseline_eval_dir / "model_id.onnx",
        output_dir=baseline_overlay_dir,
        rep16_dir=rep16_dir,
        annotations=annotations,
        vis_thresh=float(args.vis_thresh),
        overwrite=True,
        log_path=logs_dir / "baseline_overlay.log",
    )

    train_command = [
        train_python,
        str(PROJECT_DIR / "train.py"),
        "--data_root",
        str(train_image_dir),
        "--train_ann",
        str(train_annotations),
        "--val_root",
        str(calib_image_dir),
        "--val_ann",
        str(calib_annotations),
        "--model-type",
        "plain_follow",
        "--follow-head-type",
        args.follow_head_type,
        "--stem-mode",
        args.stem_mode,
        "--init-ckpt",
        str(ckpt_path),
        "--quant-aware-finetune",
        "--activation-range-regularization",
        "--trainable-module-prefixes",
        "stem.,stage1.,output_head.",
        "--epochs",
        str(int(args.epochs)),
        "--batch_size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--num_workers",
        str(int(args.num_workers)),
        "--qat-bits",
        str(int(args.qat_bits)),
        "--qat-calib-batches",
        str(int(args.qat_calib_batches)),
        "--train-sample-manifest",
        str(train_sample_manifest),
        "--output_dir",
        str(training_dir),
    ]
    run_logged(train_command, log_path=logs_dir / "focused_qat_train.log")

    epoch_results: list[dict[str, Any]] = []
    for epoch_ckpt in epoch_checkpoints(training_dir, "plain_follow"):
        epoch_token = epoch_ckpt.stem.rsplit("_", 1)[-1]
        filtered_ckpt = filtered_dir / f"{epoch_ckpt.stem}_eval.pth"
        filter_report = filter_qat_checkpoint(
            python_bin=eval_python,
            ckpt_path=epoch_ckpt,
            output_path=filtered_ckpt,
            report_path=filtered_dir / f"{epoch_ckpt.stem}_eval_filter_report.json",
            overwrite=True,
            log_path=logs_dir / f"{epoch_ckpt.stem}_filter.log",
        )
        eval_dir = epoch_eval_root / f"epoch_{epoch_token}"
        epoch_summary = evaluate_checkpoint(
            python_bin=eval_python,
            ckpt_path=filtered_ckpt,
            output_dir=eval_dir,
            candidate_name=f"plain_follow_focused_qat_epoch_{epoch_token}",
            rep16_dir=rep16_dir,
            hard_case_dir=hard_case_dir,
            annotations=annotations,
            calib_manifest=calibration_manifest,
            calib_dir=calib_image_dir,
            vis_thresh=float(args.vis_thresh),
            overwrite=True,
            log_path=logs_dir / f"{epoch_ckpt.stem}_eval.log",
        )
        epoch_results.append(
            {
                "epoch": int(epoch_token),
                "epoch_checkpoint": str(epoch_ckpt),
                "filtered_eval_checkpoint": str(filtered_ckpt),
                "filter_report": filter_report,
                "summary_path": str(eval_dir / "summary.json"),
                "activation_sensitivity_report": str(eval_dir / "activation_sensitivity_report.json"),
                "rep16_onnx": rep16_metrics(epoch_summary),
            }
        )

    if not epoch_results:
        raise RuntimeError(f"No epoch checkpoints were produced under {training_dir}")

    epoch_results.sort(key=epoch_selection_key)
    best_epoch = epoch_results[0]
    best_summary = read_json(Path(best_epoch["summary_path"]))
    best_overlay = build_overlay_summary(
        python_bin=eval_python,
        ckpt_path=Path(best_epoch["filtered_eval_checkpoint"]),
        onnx_path=Path(best_epoch["summary_path"]).parent / "model_id.onnx",
        output_dir=best_overlay_dir,
        rep16_dir=rep16_dir,
        annotations=annotations,
        vis_thresh=float(args.vis_thresh),
        overwrite=True,
        log_path=logs_dir / "best_qat_overlay.log",
    )

    best_eval_ckpt_copy = output_dir / "best_qat_eval_checkpoint.pth"
    shutil.copy2(Path(best_epoch["filtered_eval_checkpoint"]), best_eval_ckpt_copy)

    summary = {
        "checkpoint_path": str(ckpt_path),
        "train_python": train_python,
        "eval_python": eval_python,
        "artifacts": {
            "calibration_manifest": str(calibration_manifest),
            "train_sample_manifest": str(train_sample_manifest),
            "best_qat_eval_checkpoint": str(best_eval_ckpt_copy),
            "qat_stem_mode": args.stem_mode,
        },
        "baseline": {
            "summary_path": str(baseline_eval_dir / "summary.json"),
            "activation_sensitivity_report": str(baseline_eval_dir / "activation_sensitivity_report.json"),
            "overlay_summary_path": str(baseline_overlay_dir / "comparison_summary.json"),
            "rep16_onnx": rep16_metrics(baseline_summary),
        },
        "epoch_results": epoch_results,
        "best_qat": {
            "epoch": best_epoch["epoch"],
            "summary_path": best_epoch["summary_path"],
            "activation_sensitivity_report": best_epoch["activation_sensitivity_report"],
            "overlay_summary_path": str(best_overlay_dir / "comparison_summary.json"),
            "filtered_eval_checkpoint": best_epoch["filtered_eval_checkpoint"],
            "rep16_onnx": rep16_metrics(best_summary),
            "promotion_gate": promotion_gate(rep16_metrics(best_summary)),
        },
        "decision": {
            "improved_over_baseline": epoch_selection_key(best_epoch) < epoch_selection_key(
                {
                    "epoch": 0,
                    "rep16_onnx": rep16_metrics(baseline_summary),
                }
            ),
            "next_step_if_not_promoted": (
                "Pivot to model-architecture or output-contract changes if focused QAT does not clear the promotion gate."
            ),
        },
    }
    write_json(output_dir / "study_summary.json", summary)
    write_markdown(output_dir / "study_summary.md", build_summary_markdown(summary))


if __name__ == "__main__":
    main()

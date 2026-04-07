#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_WINDOWS_PYTHON = Path("/mnt/c/Python313/python.exe")
DEFAULT_CKPT = PROJECT_DIR / "training" / "plain_follow" / "plain_follow_best_follow_score.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "plain_follow_prod"
DEFAULT_VAL_IMAGE_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical plain_follow production wrapper: build the calibration manifest, "
            "materialize a larger validation pack, run float overlays, run quant export/eval, "
            "and write a single release summary."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-image-dir", default=str(DEFAULT_VAL_IMAGE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument(
        "--calib-target-count",
        type=int,
        default=128,
        help="Number of calibration images selected from COCO val.",
    )
    parser.add_argument(
        "--calib-max-images",
        type=int,
        default=0,
        help="Optional max-images cap passed to the calibration manifest builder.",
    )
    parser.add_argument(
        "--expanded-pack-extra-count",
        type=int,
        default=48,
        help="Additional COCO val images to add on top of the rep16 set for the expanded validation pack.",
    )
    parser.add_argument(
        "--expanded-pack-max-images",
        type=int,
        default=0,
        help="Optional max-images cap passed to the validation-pack ranking manifest builder.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["auto", "copy", "hardlink"],
        default="auto",
        help="How to materialize the local input-set copies under the output directory.",
    )
    parser.add_argument(
        "--vis-thresh",
        type=float,
        default=None,
        help="Visibility threshold. Defaults to the checkpoint's selected validation threshold.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used for the wrapped export/validation scripts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_python(user_value: str | None) -> str:
    if user_value:
        return user_value
    if sys.executable:
        return sys.executable
    if os.name == "nt" and DEFAULT_WINDOWS_PYTHON.is_file():
        return str(DEFAULT_WINDOWS_PYTHON)
    return "python3"


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_logged(command: list[str], *, log_path: Path, cwd: Path = PROJECT_DIR) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {format_command(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: see {log_path}")


def discover_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is not a dict: {path}")
    return payload


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint_payload(ckpt_path)
    model_type = str(payload.get("model_type") or "")
    if model_type != "plain_follow":
        raise ValueError(
            f"run_plain_follow_release.py only supports plain_follow checkpoints, got {model_type!r}."
        )

    head_type = str(payload.get("follow_head_type") or "xbin9_size_bucket4")
    selected_vis_thresh = ((payload.get("val_stats") or {}).get("selected_vis_threshold")) or 0.5
    vis_thresh = float(args.vis_thresh if args.vis_thresh is not None else selected_vis_thresh)

    return {
        "ckpt_path": ckpt_path,
        "payload": payload,
        "model_type": model_type,
        "follow_head_type": head_type,
        "vis_thresh": vis_thresh,
        "python": resolve_python(args.python),
        "output_dir": Path(args.output_dir).expanduser().resolve(),
        "val_image_dir": Path(args.val_image_dir).expanduser().resolve(),
        "annotations": Path(args.annotations).expanduser().resolve(),
        "rep16_dir": Path(args.rep16_dir).expanduser().resolve(),
        "hard_case_dir": Path(args.hard_case_dir).expanduser().resolve(),
    }


def materialize_file(src: Path, dst: Path, *, copy_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_input_set(
    source_paths: list[Path],
    output_dir: Path,
    *,
    copy_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    link_modes: list[str] = []
    for index, src in enumerate(source_paths, start=1):
        dst = output_dir / src.name
        link_mode = materialize_file(src, dst, copy_mode=copy_mode)
        link_modes.append(link_mode)
        rows.append(
            {
                "index": index,
                "source_path": str(src),
                "output_path": str(dst),
                "image_name": src.name,
                "materialization": link_mode,
            }
        )
    materialization_mode = "mixed" if len(set(link_modes)) > 1 else (link_modes[0] if link_modes else None)
    summary = {
        "output_dir": str(output_dir),
        "image_count": len(rows),
        "materialization_mode": materialization_mode,
        "rows": rows,
    }
    write_json(output_dir / "input_set_summary.json", summary)
    return summary


def build_expanded_pack(
    *,
    rep16_dir: Path,
    hard_case_dir: Path,
    manifest_payload: dict[str, Any],
    extra_count: int,
    output_root: Path,
    copy_mode: str,
) -> dict[str, Any]:
    rep16_paths = discover_images(rep16_dir)
    hard_case_paths = discover_images(hard_case_dir)
    if not rep16_paths:
        raise FileNotFoundError(f"No images found in rep16 dir: {rep16_dir}")
    if not hard_case_paths:
        raise FileNotFoundError(f"No images found in hard-case dir: {hard_case_dir}")

    used_names: set[str] = set()
    expanded_sources: list[Path] = []
    extra_sources: list[Path] = []

    def add_once(path: Path, *, count_as_extra: bool) -> None:
        if path.name in used_names:
            return
        used_names.add(path.name)
        expanded_sources.append(path)
        if count_as_extra:
            extra_sources.append(path)

    for path in rep16_paths:
        add_once(path, count_as_extra=False)
    for path in hard_case_paths:
        add_once(path, count_as_extra=False)

    for row in manifest_payload.get("ordered_samples") or []:
        if len(extra_sources) >= int(extra_count):
            break
        source_path = Path(str(row.get("source_path") or "")).expanduser().resolve()
        if not source_path.is_file():
            continue
        add_once(source_path, count_as_extra=True)

    input_sets_dir = output_root / "input_sets"
    rep16_local = materialize_input_set(rep16_paths, input_sets_dir / "rep16", copy_mode=copy_mode)
    hard_case_local = materialize_input_set(hard_case_paths, input_sets_dir / "hard_case_subset", copy_mode=copy_mode)
    expanded_local = materialize_input_set(expanded_sources, input_sets_dir / "expanded_eval", copy_mode=copy_mode)

    summary = {
        "rep16": rep16_local,
        "hard_case": hard_case_local,
        "expanded_eval": expanded_local,
        "rep16_source_dir": str(rep16_dir),
        "hard_case_source_dir": str(hard_case_dir),
        "expanded_extra_count": len(extra_sources),
        "expanded_extra_sources": [str(path) for path in extra_sources],
    }
    write_json(output_root / "expanded_eval_pack_summary.json", summary)
    return summary


def build_release_summary_markdown(summary: dict[str, Any]) -> str:
    checkpoint_metrics = summary["metrics"]["checkpoint_val"]
    float_metrics = summary["metrics"]["float_validation"]
    quant_metrics = summary["metrics"]["quant_validation"]
    compare_metrics = summary["metrics"]["pre_to_post_compare"]
    expanded_label = str(summary["expanded_dataset_label"])
    hard_case_label = str(summary["hard_case_dataset_label"])

    lines = [
        "# plain_follow production validation",
        "",
        "## Contract",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- model_type: `{summary['model_type']}`",
        f"- follow_head_type: `{summary['follow_head_type']}`",
        f"- vis_thresh: `{summary['vis_thresh']}`",
        "",
        "## Checkpoint Validation",
        f"- follow_score: `{checkpoint_metrics.get('follow_score')}`",
        f"- x_mae: `{checkpoint_metrics.get('x_mae')}`",
        f"- size_mae: `{checkpoint_metrics.get('size_mae')}`",
        f"- recall: `{checkpoint_metrics.get('recall')}`",
        f"- no_person_fp_rate: `{checkpoint_metrics.get('no_person_fp_rate')}`",
        "",
        "## Float Overlay Validation",
        f"- rep16 follow_score: `{(float_metrics.get('rep16') or {}).get('follow_score')}`",
        f"- hard_case follow_score: `{(float_metrics.get('hard_case') or {}).get('follow_score')}`",
        f"- {expanded_label} follow_score: `{(float_metrics.get(expanded_label) or {}).get('follow_score')}`",
        f"- {expanded_label} recall: `{(float_metrics.get(expanded_label) or {}).get('recall')}`",
        f"- {expanded_label} no_person_fp_rate: `{(float_metrics.get(expanded_label) or {}).get('no_person_fp_rate')}`",
        "",
        "## Quant Validation",
        f"- {expanded_label} onnx follow_score: `{(quant_metrics.get(expanded_label) or {}).get('follow_score')}`",
        f"- {expanded_label} onnx x_mae: `{(quant_metrics.get(expanded_label) or {}).get('x_mae')}`",
        f"- {expanded_label} onnx size_mae: `{(quant_metrics.get(expanded_label) or {}).get('size_mae')}`",
        f"- {expanded_label} onnx no_person_fp_rate: `{(quant_metrics.get(expanded_label) or {}).get('no_person_fp_rate')}`",
        f"- {hard_case_label} onnx follow_score: `{(quant_metrics.get(hard_case_label) or {}).get('follow_score')}`",
        f"- visibility_gate_agreement: `{(quant_metrics.get('float_to_onnx_bin_preservation') or {}).get('visibility_gate_agreement')}`",
        f"- x_bin_exact_match_rate: `{(quant_metrics.get('float_to_onnx_bin_preservation') or {}).get('x_bin_exact_match_rate')}`",
        f"- size_bucket_exact_match_rate: `{(quant_metrics.get('float_to_onnx_bin_preservation') or {}).get('size_bucket_exact_match_rate')}`",
        "",
        "## Visual Drift Checks",
        f"- rep16 visibility_gate_agreement: `{(compare_metrics.get('rep16') or {}).get('visibility_gate_agreement')}`",
        f"- hard_case visibility_gate_agreement: `{(compare_metrics.get('hard_case') or {}).get('visibility_gate_agreement')}`",
        "",
        "## Artifacts",
        f"- calibration_manifest: `{summary['artifacts']['calibration_manifest']}`",
        f"- validation_pack_manifest: `{summary['artifacts']['expanded_pack_manifest']}`",
        f"- expanded_eval_pack: `{summary['artifacts']['expanded_eval_dir']}`",
        f"- quant_summary: `{summary['artifacts']['quant_summary']}`",
        f"- release_summary_json: `{summary['artifacts']['release_summary_json']}`",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    ensure_output_dir(context["output_dir"], overwrite=bool(args.overwrite))

    python_bin = str(context["python"])
    ckpt_path = Path(context["ckpt_path"])
    follow_head_type = str(context["follow_head_type"])
    vis_thresh = float(context["vis_thresh"])
    output_dir = Path(context["output_dir"])

    command_dir = output_dir / "commands"
    calibration_manifest_path = output_dir / "calibration_manifest_128.json"
    expanded_pack_manifest_path = output_dir / "expanded_eval_manifest.json"

    build_calibration_manifest_cmd = [
        python_bin,
        str(SCRIPT_DIR / "build_follow_calibration_manifest.py"),
        "--image-dir",
        str(context["val_image_dir"]),
        "--annotations",
        str(context["annotations"]),
        "--model-type",
        "plain_follow",
        "--follow-head-type",
        follow_head_type,
        "--target-count",
        str(int(args.calib_target_count)),
        "--output",
        str(calibration_manifest_path),
        "--overwrite",
    ]
    if int(args.calib_max_images) > 0:
        build_calibration_manifest_cmd.extend(["--max-images", str(int(args.calib_max_images))])
    run_logged(
        build_calibration_manifest_cmd,
        log_path=command_dir / "01_build_calibration_manifest.log",
    )

    build_expanded_pack_manifest_cmd = [
        python_bin,
        str(SCRIPT_DIR / "build_follow_calibration_manifest.py"),
        "--image-dir",
        str(context["val_image_dir"]),
        "--annotations",
        str(context["annotations"]),
        "--model-type",
        "plain_follow",
        "--follow-head-type",
        follow_head_type,
        "--target-count",
        str(max(int(args.expanded_pack_extra_count), 1)),
        "--output",
        str(expanded_pack_manifest_path),
        "--overwrite",
    ]
    if int(args.expanded_pack_max_images) > 0:
        build_expanded_pack_manifest_cmd.extend(["--max-images", str(int(args.expanded_pack_max_images))])
    run_logged(
        build_expanded_pack_manifest_cmd,
        log_path=command_dir / "02_build_expanded_eval_manifest.log",
    )

    expanded_pack = build_expanded_pack(
        rep16_dir=Path(context["rep16_dir"]),
        hard_case_dir=Path(context["hard_case_dir"]),
        manifest_payload=read_json(expanded_pack_manifest_path),
        extra_count=int(args.expanded_pack_extra_count),
        output_root=output_dir,
        copy_mode=str(args.copy_mode),
    )

    rep16_local_dir = Path((expanded_pack["rep16"] or {})["output_dir"])
    hard_case_local_dir = Path((expanded_pack["hard_case"] or {})["output_dir"])
    expanded_eval_local_dir = Path((expanded_pack["expanded_eval"] or {})["output_dir"])
    expanded_dataset_label = "expanded_eval"
    hard_case_dataset_label = "hard_case"

    float_jobs = [
        ("rep16", rep16_local_dir, output_dir / "float_rep16", "03_float_rep16.log"),
        ("hard_case", hard_case_local_dir, output_dir / "float_hard_case", "04_float_hard_case.log"),
        (
            expanded_dataset_label,
            expanded_eval_local_dir,
            output_dir / f"float_{expanded_dataset_label}",
            "05_float_expanded_eval.log",
        ),
    ]
    float_summaries: dict[str, dict[str, Any]] = {}
    for dataset_label, images_dir, dataset_output_dir, log_name in float_jobs:
        command = [
            python_bin,
            str(SCRIPT_DIR / "validate_follow_rep16_overlays.py"),
            "--ckpt",
            str(ckpt_path),
            "--output-dir",
            str(dataset_output_dir),
            "--images-dir",
            str(images_dir),
            "--annotations",
            str(context["annotations"]),
            "--dataset-label",
            dataset_label,
            "--vis-thresh",
            str(vis_thresh),
            "--overwrite",
        ]
        run_logged(command, log_path=command_dir / log_name)
        float_summaries[dataset_label] = read_json(dataset_output_dir / "summary.json")

    quant_output_dir = output_dir / "quant_eval"
    quant_command = [
        python_bin,
        str(SCRIPT_DIR / "evaluate_quant_native_follow.py"),
        "--ckpt",
        str(ckpt_path),
        "--output-dir",
        str(quant_output_dir),
        "--rep16-dir",
        str(expanded_eval_local_dir),
        "--hard-case-dir",
        str(hard_case_local_dir),
        "--primary-dataset-label",
        expanded_dataset_label,
        "--secondary-dataset-label",
        hard_case_dataset_label,
        "--annotations",
        str(context["annotations"]),
        "--calib-dir",
        str(context["val_image_dir"]),
        "--calib-manifest",
        str(calibration_manifest_path),
        "--vis-thresh",
        str(vis_thresh),
        "--overwrite",
    ]
    run_logged(quant_command, log_path=command_dir / "06_quant_eval.log")
    quant_summary = read_json(quant_output_dir / "summary.json")

    compare_jobs = [
        ("rep16", rep16_local_dir, output_dir / "compare_rep16", "07_compare_rep16.log"),
        ("hard_case", hard_case_local_dir, output_dir / "compare_hard_case", "08_compare_hard_case.log"),
    ]
    compare_summaries: dict[str, dict[str, Any]] = {}
    quant_onnx_path = Path((quant_summary.get("artifacts") or {}).get("onnx") or (quant_output_dir / "model_id.onnx"))
    for dataset_label, images_dir, dataset_output_dir, log_name in compare_jobs:
        command = [
            python_bin,
            str(SCRIPT_DIR / "compare_quant_native_follow_rep16_overlays.py"),
            "--ckpt",
            str(ckpt_path),
            "--onnx",
            str(quant_onnx_path),
            "--output-dir",
            str(dataset_output_dir),
            "--images-dir",
            str(images_dir),
            "--annotations",
            str(context["annotations"]),
            "--dataset-label",
            dataset_label,
            "--vis-thresh",
            str(vis_thresh),
            "--overwrite",
        ]
        run_logged(command, log_path=command_dir / log_name)
        compare_summaries[dataset_label] = read_json(dataset_output_dir / "comparison_summary.json")

    release_summary = {
        "checkpoint_path": str(ckpt_path),
        "model_type": "plain_follow",
        "follow_head_type": follow_head_type,
        "vis_thresh": vis_thresh,
        "expanded_dataset_label": expanded_dataset_label,
        "hard_case_dataset_label": hard_case_dataset_label,
        "artifacts": {
            "calibration_manifest": str(calibration_manifest_path),
            "expanded_pack_manifest": str(expanded_pack_manifest_path),
            "expanded_eval_pack_summary": str(output_dir / "expanded_eval_pack_summary.json"),
            "rep16_dir": str(rep16_local_dir),
            "hard_case_dir": str(hard_case_local_dir),
            "expanded_eval_dir": str(expanded_eval_local_dir),
            "float_rep16_summary": str(output_dir / "float_rep16" / "summary.json"),
            "float_hard_case_summary": str(output_dir / "float_hard_case" / "summary.json"),
            f"float_{expanded_dataset_label}_summary": str(
                output_dir / f"float_{expanded_dataset_label}" / "summary.json"
            ),
            "quant_summary": str(quant_output_dir / "summary.json"),
            "quant_summary_md": str(quant_output_dir / "summary.md"),
            "compare_rep16_summary": str(output_dir / "compare_rep16" / "comparison_summary.json"),
            "compare_hard_case_summary": str(output_dir / "compare_hard_case" / "comparison_summary.json"),
        },
        "metrics": {
            "checkpoint_val": dict((context["payload"].get("val_stats") or {})),
            "float_validation": {
                dataset_label: dict(summary.get("metrics") or {})
                for dataset_label, summary in float_summaries.items()
            },
            "quant_validation": {
                expanded_dataset_label: dict(
                    (((quant_summary.get("datasets") or {}).get(expanded_dataset_label) or {}).get("onnx") or {})
                ),
                hard_case_dataset_label: dict(
                    (((quant_summary.get("datasets") or {}).get(hard_case_dataset_label) or {}).get("onnx") or {})
                ),
                "float_to_onnx_bin_preservation": dict(
                    (quant_summary.get("quant_fidelity") or {}).get("float_to_onnx_bin_preservation") or {}
                ),
            },
            "pre_to_post_compare": {
                dataset_label: dict(((summary.get("metrics") or {}).get("pre_to_post") or {}))
                for dataset_label, summary in compare_summaries.items()
            },
        },
    }
    release_summary_path = output_dir / "release_summary.json"
    release_summary["artifacts"]["release_summary_json"] = str(release_summary_path)
    write_json(release_summary_path, release_summary)
    write_markdown(output_dir / "release_summary.md", build_release_summary_markdown(release_summary))


if __name__ == "__main__":
    main()

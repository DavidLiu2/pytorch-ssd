#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    (parent for parent in SCRIPT_DIR.parents if (parent / "models").is_dir() and (parent / "export").is_dir()),
    SCRIPT_DIR.parent,
)
EXPORT_DIR = PROJECT_DIR / "export"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from validate_follow_rep16_overlays import build_contact_sheet  # noqa: E402


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
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "plain_follow_quant_val" / "qd_id_operator_sweep"
DEFAULT_FACTORS = (0.5, 0.7071067811865476, 1.4142135623730951, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused plain_follow QD->ID study: operator taps, explicit ID eps_dict, "
            "local scale sweeps at the first bad control module, rep16 validation, and comparison overlays."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--calib-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--local-factors", default="0.5,0.7071067811865476,1.4142135623730951,2.0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def parse_factors(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or list(DEFAULT_FACTORS)


def run_trial(
    *,
    python_bin: str,
    eval_script: Path,
    args: argparse.Namespace,
    trial_dir: Path,
    trial_name: str,
    extra_args: list[str],
) -> dict[str, Any]:
    command = [
        python_bin,
        str(eval_script),
        "--ckpt",
        str(Path(args.ckpt).expanduser().resolve()),
        "--output-dir",
        str(trial_dir),
        "--rep16-dir",
        str(Path(args.rep16_dir).expanduser().resolve()),
        "--hard-case-dir",
        str(Path(args.hard_case_dir).expanduser().resolve()),
        "--annotations",
        str(Path(args.annotations).expanduser().resolve()),
        "--calib-dir",
        str(Path(args.calib_dir).expanduser().resolve()),
        "--bits",
        str(int(args.bits)),
        "--eps-in",
        str(float(args.eps_in)),
        "--calib-batches",
        str(int(args.calib_batches)),
        "--calib-seed",
        str(int(args.calib_seed)),
        "--vis-thresh",
        str(float(args.vis_thresh)),
        "--opset-version",
        str(int(args.opset_version)),
        "--candidate-name",
        trial_name,
        "--overwrite",
        *extra_args,
    ]
    subprocess.run(command, cwd=str(PROJECT_DIR), check=True)
    summary_path = trial_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary after trial `{trial_name}`: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["trial_name"] = trial_name
    summary["trial_dir"] = str(trial_dir)
    return summary


def overlay_summary_path(output_dir: Path) -> Path:
    return output_dir / "comparison_summary.json"


def run_overlay_compare(
    *,
    python_bin: str,
    compare_script: Path,
    ckpt_path: Path,
    onnx_path: Path,
    output_dir: Path,
    rep16_dir: Path,
    annotations: Path,
    vis_thresh: float,
) -> dict[str, Any]:
    command = [
        python_bin,
        str(compare_script),
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
        "--overwrite",
    ]
    subprocess.run(command, cwd=str(PROJECT_DIR), check=True)
    summary_path = overlay_summary_path(output_dir)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def qd_id_semantic_score(summary: dict[str, Any]) -> float:
    semantic = (((summary.get("quant_fidelity") or {}).get("boundaries") or {}).get("qd_to_id") or {}).get("semantic") or {}
    return float(semantic.get("x_value_mae") or 0.0) + float(semantic.get("size_value_mae") or 0.0) + (
        1.0 - float(semantic.get("visibility_gate_agreement") or 0.0)
    )


def rep16_follow_score(summary: dict[str, Any]) -> float:
    return float((((summary.get("datasets") or {}).get("rep16") or {}).get("onnx") or {}).get("follow_score") or 0.0)


def qd_id_local_score(summary: dict[str, Any]) -> float:
    report = ((summary.get("quant_fidelity") or {}).get("qd_to_id_operator_report") or {}).get("first_bad_operator") or {}
    return float(report.get("output_mean_abs_diff_mean") or 0.0)


def summary_row(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    rep16 = ((summary.get("datasets") or {}).get("rep16") or {}).get("onnx") or {}
    qd_id = (((summary.get("quant_fidelity") or {}).get("boundaries") or {}).get("qd_to_id") or {}).get("semantic") or {}
    operator = ((summary.get("quant_fidelity") or {}).get("qd_to_id_operator_report") or {}).get("first_bad_operator") or {}
    baseline_rep16 = rep16_follow_score(baseline)
    return {
        "trial_name": summary.get("trial_name"),
        "trial_dir": summary.get("trial_dir"),
        "id_stage_config": summary.get("id_stage_config"),
        "rep16_follow_score": rep16_follow_score(summary),
        "rep16_x_mae": float(rep16.get("x_mae") or 0.0),
        "rep16_size_mae": float(rep16.get("size_mae") or 0.0),
        "rep16_no_person_fp_rate": float(rep16.get("no_person_fp_rate") or 0.0),
        "rep16_follow_score_delta_vs_current": rep16_follow_score(summary) - baseline_rep16,
        "qd_id_semantic_score": qd_id_semantic_score(summary),
        "qd_id_x_value_mae": float(qd_id.get("x_value_mae") or 0.0),
        "qd_id_size_value_mae": float(qd_id.get("size_value_mae") or 0.0),
        "qd_id_visibility_gate_agreement": float(qd_id.get("visibility_gate_agreement") or 0.0),
        "qd_id_first_bad_operator": operator.get("module_name"),
        "qd_id_scale_control_module": operator.get("scale_control_module"),
        "qd_id_operator_output_mean_abs_diff": qd_id_local_score(summary),
        "onnx_path": (((summary.get("artifacts") or {}).get("onnx"))),
    }


def select_overlay_partner(rows: list[dict[str, Any]], baseline_name: str) -> dict[str, Any]:
    non_baseline = [row for row in rows if row["trial_name"] != baseline_name]
    improved = [row for row in non_baseline if row["rep16_follow_score"] < rows[0]["rep16_follow_score"] - 1e-9]
    if improved:
        return min(
            improved,
            key=lambda row: (
                row["rep16_follow_score"],
                row["qd_id_semantic_score"],
                row["qd_id_operator_output_mean_abs_diff"],
            ),
        )
    if non_baseline:
        return min(
            non_baseline,
            key=lambda row: (
                row["qd_id_semantic_score"],
                row["rep16_follow_score"],
                row["qd_id_operator_output_mean_abs_diff"],
            ),
        )
    return rows[0]


def draw_header(
    width: int,
    *,
    image_name: str,
    left_label: str,
    right_label: str,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
) -> Image.Image:
    header = Image.new("RGB", (width, 70), color=(245, 245, 245))
    draw = ImageDraw.Draw(header)
    font = ImageFont.load_default()
    left_text = (
        f"{left_label}: x_err={left_row.get('post_x_error')} "
        f"size_err={left_row.get('post_size_error')} vis={left_row.get('post_visible')}"
    )
    right_text = (
        f"{right_label}: x_err={right_row.get('post_x_error')} "
        f"size_err={right_row.get('post_size_error')} vis={right_row.get('post_visible')}"
    )
    draw.text((12, 8), image_name, fill=(20, 20, 20), font=font)
    draw.text((12, 28), left_text, fill=(90, 40, 40), font=font)
    draw.text((12, 46), right_text, fill=(32, 74, 120), font=font)
    return header


def build_comparison_images(
    *,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_label: str,
    right_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    left_rows = {row["image_name"]: row for row in left_summary.get("rows") or []}
    right_rows = {row["image_name"]: row for row in right_summary.get("rows") or []}
    paired_paths = []
    image_rows = []
    for image_name in sorted(set(left_rows) & set(right_rows)):
        left_row = left_rows[image_name]
        right_row = right_rows[image_name]
        left_overlay = Image.open(left_row["overlay_path"]).convert("RGB")
        right_overlay = Image.open(right_row["overlay_path"]).convert("RGB")
        header = draw_header(
            left_overlay.width + right_overlay.width + 24,
            image_name=image_name,
            left_label=left_label,
            right_label=right_label,
            left_row=left_row,
            right_row=right_row,
        )
        canvas = Image.new(
            "RGB",
            (left_overlay.width + right_overlay.width + 24, max(left_overlay.height, right_overlay.height) + header.height),
            color=(255, 255, 255),
        )
        canvas.paste(header, (0, 0))
        canvas.paste(left_overlay, (0, header.height))
        canvas.paste(right_overlay, (left_overlay.width + 24, header.height))
        paired_path = output_dir / f"{sanitize_name(Path(image_name).stem)}_paired.png"
        canvas.save(paired_path)
        paired_paths.append(paired_path)
        image_rows.append(
            {
                "image_name": image_name,
                "paired_overlay_path": str(paired_path),
                "left_post_x_error": left_row.get("post_x_error"),
                "right_post_x_error": right_row.get("post_x_error"),
                "left_post_size_error": left_row.get("post_size_error"),
                "right_post_size_error": right_row.get("post_size_error"),
            }
        )
        left_overlay.close()
        right_overlay.close()
    contact_sheet = output_dir / "paired_contact_sheet.png"
    if paired_paths:
        build_contact_sheet(paired_paths, contact_sheet)
    return {
        "left_label": left_label,
        "right_label": right_label,
        "pair_count": len(paired_paths),
        "paired_contact_sheet": str(contact_sheet),
        "rows": image_rows,
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Plain Follow QD -> ID Study",
        "",
        f"- Baseline trial: `{summary['baseline_trial']}`",
        f"- Explicit eps trial: `{summary['explicit_eps_trial']}`",
        f"- Target bad operator: `{summary['target_bad_operator']}`",
        f"- Target control module: `{summary['target_control_module']}`",
        f"- Best rep16 trial: `{summary['winner_by_rep16']}`",
        f"- Best local qd->id trial: `{summary['winner_by_local_qd_id']}`",
        "",
        "## Trial Table",
        "",
        "| Trial | Rep16 follow_score | Delta vs current | qd->id semantic score | First bad operator | Control module | Local output mean abs |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in summary.get("trial_rows") or []:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {} | {} | {:.6f} |".format(
                row["trial_name"],
                float(row["rep16_follow_score"]),
                float(row["rep16_follow_score_delta_vs_current"]),
                float(row["qd_id_semantic_score"]),
                row.get("qd_id_first_bad_operator"),
                row.get("qd_id_scale_control_module"),
                float(row.get("qd_id_operator_output_mean_abs_diff") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Current overlays: `{summary['artifacts']['current_overlay_dir']}`",
            f"- Comparison overlays: `{summary['artifacts']['comparison_overlay_dir']}`",
            f"- Paired comparison images: `{summary['artifacts']['paired_comparison_dir']}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    python_bin = sys.executable
    eval_script = EXPORT_DIR / "evaluate_quant_native_follow.py"
    compare_script = EXPORT_DIR / "compare_quant_native_follow_rep16_overlays.py"
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    rep16_dir = Path(args.rep16_dir).expanduser().resolve()
    annotations = Path(args.annotations).expanduser().resolve()

    trials_dir = output_dir / "trials"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    comparison_images_dir = output_dir / "comparison_images"

    current_summary = run_trial(
        python_bin=python_bin,
        eval_script=eval_script,
        args=args,
        trial_dir=trials_dir / "current",
        trial_name="current",
        extra_args=[],
    )
    explicit_summary = run_trial(
        python_bin=python_bin,
        eval_script=eval_script,
        args=args,
        trial_dir=trials_dir / "explicit_eps_dict",
        trial_name="explicit_eps_dict",
        extra_args=["--id-explicit-eps-dict"],
    )

    baseline_bad = (((current_summary.get("quant_fidelity") or {}).get("qd_to_id_operator_report") or {}).get("first_bad_operator") or {})
    explicit_bad = (((explicit_summary.get("quant_fidelity") or {}).get("qd_to_id_operator_report") or {}).get("first_bad_operator") or {})
    target_bad = explicit_bad or baseline_bad
    target_bad_operator = target_bad.get("module_name")
    target_control_module = target_bad.get("scale_control_module")
    if not target_control_module:
        raise RuntimeError("Could not determine a scale-control module for the first bad QD->ID operator.")

    trial_summaries = [current_summary, explicit_summary]
    for factor in parse_factors(args.local_factors):
        trial_name = f"{sanitize_name(target_control_module)}_x{factor:g}"
        trial_summaries.append(
            run_trial(
                python_bin=python_bin,
                eval_script=eval_script,
                args=args,
                trial_dir=trials_dir / trial_name,
                trial_name=trial_name,
                extra_args=[
                    "--id-explicit-eps-dict",
                    "--id-local-scale-module",
                    str(target_control_module),
                    "--id-local-scale-factor",
                    str(float(factor)),
                ],
            )
        )

    rows = [summary_row(summary, current_summary) for summary in trial_summaries]
    rows.sort(
        key=lambda row: (
            0 if row["trial_name"] == "current" else 1,
            row["rep16_follow_score"],
            row["qd_id_semantic_score"],
            row["qd_id_operator_output_mean_abs_diff"],
        )
    )
    by_name = {summary["trial_name"]: summary for summary in trial_summaries}
    winner_by_rep16 = min(rows, key=lambda row: (row["rep16_follow_score"], row["qd_id_semantic_score"]))
    winner_by_local = min(rows, key=lambda row: (row["qd_id_semantic_score"], row["qd_id_operator_output_mean_abs_diff"], row["rep16_follow_score"]))
    overlay_partner = select_overlay_partner(rows, baseline_name="current")

    current_overlay = run_overlay_compare(
        python_bin=python_bin,
        compare_script=compare_script,
        ckpt_path=ckpt_path,
        onnx_path=Path(rows[0]["onnx_path"]).expanduser().resolve(),
        output_dir=overlays_dir / "current",
        rep16_dir=rep16_dir,
        annotations=annotations,
        vis_thresh=float(args.vis_thresh),
    )
    partner_overlay = run_overlay_compare(
        python_bin=python_bin,
        compare_script=compare_script,
        ckpt_path=ckpt_path,
        onnx_path=Path(overlay_partner["onnx_path"]).expanduser().resolve(),
        output_dir=overlays_dir / sanitize_name(overlay_partner["trial_name"]),
        rep16_dir=rep16_dir,
        annotations=annotations,
        vis_thresh=float(args.vis_thresh),
    )
    paired = build_comparison_images(
        left_summary=current_overlay,
        right_summary=partner_overlay,
        left_label="current",
        right_label=str(overlay_partner["trial_name"]),
        output_dir=comparison_images_dir,
    )

    summary = {
        "baseline_trial": "current",
        "explicit_eps_trial": "explicit_eps_dict",
        "target_bad_operator": target_bad_operator,
        "target_control_module": target_control_module,
        "winner_by_rep16": winner_by_rep16["trial_name"],
        "winner_by_local_qd_id": winner_by_local["trial_name"],
        "trial_rows": rows,
        "artifacts": {
            "current_overlay_dir": str((overlays_dir / "current").resolve()),
            "comparison_overlay_dir": str((overlays_dir / sanitize_name(overlay_partner["trial_name"])).resolve()),
            "paired_comparison_dir": str(comparison_images_dir.resolve()),
            "paired_contact_sheet": paired.get("paired_contact_sheet"),
        },
        "paired_comparison": paired,
    }
    write_json(output_dir / "study_summary.json", summary)
    write_markdown(output_dir / "study_summary.md", build_summary_markdown(summary))


if __name__ == "__main__":
    main()

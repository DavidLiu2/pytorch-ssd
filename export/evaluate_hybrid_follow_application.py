#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from statistics import mean, median
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from inference_follow_demo import load_checkpoint as demo_load_checkpoint  # noqa: E402
from inference_follow_demo import preprocess_image as demo_preprocess_image  # noqa: E402
from models.hybrid_follow_net import HybridFollowNet  # noqa: E402
from visualize_hybrid_follow_prediction import (  # noqa: E402
    BG_COLOR,
    CROP_COLOR,
    DEFAULT_ANN,
    GT_COLOR,
    MODEL_INPUT_SIZE,
    PANEL_PADDING,
    RESAMPLE_BILINEAR,
    RESAMPLE_NEAREST,
    TEXT_COLOR,
    _compute_crop_geometry,
    _crop_box_to_square,
    _draw_box,
    _draw_vertical_line,
    _extract_image_id,
    _fit_text,
    _gt_follow_target,
    _load_largest_person_box,
)

CHECKPOINT_COLOR = (220, 53, 69)
APPLICATION_COLOR = (46, 117, 182)
CENTER_COLOR = (120, 120, 120)
OVERLAY_HEADER_HEIGHT = 220
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current generated hybrid_follow application against the "
            "pre-export checkpoint on the same images, then write overlays and "
            "aggregate logs under logs/hybrid_follow_val/."
        )
    )
    parser.add_argument(
        "--images-dir",
        default=(
            "logs/hybrid_follow_val/1_real_image_validation/"
            "input_sets/representative16_20260324"
        ),
        help="Image directory to evaluate. Discovered recursively.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "logs/hybrid_follow_val/application_vs_checkpoint_<timestamp>."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default="training/hybrid_follow/hybrid_follow_best_follow_score.pth",
        help="Checkpoint used for the pre-export inference demo path.",
    )
    parser.add_argument(
        "--onnx",
        default="export/hybrid_follow/hybrid_follow_dory.onnx",
        help="Exported ONNX used by the stage-drift comparison.",
    )
    parser.add_argument(
        "--app-dir",
        default="application",
        help="Generated application directory relative to pytorch_ssd.",
    )
    parser.add_argument(
        "--run-script",
        default="tools/run_real_image_val_impl.sh",
        help="Real-image validation shell script used to generate application outputs.",
    )
    parser.add_argument(
        "--platform",
        default="gvsoc",
        help="Platform forwarded to run_real_image_val.sh.",
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANN),
        help="Optional COCO annotations json for GT overlays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the results directory first if it already exists.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used for the local stage-drift runs.",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None

    raw = str(path_value)
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        translated_rest = raw[7:].replace("/", "\\")
        translated = Path(f"{drive}:\\{translated_rest}")
        if translated.exists():
            return translated.resolve()

    relative = (PROJECT_DIR / candidate).resolve()
    if relative.exists():
        return relative
    return candidate


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    raw = str(resolved)
    if raw.startswith("/mnt/"):
        return raw
    pure = PureWindowsPath(resolved)
    drive = pure.drive.rstrip(":").lower()
    tail = "/".join(pure.parts[1:])
    return f"/mnt/{drive}/{tail}"


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_images(images_dir: Path) -> list[Path]:
    images = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sanitize_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name).strip("._")
    return cleaned or "image"


def clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def resolve_stage_drift_python(requested: str | Path | None) -> Path:
    candidates: list[Path] = []
    if requested:
        requested_path = Path(str(requested)).expanduser()
        if not requested_path.is_absolute():
            requested_path = (PROJECT_DIR / requested_path)
        candidates.append(requested_path)
        resolved = resolve_repo_path(requested)
        if resolved is not None and resolved != requested_path:
            candidates.append(resolved)
    candidates.extend(
        [
            (PROJECT_DIR.parent / "nemoenv" / "bin" / "python3"),
            (PROJECT_DIR.parent / "nemoenv" / "Scripts" / "python.exe"),
            Path(sys.executable),
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.absolute()
        except OSError:
            continue
    raise FileNotFoundError(f"Python interpreter not found: {requested}")


def load_demo_model(ckpt_path: Path) -> tuple[HybridFollowNet, torch.device]:
    device = torch.device("cpu")
    model = HybridFollowNet(input_channels=1, image_size=(128, 128)).to(device)
    demo_load_checkpoint(model, ckpt_path, device)
    model.eval()
    return model, device


def run_demo_inference(
    model: HybridFollowNet,
    device: torch.device,
    image_path: Path,
    vis_threshold: float = 0.5,
) -> dict[str, Any]:
    x = demo_preprocess_image(image_path).to(device)
    with torch.no_grad():
        raw = model(x)[0]

    raw_x_offset = float(raw[0].detach().cpu().item())
    raw_size_proxy = float(raw[1].detach().cpu().item())
    visibility_logit = float(raw[2].detach().cpu().item())
    visibility_confidence = float(torch.sigmoid(raw[2]).detach().cpu().item())

    return {
        "raw_x_offset": raw_x_offset,
        "raw_size_proxy": raw_size_proxy,
        "raw_visibility_logit": visibility_logit,
        "x_offset": clamp_unit(raw_x_offset),
        "size_proxy": max(0.0, min(1.0, raw_size_proxy)),
        "visibility_confidence": visibility_confidence,
        "target_visible": int(visibility_confidence >= vis_threshold),
        "preprocess_source": "inference_follow_demo.py",
    }


def compact_stage(stage_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(stage_payload, dict):
        return None

    decoded = stage_payload.get("decoded") or {}
    raw_semantic = stage_payload.get("raw_semantic") or []
    return {
        "status": stage_payload.get("status"),
        "label": stage_payload.get("label"),
        "stage_tag": stage_payload.get("stage_tag"),
        "representation": stage_payload.get("representation"),
        "x_offset_raw": decoded.get("x_offset"),
        "x_offset_clamped": None if "x_offset" not in decoded else clamp_unit(decoded["x_offset"]),
        "size_proxy": decoded.get("size_proxy"),
        "visibility_confidence": decoded.get("visibility_confidence"),
        "visibility_logit": decoded.get("visibility_logit"),
        "raw_semantic": raw_semantic,
        "raw_native": stage_payload.get("raw_native"),
        "error": stage_payload.get("error"),
    }


def run_stage_drift(
    python_exe: Path,
    image_path: Path,
    ckpt_path: Path,
    onnx_path: Path,
    golden_path: Path,
    gvsoc_json_path: Path,
    output_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    cmd = [
        str(python_exe),
        str((PROJECT_DIR / "export" / "compare_hybrid_follow_stages.py").resolve()),
        "--image",
        str(image_path),
        "--ckpt",
        str(ckpt_path),
        "--onnx",
        str(onnx_path),
        "--golden",
        str(golden_path),
        "--gvsoc-json",
        str(gvsoc_json_path),
        "--output-dir",
        str(output_dir),
        "--overwrite",
        "--nemo-stage",
        "skip",
    ]
    write_json(
        output_dir / "stage_drift.command.json",
        {
            "python_exe": str(python_exe),
            "cmd": cmd,
            "cwd": str(PROJECT_DIR),
        },
    )
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    (output_dir / "stage_drift.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stage_drift.stderr.txt").write_text(result.stderr, encoding="utf-8")

    report_path = output_dir / "stage_drift_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Stage-drift report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return result, report


def run_real_image_validation(
    *,
    run_script: Path,
    images_dir: Path,
    results_dir: Path,
    app_dir: Path,
    platform: str,
    limit: int | None,
) -> subprocess.CompletedProcess[str]:
    run_script_wsl = to_wsl_path(run_script)
    command = [
        "cd",
        shlex.quote(to_wsl_path(PROJECT_DIR)),
        "&&",
        f"STAGE_DRIFT=0 bash {shlex.quote(run_script_wsl)}",
        "--images-dir",
        shlex.quote(to_wsl_path(images_dir)),
        "--results-dir",
        shlex.quote(to_wsl_path(results_dir)),
        "--app-dir",
        shlex.quote(to_wsl_path(app_dir)),
        "--platform",
        shlex.quote(platform),
        "--overwrite",
        "--no-stage-drift",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])

    return subprocess.run(
        ["wsl.exe", "bash", "-lc", " ".join(command)],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )


def x_to_image_px(x_offset: float, crop_left: float, crop_size: float) -> float:
    return crop_left + ((clamp_unit(x_offset) + 1.0) * 0.5 * crop_size)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}"


def build_comparison_overlay(
    *,
    image_path: Path,
    output_path: Path,
    annotations_path: Path | None,
    checkpoint_demo: dict[str, Any],
    pytorch_stage: dict[str, Any] | None,
    onnx_stage: dict[str, Any] | None,
    application_stage: dict[str, Any] | None,
    drift_message: str,
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop = _compute_crop_geometry(image.width, image.height)

        gt_box_xyxy = None
        gt_crop_box_xyxy = None
        gt_x_offset = None
        gt_size_proxy = None
        image_id = _extract_image_id(image_path)
        if image_id is not None and annotations_path is not None and annotations_path.is_file():
            gt_box_xyxy = _load_largest_person_box(annotations_path, image_id)
            if gt_box_xyxy is not None:
                gt_crop_box_xyxy = _crop_box_to_square(gt_box_xyxy, crop)
                if gt_crop_box_xyxy is not None:
                    gt_x_offset, gt_size_proxy = _gt_follow_target(gt_crop_box_xyxy, crop.size)

        checkpoint_x = checkpoint_demo["x_offset"]
        application_x = (
            application_stage["x_offset_clamped"] if application_stage and application_stage["x_offset_clamped"] is not None else 0.0
        )

        original_panel = image.copy()
        original_draw = ImageDraw.Draw(original_panel)
        crop_rect = [crop.left, crop.top, crop.left + crop.size, crop.top + crop.size]
        _draw_box(original_draw, crop_rect, CROP_COLOR, width=4)
        _draw_vertical_line(
            original_draw,
            x_to_image_px(checkpoint_x, crop.left, crop.size),
            crop.top,
            crop.top + crop.size,
            CHECKPOINT_COLOR,
            width=4,
        )
        _draw_vertical_line(
            original_draw,
            x_to_image_px(application_x, crop.left, crop.size),
            crop.top,
            crop.top + crop.size,
            APPLICATION_COLOR,
            width=4,
        )
        if gt_box_xyxy is not None:
            _draw_box(original_draw, gt_box_xyxy, GT_COLOR, width=4)
            gt_center_x = 0.5 * (gt_box_xyxy[0] + gt_box_xyxy[2])
            _draw_vertical_line(original_draw, gt_center_x, gt_box_xyxy[1], gt_box_xyxy[3], GT_COLOR, width=3)

        square = image.crop(
            (
                int(round(crop.left)),
                int(round(crop.top)),
                int(round(crop.left + crop.size)),
                int(round(crop.top + crop.size)),
            )
        )
        square = square.convert("L").resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), RESAMPLE_BILINEAR)
        display_size = 384
        crop_panel = square.convert("RGB").resize((display_size, display_size), RESAMPLE_NEAREST)
        crop_draw = ImageDraw.Draw(crop_panel)
        center_x = 0.5 * display_size
        _draw_vertical_line(crop_draw, center_x, 0.0, float(display_size), CENTER_COLOR, width=2)
        _draw_vertical_line(
            crop_draw,
            ((clamp_unit(checkpoint_x) + 1.0) * 0.5 * display_size),
            0.0,
            float(display_size),
            CHECKPOINT_COLOR,
            width=4,
        )
        _draw_vertical_line(
            crop_draw,
            ((clamp_unit(application_x) + 1.0) * 0.5 * display_size),
            0.0,
            float(display_size),
            APPLICATION_COLOR,
            width=4,
        )
        if gt_crop_box_xyxy is not None:
            scale = display_size / crop.size
            gt_box_display = [coord * scale for coord in gt_crop_box_xyxy]
            _draw_box(crop_draw, gt_box_display, GT_COLOR, width=4)
            gt_center_x = 0.5 * (gt_box_display[0] + gt_box_display[2])
            _draw_vertical_line(crop_draw, gt_center_x, gt_box_display[1], gt_box_display[3], GT_COLOR, width=3)

        canvas_width = original_panel.width + crop_panel.width + (PANEL_PADDING * 3)
        canvas_height = OVERLAY_HEADER_HEIGHT + max(original_panel.height, crop_panel.height) + PANEL_PADDING
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=BG_COLOR)
        canvas.paste(original_panel, (PANEL_PADDING, OVERLAY_HEADER_HEIGHT))
        canvas.paste(crop_panel, (original_panel.width + (PANEL_PADDING * 2), OVERLAY_HEADER_HEIGHT))

        header = ImageDraw.Draw(canvas)
        gt_line = "gt=not found"
        if gt_x_offset is not None:
            gt_line = f"gt_x={gt_x_offset:+.3f} gt_scale={gt_size_proxy:.3f}"

        demo_vs_app = abs(checkpoint_demo["x_offset"] - application_x)
        pytorch_x = None if not pytorch_stage else pytorch_stage["x_offset_raw"]
        onnx_x = None if not onnx_stage else onnx_stage["x_offset_raw"]
        app_x_raw = None if not application_stage else application_stage["x_offset_raw"]
        lines = [
            f"image={image_path.name}",
            (
                f"checkpoint demo: x={checkpoint_demo['x_offset']:+.3f} "
                f"raw_x={checkpoint_demo['raw_x_offset']:+.3f} "
                f"scale={checkpoint_demo['size_proxy']:.3f} "
                f"conf={checkpoint_demo['visibility_confidence']:.4f}"
            ),
            (
                f"stage drift: pytorch_x={_format_optional(pytorch_x)} "
                f"onnx_x={_format_optional(onnx_x)} "
                f"application_x={_format_optional(app_x_raw)}"
            ),
            f"|checkpoint-demo - application| = {demo_vs_app:.6f}",
            drift_message,
            gt_line,
            "legend: red=checkpoint before export, blue=generated application, yellow=model crop, green=largest GT person",
        ]
        _fit_text(header, lines, PANEL_PADDING, 18)

        font = ImageFont.load_default()
        label_y = OVERLAY_HEADER_HEIGHT - 24
        header.text((PANEL_PADDING, label_y), "original image", fill=TEXT_COLOR, font=font)
        header.text(
            (original_panel.width + (PANEL_PADDING * 2), label_y),
            "model crop (grayscale, 128x128 view)",
            fill=TEXT_COLOR,
            font=font,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    return {
        "output_path": str(output_path),
        "gt_x_offset": gt_x_offset,
        "gt_size_proxy": gt_size_proxy,
    }


def summarize_metric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "max": None,
            "min": None,
            "count_gt_0_05": 0,
            "count_gt_0_10": 0,
            "count_gt_0_25": 0,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
        "min": min(values),
        "count_gt_0_05": sum(value > 0.05 for value in values),
        "count_gt_0_10": sum(value > 0.10 for value in values),
        "count_gt_0_25": sum(value > 0.25 for value in values),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "image_name",
        "validation_status",
        "checkpoint_demo_x",
        "stage_pytorch_x",
        "stage_onnx_x",
        "application_x",
        "abs_diff_demo_vs_pytorch",
        "abs_diff_demo_vs_onnx",
        "abs_diff_demo_vs_application",
        "abs_diff_pytorch_vs_onnx",
        "abs_diff_onnx_vs_application",
        "abs_diff_pytorch_vs_application",
        "comparison_overlay_path",
        "application_sample_dir",
        "stage_drift_dir",
        "drift_onset",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "image_name": row["image_name"],
                    "validation_status": row["validation_status"],
                    "checkpoint_demo_x": row["checkpoint_demo"]["x_offset"],
                    "stage_pytorch_x": row["stage_outputs"]["pytorch"]["x_offset_raw"],
                    "stage_onnx_x": row["stage_outputs"]["onnx"]["x_offset_raw"],
                    "application_x": row["stage_outputs"]["gvsoc"]["x_offset_raw"],
                    "abs_diff_demo_vs_pytorch": row["x_abs_diff"]["demo_vs_pytorch"],
                    "abs_diff_demo_vs_onnx": row["x_abs_diff"]["demo_vs_onnx"],
                    "abs_diff_demo_vs_application": row["x_abs_diff"]["demo_vs_gvsoc"],
                    "abs_diff_pytorch_vs_onnx": row["x_abs_diff"]["pytorch_vs_onnx"],
                    "abs_diff_onnx_vs_application": row["x_abs_diff"]["onnx_vs_gvsoc"],
                    "abs_diff_pytorch_vs_application": row["x_abs_diff"]["pytorch_vs_gvsoc"],
                    "comparison_overlay_path": row["comparison_overlay_path"],
                    "application_sample_dir": row["application_sample_dir"],
                    "stage_drift_dir": row["stage_drift_dir"],
                    "drift_onset": row["drift_onset"]["message"],
                }
            )


def build_contact_sheet(overlay_paths: list[Path], output_path: Path) -> None:
    if not overlay_paths:
        return

    thumbs: list[Image.Image] = []
    thumb_width = 420
    thumb_height = 280
    for overlay_path in overlay_paths:
        with Image.open(overlay_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_width, thumb_height), RESAMPLE_BILINEAR)
            thumbs.append(image.copy())

    columns = min(4, max(1, math.ceil(math.sqrt(len(thumbs)))))
    rows = math.ceil(len(thumbs) / columns)
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs)
    label_height = 20
    sheet = Image.new(
        "RGB",
        (
            columns * (cell_width + PANEL_PADDING) + PANEL_PADDING,
            rows * (cell_height + label_height + PANEL_PADDING) + PANEL_PADDING,
        ),
        color=BG_COLOR,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (overlay_path, image) in enumerate(zip(overlay_paths, thumbs), start=0):
        row = index // columns
        col = index % columns
        x = PANEL_PADDING + (col * (cell_width + PANEL_PADDING))
        y = PANEL_PADDING + (row * (cell_height + label_height + PANEL_PADDING))
        sheet.paste(image, (x, y))
        draw.text((x, y + image.height + 4), overlay_path.stem, fill=TEXT_COLOR, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_markdown_summary(summary: dict[str, Any]) -> str:
    metric_demo_vs_app = summary["metrics"]["demo_vs_gvsoc"]
    metric_pytorch_vs_onnx = summary["metrics"]["pytorch_vs_onnx"]
    metric_onnx_vs_gvsoc = summary["metrics"]["onnx_vs_gvsoc"]

    lines = [
        "# Hybrid Follow Application vs Checkpoint",
        "",
        f"- Images dir: `{summary['images_dir']}`",
        f"- Validation dir: `{summary['validation_dir']}`",
        f"- Checkpoint: `{summary['ckpt']}`",
        f"- ONNX: `{summary['onnx']}`",
        f"- Validation command return code: `{summary['validation_returncode']}`",
        f"- Count: `{summary['count']}`",
        "",
        "## Aggregate X Diffs",
        "",
        (
            f"- checkpoint-demo vs application: mean={metric_demo_vs_app['mean']:.6f} "
            f"median={metric_demo_vs_app['median']:.6f} max={metric_demo_vs_app['max']:.6f} "
            f"gt0.10={metric_demo_vs_app['count_gt_0_10']}/{metric_demo_vs_app['count']}"
        ),
        (
            f"- stage-drift pytorch vs onnx: mean={metric_pytorch_vs_onnx['mean']:.6f} "
            f"median={metric_pytorch_vs_onnx['median']:.6f} max={metric_pytorch_vs_onnx['max']:.6f} "
            f"gt0.10={metric_pytorch_vs_onnx['count_gt_0_10']}/{metric_pytorch_vs_onnx['count']}"
        ),
        (
            f"- onnx vs application: mean={metric_onnx_vs_gvsoc['mean']:.6f} "
            f"median={metric_onnx_vs_gvsoc['median']:.6f} max={metric_onnx_vs_gvsoc['max']:.6f} "
            f"gt0.10={metric_onnx_vs_gvsoc['count_gt_0_10']}/{metric_onnx_vs_gvsoc['count']}"
        ),
        "",
        "## Per Image",
        "",
        "| image | demo x | pytorch x | onnx x | application x | |demo-app| | drift onset |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in summary["results"]:
        lines.append(
            "| "
            f"{row['image_name']} | "
            f"{row['checkpoint_demo']['x_offset']:+.3f} | "
            f"{row['stage_outputs']['pytorch']['x_offset_raw']:+.3f} | "
            f"{row['stage_outputs']['onnx']['x_offset_raw']:+.3f} | "
            f"{row['stage_outputs']['gvsoc']['x_offset_raw']:+.3f} | "
            f"{row['x_abs_diff']['demo_vs_gvsoc']:.3f} | "
            f"{row['drift_onset']['message']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Comparison CSV: `{summary['summary_csv']}`",
            f"- Comparison contact sheet: `{summary['contact_sheet']}`",
            f"- Per-image overlays: `{summary['comparison_overlay_dir']}`",
            f"- Application validation outputs: `{summary['validation_dir']}`",
            f"- Per-image stage-drift outputs: `{summary['stage_drift_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    images_dir = resolve_repo_path(args.images_dir)
    if images_dir is None or not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    ckpt_path = resolve_repo_path(args.ckpt)
    onnx_path = resolve_repo_path(args.onnx)
    run_script = resolve_repo_path(args.run_script)
    app_dir = resolve_repo_path(args.app_dir)
    annotations_path = resolve_repo_path(args.annotations)
    python_exe = resolve_stage_drift_python(args.python)

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if onnx_path is None or not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if run_script is None or not run_script.is_file():
        raise FileNotFoundError(f"Validation script not found: {run_script}")
    if app_dir is None or not app_dir.is_dir():
        raise FileNotFoundError(f"Application directory not found: {app_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = (
        resolve_repo_path(args.results_dir)
        if args.results_dir
        else (PROJECT_DIR / "logs" / "hybrid_follow_val" / f"application_vs_checkpoint_{timestamp}")
    )
    if results_dir is None:
        raise RuntimeError("Could not resolve results directory.")
    ensure_output_dir(results_dir, overwrite=args.overwrite)

    application_validation_dir = results_dir / "application_validation"
    stage_drift_root = results_dir / "stage_drift"
    comparison_overlay_dir = results_dir / "comparison_overlays"
    application_validation_dir.mkdir(parents=True, exist_ok=True)
    stage_drift_root.mkdir(parents=True, exist_ok=True)
    comparison_overlay_dir.mkdir(parents=True, exist_ok=True)

    validation_result = run_real_image_validation(
        run_script=run_script,
        images_dir=images_dir,
        results_dir=application_validation_dir,
        app_dir=app_dir,
        platform=args.platform,
        limit=args.limit,
    )
    (results_dir / "run_real_image_val.stdout.txt").write_text(validation_result.stdout, encoding="utf-8")
    (results_dir / "run_real_image_val.stderr.txt").write_text(validation_result.stderr, encoding="utf-8")

    validation_summary_path = application_validation_dir / "summary.json"
    if not validation_summary_path.is_file():
        raise FileNotFoundError(
            f"Validation summary not found after run_real_image_val.sh completed: {validation_summary_path}"
        )
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    validation_rows = validation_summary.get("results", [])
    if not validation_rows:
        raise RuntimeError(f"No validation rows found in {validation_summary_path}")

    demo_model, demo_device = load_demo_model(ckpt_path)

    results: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    for index, validation_row in enumerate(validation_rows, start=1):
        image_path = resolve_repo_path(validation_row["image_path"])
        sample_dir = resolve_repo_path(validation_row["sample_dir"])
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"Validation image not found: {validation_row['image_path']}")
        if sample_dir is None or not sample_dir.is_dir():
            raise FileNotFoundError(f"Validation sample dir not found: {validation_row['sample_dir']}")

        stage_drift_dir = stage_drift_root / f"{index:04d}_{sanitize_name(image_path.stem)}"
        golden_path = sample_dir / "output.txt"
        gvsoc_json_path = sample_dir / "gvsoc_final_tensor.json"
        if not golden_path.is_file():
            raise FileNotFoundError(f"Golden output artifact not found: {golden_path}")
        if not gvsoc_json_path.is_file():
            raise FileNotFoundError(f"GVSOC final tensor artifact not found: {gvsoc_json_path}")

        checkpoint_demo = run_demo_inference(demo_model, demo_device, image_path)
        write_json(stage_drift_dir / "checkpoint_demo.json", checkpoint_demo)

        stage_drift_process, stage_drift_report = run_stage_drift(
            python_exe=python_exe,
            image_path=image_path,
            ckpt_path=ckpt_path,
            onnx_path=onnx_path,
            golden_path=golden_path,
            gvsoc_json_path=gvsoc_json_path,
            output_dir=stage_drift_dir,
        )

        stages = stage_drift_report["stages"]
        pytorch_stage = compact_stage(stages.get("pytorch"))
        onnx_stage = compact_stage(stages.get("onnx"))
        gvsoc_stage = compact_stage(stages.get("gvsoc"))
        golden_stage = compact_stage(stages.get("golden"))
        if pytorch_stage is None or onnx_stage is None or gvsoc_stage is None or golden_stage is None:
            raise RuntimeError(f"Missing required stages in {stage_drift_dir / 'stage_drift_report.json'}")

        comparison_overlay_path = comparison_overlay_dir / f"{index:04d}_{sanitize_name(image_path.stem)}.png"
        overlay_metadata = build_comparison_overlay(
            image_path=image_path,
            output_path=comparison_overlay_path,
            annotations_path=annotations_path,
            checkpoint_demo=checkpoint_demo,
            pytorch_stage=pytorch_stage,
            onnx_stage=onnx_stage,
            application_stage=gvsoc_stage,
            drift_message=stage_drift_report["drift_onset"]["message"],
        )
        overlay_paths.append(comparison_overlay_path)

        x_abs_diff = {
            "demo_vs_pytorch": abs(checkpoint_demo["x_offset"] - clamp_unit(pytorch_stage["x_offset_raw"])),
            "demo_vs_onnx": abs(checkpoint_demo["x_offset"] - clamp_unit(onnx_stage["x_offset_raw"])),
            "demo_vs_gvsoc": abs(checkpoint_demo["x_offset"] - clamp_unit(gvsoc_stage["x_offset_raw"])),
            "pytorch_vs_onnx": abs(float(pytorch_stage["x_offset_raw"]) - float(onnx_stage["x_offset_raw"])),
            "onnx_vs_gvsoc": abs(float(onnx_stage["x_offset_raw"]) - float(gvsoc_stage["x_offset_raw"])),
            "pytorch_vs_gvsoc": abs(float(pytorch_stage["x_offset_raw"]) - float(gvsoc_stage["x_offset_raw"])),
        }

        result_row = {
            "index": index,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "validation_status": validation_row.get("status"),
            "validation_returncode": validation_row.get("returncode"),
            "application_sample_dir": str(sample_dir),
            "application_overlay_path": validation_row.get("overlay_path"),
            "comparison_overlay_path": str(comparison_overlay_path),
            "checkpoint_demo": checkpoint_demo,
            "stage_outputs": {
                "pytorch": pytorch_stage,
                "onnx": onnx_stage,
                "golden": golden_stage,
                "gvsoc": gvsoc_stage,
            },
            "drift_onset": stage_drift_report["drift_onset"],
            "pairwise": stage_drift_report["pairwise"],
            "x_abs_diff": x_abs_diff,
            "gt_x_offset": overlay_metadata["gt_x_offset"],
            "gt_size_proxy": overlay_metadata["gt_size_proxy"],
            "stage_drift_dir": str(stage_drift_dir),
            "stage_drift_returncode": stage_drift_process.returncode,
        }
        write_json(stage_drift_dir / "checkpoint_vs_application.json", result_row)
        results.append(result_row)

    metrics = {
        "demo_vs_gvsoc": summarize_metric([row["x_abs_diff"]["demo_vs_gvsoc"] for row in results]),
        "demo_vs_onnx": summarize_metric([row["x_abs_diff"]["demo_vs_onnx"] for row in results]),
        "demo_vs_pytorch": summarize_metric([row["x_abs_diff"]["demo_vs_pytorch"] for row in results]),
        "pytorch_vs_onnx": summarize_metric([row["x_abs_diff"]["pytorch_vs_onnx"] for row in results]),
        "onnx_vs_gvsoc": summarize_metric([row["x_abs_diff"]["onnx_vs_gvsoc"] for row in results]),
        "pytorch_vs_gvsoc": summarize_metric([row["x_abs_diff"]["pytorch_vs_gvsoc"] for row in results]),
    }

    contact_sheet_path = results_dir / "comparison_contact_sheet.png"
    build_contact_sheet(overlay_paths, contact_sheet_path)

    summary_csv_path = results_dir / "summary.csv"
    write_summary_csv(summary_csv_path, results)

    summary = {
        "images_dir": str(images_dir),
        "validation_dir": str(application_validation_dir),
        "stage_drift_dir": str(stage_drift_root),
        "comparison_overlay_dir": str(comparison_overlay_dir),
        "contact_sheet": str(contact_sheet_path),
        "summary_csv": str(summary_csv_path),
        "results_dir": str(results_dir),
        "ckpt": str(ckpt_path),
        "onnx": str(onnx_path),
        "app_dir": str(app_dir),
        "count": len(results),
        "validation_returncode": validation_result.returncode,
        "metrics": metrics,
        "results": results,
    }
    write_json(results_dir / "summary.json", summary)
    (results_dir / "summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")

    print(f"Results dir: {results_dir}")
    print(f"Validation return code: {validation_result.returncode}")
    print(
        "checkpoint-demo vs application x abs diff: "
        f"mean={metrics['demo_vs_gvsoc']['mean']:.6f} "
        f"median={metrics['demo_vs_gvsoc']['median']:.6f} "
        f"max={metrics['demo_vs_gvsoc']['max']:.6f}"
    )
    print(f"Comparison contact sheet: {contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

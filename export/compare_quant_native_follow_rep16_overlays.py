#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

from models.follow_model_factory import build_follow_model_from_checkpoint, load_checkpoint_payload  # noqa: E402
from export_nemo_quant_core import semantic_output  # noqa: E402
from utils.coco_follow_regression import compute_follow_target  # noqa: E402
from utils.follow_task import (  # noqa: E402
    compute_follow_metrics,
    follow_runtime_decode_summary,
    summarize_follow_bin_preservation,
)
from utils.transforms import get_val_transforms  # noqa: E402
from validate_follow_rep16_overlays import (  # noqa: E402
    AnnotationIndex,
    build_contact_sheet,
    discover_images,
)
from visualize_hybrid_follow_prediction import (  # noqa: E402
    BG_COLOR,
    CENTER_COLOR,
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
    _extract_image_id,
    _fit_text,
    _gt_follow_target,
    _load_largest_person_box,
)


DEFAULT_REP16_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
DEFAULT_CKPT = PROJECT_DIR / "training" / "dronet_lite_follow" / "dronet_lite_follow_best_x.pth"
DEFAULT_ONNX = PROJECT_DIR / "logs" / "dronet_lite_follow_val" / "model_id.onnx"
OVERLAY_DIRNAME = "comparison_overlays"
PRE_COLOR = (220, 53, 69)
POST_COLOR = (46, 117, 182)
PRE_SIZE_COLOR = PRE_COLOR
POST_SIZE_COLOR = POST_COLOR
GT_SIZE_COLOR = (35, 138, 65)
HEADER_HEIGHT = 250
DISPLAY_SIZE = 384
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare float checkpoint and post-quant ONNX outputs on rep16 and render side-by-side overlays."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Pre-quant checkpoint.")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Post-quant ONNX model.")
    parser.add_argument("--output-dir", required=True, help="Root output directory.")
    parser.add_argument("--images-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANN))
    parser.add_argument(
        "--dataset-label",
        default="rep16",
        help="Label used in summary metadata and markdown headings for the compared image set.",
    )
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_layout(root_dir: Path, overwrite: bool) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = root_dir / OVERLAY_DIRNAME
    if overlay_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Overlay directory already exists: {overlay_dir}")
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return overlay_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "image_name",
        "image_path",
        "overlay_path",
        "gt_visible",
        "gt_x_offset",
        "gt_size_proxy",
        "pre_visible",
        "pre_visibility_confidence",
        "pre_x_value",
        "pre_size_value",
        "post_visible",
        "post_visibility_confidence",
        "post_x_value",
        "post_size_value",
        "x_delta_abs",
        "size_delta_abs",
        "visibility_agreement",
        "pre_x_error",
        "pre_size_error",
        "post_x_error",
        "post_size_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _draw_vertical_line(
    draw: ImageDraw.ImageDraw,
    x: float,
    y1: float,
    y2: float,
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    draw.line([(x, y1), (x, y2)], fill=color, width=width)


def _draw_size_ruler(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    bottom: float,
    crop_size: float,
    size_value: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    height = max(0.0, min(1.0, float(size_value))) * crop_size
    y_top = bottom - height
    draw.line([(x, bottom), (x, y_top)], fill=color, width=width)
    draw.line([(x - 8, y_top), (x + 8, y_top)], fill=color, width=width)
    draw.line([(x - 8, bottom), (x + 8, bottom)], fill=color, width=width)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}"


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _x_to_image_px(x_offset: float, crop_left: float, crop_size: float) -> float:
    return crop_left + ((_clamp_unit(x_offset) + 1.0) * 0.5 * crop_size)


def build_comparison_overlay(
    *,
    image_path: Path,
    output_path: Path,
    annotations_path: Path,
    pre_summary: dict[str, Any],
    post_summary: dict[str, Any],
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop = _compute_crop_geometry(image.width, image.height)
        image_id = _extract_image_id(image_path)

        gt_box_xyxy = None
        gt_crop_box_xyxy = None
        gt_x_offset = None
        gt_size_proxy = None
        if image_id is not None and annotations_path.is_file():
            gt_box_xyxy = _load_largest_person_box(annotations_path, image_id)
            if gt_box_xyxy is not None:
                gt_crop_box_xyxy = _crop_box_to_square(gt_box_xyxy, crop)
                if gt_crop_box_xyxy is not None:
                    gt_x_offset, gt_size_proxy = _gt_follow_target(gt_crop_box_xyxy, crop.size)

        original_panel = image.copy()
        original_draw = ImageDraw.Draw(original_panel)
        crop_rect = [crop.left, crop.top, crop.left + crop.size, crop.top + crop.size]
        original_draw.rectangle(crop_rect, outline=CROP_COLOR, width=4)
        _draw_vertical_line(
            original_draw,
            _x_to_image_px(float(pre_summary["x_value"]), crop.left, crop.size),
            crop.top,
            crop.top + crop.size,
            PRE_COLOR,
            width=4,
        )
        _draw_vertical_line(
            original_draw,
            _x_to_image_px(float(post_summary["x_value"]), crop.left, crop.size),
            crop.top,
            crop.top + crop.size,
            POST_COLOR,
            width=4,
        )
        if gt_box_xyxy is not None:
            original_draw.rectangle(gt_box_xyxy, outline=GT_COLOR, width=4)
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
        crop_panel = square.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), RESAMPLE_NEAREST)
        crop_draw = ImageDraw.Draw(crop_panel)
        center_x = 0.5 * DISPLAY_SIZE
        _draw_vertical_line(crop_draw, center_x, 0.0, float(DISPLAY_SIZE), CENTER_COLOR, width=2)
        _draw_vertical_line(
            crop_draw,
            ((_clamp_unit(float(pre_summary["x_value"])) + 1.0) * 0.5 * DISPLAY_SIZE),
            0.0,
            float(DISPLAY_SIZE),
            PRE_COLOR,
            width=4,
        )
        _draw_vertical_line(
            crop_draw,
            ((_clamp_unit(float(post_summary["x_value"])) + 1.0) * 0.5 * DISPLAY_SIZE),
            0.0,
            float(DISPLAY_SIZE),
            POST_COLOR,
            width=4,
        )
        _draw_size_ruler(
            crop_draw,
            x=DISPLAY_SIZE - 24,
            bottom=DISPLAY_SIZE - 12,
            crop_size=DISPLAY_SIZE - 24,
            size_value=float(pre_summary["size_value"]),
            color=PRE_SIZE_COLOR,
            width=4,
        )
        _draw_size_ruler(
            crop_draw,
            x=DISPLAY_SIZE - 44,
            bottom=DISPLAY_SIZE - 12,
            crop_size=DISPLAY_SIZE - 24,
            size_value=float(post_summary["size_value"]),
            color=POST_SIZE_COLOR,
            width=4,
        )
        if gt_crop_box_xyxy is not None:
            scale = DISPLAY_SIZE / crop.size
            gt_box_display = [coord * scale for coord in gt_crop_box_xyxy]
            crop_draw.rectangle(gt_box_display, outline=GT_COLOR, width=4)
            gt_center_x = 0.5 * (gt_box_display[0] + gt_box_display[2])
            _draw_vertical_line(crop_draw, gt_center_x, gt_box_display[1], gt_box_display[3], GT_COLOR, width=3)
            _draw_size_ruler(
                crop_draw,
                x=DISPLAY_SIZE - 64,
                bottom=DISPLAY_SIZE - 12,
                crop_size=DISPLAY_SIZE - 24,
                size_value=float(gt_size_proxy or 0.0),
                color=GT_SIZE_COLOR,
                width=4,
            )

        canvas_width = original_panel.width + crop_panel.width + (PANEL_PADDING * 3)
        canvas_height = HEADER_HEIGHT + max(original_panel.height, crop_panel.height) + PANEL_PADDING
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=BG_COLOR)
        canvas.paste(original_panel, (PANEL_PADDING, HEADER_HEIGHT))
        canvas.paste(crop_panel, (original_panel.width + (PANEL_PADDING * 2), HEADER_HEIGHT))

        header = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        gt_line = "gt=negative"
        if gt_x_offset is not None:
            gt_line = f"gt_x={gt_x_offset:+.3f} gt_size={gt_size_proxy:.3f}"
        lines = [
            image_path.name,
            (
                f"pre : vis={bool(int(pre_summary['target_visible']))} "
                f"conf={float(pre_summary['visibility_confidence']):.3f} "
                f"x={float(pre_summary['x_value']):+.3f} "
                f"size={float(pre_summary['size_value']):.3f}"
            ),
            (
                f"post: vis={bool(int(post_summary['target_visible']))} "
                f"conf={float(post_summary['visibility_confidence']):.3f} "
                f"x={float(post_summary['x_value']):+.3f} "
                f"size={float(post_summary['size_value']):.3f}"
            ),
            (
                f"|pre-post|: dx={abs(float(pre_summary['x_value']) - float(post_summary['x_value'])):.3f} "
                f"dsize={abs(float(pre_summary['size_value']) - float(post_summary['size_value'])):.3f} "
                f"vis_agree={bool(int(pre_summary['target_visible']) == int(post_summary['target_visible']))}"
            ),
            gt_line,
            "legend: red=pre-quant checkpoint, blue=post-quant ONNX, yellow=model crop, green=largest GT person",
        ]
        _fit_text(header, lines, PANEL_PADDING, 18, line_gap=10)
        header.text(
            (PANEL_PADDING, HEADER_HEIGHT - 24),
            "size rulers: red=pre, blue=post, green=gt",
            fill=TEXT_COLOR,
            font=font,
        )

        canvas.save(output_path)
        return {
            "overlay_path": str(output_path),
            "gt_x_offset": gt_x_offset,
            "gt_size_proxy": gt_size_proxy,
        }


def build_eval_inputs(
    *,
    image_path: Path,
    annotations: AnnotationIndex,
    model_type: str,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_id = _extract_image_id(image_path)
    if image_id is None:
        raise ValueError(f"Could not extract COCO image id from {image_path.name}")

    boxes = annotations.boxes_for_image(image_id)
    target = {
        "boxes": boxes,
        "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
        "area": torch.zeros((boxes.shape[0],), dtype=torch.float32),
        "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "true_no_person": torch.tensor([1 if boxes.numel() == 0 else 0], dtype=torch.int64),
    }
    transform = get_val_transforms(model_type=model_type, input_channels=1, image_size=image_size)
    with Image.open(image_path) as image:
        tensor, transformed = transform(image.convert("L"), target)

    follow_target, _ = compute_follow_target(
        transformed["boxes"],
        image_height=int(tensor.shape[-2]),
        image_width=int(tensor.shape[-1]),
    )
    float_input = tensor.unsqueeze(0).to(dtype=torch.float32)
    staged_input = torch.round(torch.clamp(float_input, 0.0, 1.0) * 255.0).to(dtype=torch.float32)
    return (
        float_input,
        staged_input,
        follow_target.to(dtype=torch.float32),
        transformed["true_no_person"].view(-1).to(dtype=torch.int64),
    )


def build_summary_markdown(summary: dict[str, Any]) -> str:
    dataset_label = str(summary.get("dataset_label") or "rep16")
    model_type = str(summary.get("model_type") or "follow_model")
    pre = summary["metrics"]["pre_quant"]
    post = summary["metrics"]["post_quant"]
    drift = summary["metrics"]["pre_to_post"]
    return "\n".join(
        [
            f"# {model_type} {dataset_label} comparison",
            "",
            "## Pre-Quant",
            f"- follow_score: `{pre.get('follow_score')}`",
            f"- x_mae: `{pre.get('x_mae')}`",
            f"- size_mae: `{pre.get('size_mae')}`",
            f"- no_person_fp_rate: `{pre.get('no_person_fp_rate')}`",
            "",
            "## Post-Quant",
            f"- follow_score: `{post.get('follow_score')}`",
            f"- x_mae: `{post.get('x_mae')}`",
            f"- size_mae: `{post.get('size_mae')}`",
            f"- no_person_fp_rate: `{post.get('no_person_fp_rate')}`",
            "",
            "## Pre->Post Drift",
            f"- visibility_gate_agreement: `{drift.get('visibility_gate_agreement')}`",
            f"- x_value_mae: `{drift.get('x_value_mae')}`",
            f"- size_value_mae: `{drift.get('size_value_mae')}`",
            f"- x_bin_exact_match_rate: `{drift.get('x_bin_exact_match_rate')}`",
            f"- x_bin_adjacent_match_rate: `{drift.get('x_bin_adjacent_match_rate')}`",
            f"- mean_abs_bin_delta: `{drift.get('mean_abs_bin_delta')}`",
            "",
            "## Artifacts",
            f"- overlays: `{summary['comparison_overlay_dir']}`",
            f"- contact_sheet: `{summary['contact_sheet']}`",
        ]
    )


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    overlay_dir = ensure_layout(output_dir, args.overwrite)

    device = torch.device("cpu")
    payload = load_checkpoint_payload(ckpt_path, device)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint metadata is not a dict.")
    model_type = str(payload.get("model_type"))
    head_type = payload.get("follow_head_type")
    image_size = (int(payload.get("height", 128)), int(payload.get("width", 128)))

    model = build_follow_model_from_checkpoint(ckpt_path, device).eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_name = session.get_inputs()[0].name
    onnx_output_name = session.get_outputs()[0].name

    annotations = AnnotationIndex(annotations_path)
    image_paths = discover_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No rep16 images found under {images_dir}")

    pre_outputs: list[torch.Tensor] = []
    post_outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    no_person: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        float_input, staged_input, follow_target, true_no_person = build_eval_inputs(
            image_path=image_path,
            annotations=annotations,
            model_type=model_type,
            image_size=image_size,
        )

        with torch.no_grad():
            pre_raw = model(float_input.to(device)).detach().cpu()
        post_raw_onnx = np.asarray(
            session.run(
                [onnx_output_name],
                {onnx_input_name: staged_input.detach().cpu().numpy()},
            )[0],
            dtype=np.float32,
        )
        post_raw = torch.tensor(
            np.asarray(semantic_output(post_raw_onnx, "id"), dtype=np.float32).reshape(1, -1)
        )

        pre_summary = follow_runtime_decode_summary(
            pre_raw[0],
            head_type=head_type,
            model_type=model_type,
            vis_thresh=float(args.vis_thresh),
        )
        post_summary = follow_runtime_decode_summary(
            post_raw[0],
            head_type=head_type,
            model_type=model_type,
            vis_thresh=float(args.vis_thresh),
        )

        overlay_path = overlay_dir / f"{index:04d}_{image_path.stem}.png"
        overlay_meta = build_comparison_overlay(
            image_path=image_path,
            output_path=overlay_path,
            annotations_path=annotations_path,
            pre_summary=pre_summary,
            post_summary=post_summary,
        )

        gt_visible = bool(float(follow_target[2].item()) > 0.5)
        row = {
            "index": index,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "gt_visible": gt_visible,
            "gt_x_offset": float(follow_target[0].item()) if gt_visible else None,
            "gt_size_proxy": float(follow_target[1].item()) if gt_visible else None,
            "pre_visible": bool(int(pre_summary["target_visible"])),
            "pre_visibility_confidence": float(pre_summary["visibility_confidence"]),
            "pre_x_value": float(pre_summary["x_value"]),
            "pre_size_value": float(pre_summary["size_value"]),
            "post_visible": bool(int(post_summary["target_visible"])),
            "post_visibility_confidence": float(post_summary["visibility_confidence"]),
            "post_x_value": float(post_summary["x_value"]),
            "post_size_value": float(post_summary["size_value"]),
            "x_delta_abs": abs(float(pre_summary["x_value"]) - float(post_summary["x_value"])),
            "size_delta_abs": abs(float(pre_summary["size_value"]) - float(post_summary["size_value"])),
            "visibility_agreement": bool(int(pre_summary["target_visible"]) == int(post_summary["target_visible"])),
            "pre_x_error": (
                abs(float(pre_summary["x_value"]) - float(follow_target[0].item())) if gt_visible else None
            ),
            "pre_size_error": (
                abs(float(pre_summary["size_value"]) - float(follow_target[1].item())) if gt_visible else None
            ),
            "post_x_error": (
                abs(float(post_summary["x_value"]) - float(follow_target[0].item())) if gt_visible else None
            ),
            "post_size_error": (
                abs(float(post_summary["size_value"]) - float(follow_target[1].item())) if gt_visible else None
            ),
            "overlay_gt_x_offset": overlay_meta["gt_x_offset"],
            "overlay_gt_size_proxy": overlay_meta["gt_size_proxy"],
        }
        rows.append(row)
        overlay_paths.append(overlay_path)
        pre_outputs.append(pre_raw.squeeze(0))
        post_outputs.append(post_raw.squeeze(0))
        targets.append(follow_target)
        no_person.append(true_no_person)

    pre_tensor = torch.stack(pre_outputs, dim=0)
    post_tensor = torch.stack(post_outputs, dim=0)
    target_tensor = torch.stack(targets, dim=0)
    no_person_tensor = torch.cat(no_person, dim=0)

    pre_metrics = compute_follow_metrics(
        pre_tensor,
        target_tensor,
        head_type=head_type,
        model_type=model_type,
        vis_thresh=float(args.vis_thresh),
        true_no_person=no_person_tensor,
    )
    post_metrics = compute_follow_metrics(
        post_tensor,
        target_tensor,
        head_type=head_type,
        model_type=model_type,
        vis_thresh=float(args.vis_thresh),
        true_no_person=no_person_tensor,
    )
    drift_metrics = summarize_follow_bin_preservation(
        pre_tensor,
        post_tensor,
        head_type=head_type,
        model_type=model_type,
        vis_thresh=float(args.vis_thresh),
    )

    contact_sheet_path = output_dir / "comparison_contact_sheet.png"
    build_contact_sheet(overlay_paths, contact_sheet_path)
    csv_path = output_dir / "comparison_predictions.csv"
    write_csv(csv_path, rows)

    summary = {
        "checkpoint_path": str(ckpt_path),
        "onnx_path": str(onnx_path),
        "model_type": model_type,
        "follow_head_type": head_type,
        "dataset_label": str(args.dataset_label),
        "images_dir": str(images_dir),
        "annotations": str(annotations_path),
        "image_count": len(rows),
        "comparison_overlay_dir": str(overlay_dir),
        "contact_sheet": str(contact_sheet_path),
        "predictions_csv": str(csv_path),
        "metrics": {
            "pre_quant": pre_metrics,
            "post_quant": post_metrics,
            "pre_to_post": drift_metrics,
        },
        "rows": rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    write_markdown(output_dir / "comparison_summary.md", build_summary_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

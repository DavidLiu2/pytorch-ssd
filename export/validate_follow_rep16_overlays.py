#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.follow_model_factory import build_follow_model_from_checkpoint, load_checkpoint_payload  # noqa: E402
from utils.coco_follow_regression import compute_follow_target  # noqa: E402
from utils.follow_task import (  # noqa: E402
    compute_follow_metrics,
    decode_follow_outputs,
    follow_runtime_decode_summary,
)
from utils.transforms import get_val_transforms  # noqa: E402
from visualize_hybrid_follow_prediction import (  # noqa: E402
    BG_COLOR,
    CENTER_COLOR,
    CROP_COLOR,
    DEFAULT_ANN,
    GT_COLOR,
    MODEL_INPUT_SIZE,
    PANEL_PADDING,
    PRED_COLOR,
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
HEADER_HEIGHT = 220
DISPLAY_SIZE = 384
VISIBLE_COLOR = PRED_COLOR
HIDDEN_COLOR = (140, 140, 140)
SIZE_RULER_COLOR = (46, 117, 182)
GT_SIZE_RULER_COLOR = (35, 138, 65)


class AnnotationIndex:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        person_cat_ids = {
            int(category["id"])
            for category in payload.get("categories", [])
            if category.get("name") == "person"
        }
        if not person_cat_ids:
            person_cat_ids = {int(category["id"]) for category in payload.get("categories", [])}

        self.person_annotations: dict[int, list[dict[str, Any]]] = {}
        for ann in payload.get("annotations", []):
            if int(ann.get("category_id", -1)) not in person_cat_ids:
                continue
            self.person_annotations.setdefault(int(ann["image_id"]), []).append(ann)

    def boxes_for_image(self, image_id: int) -> torch.Tensor:
        anns = self.person_annotations.get(int(image_id), [])
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a follow checkpoint on rep16 and write prediction overlays."
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint to validate.")
    parser.add_argument("--output-dir", required=True, help="Directory for overlays and summary files.")
    parser.add_argument("--images-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANN))
    parser.add_argument(
        "--dataset-label",
        default="rep16",
        help="Label used in summary metadata for the evaluated image set.",
    )
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "image_name",
        "image_path",
        "overlay_path",
        "visible",
        "visibility_confidence",
        "x_value",
        "size_value",
        "x_error",
        "size_error",
        "gt_visible",
        "gt_x_offset",
        "gt_size_proxy",
        "head_type",
        "model_type",
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
    draw.line([(x - 10, y_top), (x + 10, y_top)], fill=color, width=width)
    draw.line([(x - 10, bottom), (x + 10, bottom)], fill=color, width=width)


def _format_opt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.3f}"


def build_overlay(
    *,
    image_path: Path,
    output_path: Path,
    annotations_path: Path,
    decoded_summary: dict[str, Any],
    vis_thresh: float,
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

        x_value = float(decoded_summary["x_value"])
        size_value = float(decoded_summary["size_value"])
        vis_conf = float(decoded_summary["visibility_confidence"])
        visible = bool(int(decoded_summary["target_visible"]))
        pred_color = VISIBLE_COLOR if visible else HIDDEN_COLOR

        original_panel = image.copy()
        original_draw = ImageDraw.Draw(original_panel)
        crop_rect = [crop.left, crop.top, crop.left + crop.size, crop.top + crop.size]
        original_draw.rectangle(crop_rect, outline=CROP_COLOR, width=4)
        pred_x = crop.left + ((max(-1.0, min(1.0, x_value)) + 1.0) * 0.5 * crop.size)
        _draw_vertical_line(
            original_draw,
            pred_x,
            crop.top,
            crop.top + crop.size,
            pred_color,
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
            ((max(-1.0, min(1.0, x_value)) + 1.0) * 0.5 * DISPLAY_SIZE),
            0.0,
            float(DISPLAY_SIZE),
            pred_color,
            width=4,
        )
        _draw_size_ruler(
            crop_draw,
            x=DISPLAY_SIZE - 26,
            bottom=DISPLAY_SIZE - 12,
            crop_size=DISPLAY_SIZE - 24,
            size_value=size_value,
            color=SIZE_RULER_COLOR,
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
                x=DISPLAY_SIZE - 48,
                bottom=DISPLAY_SIZE - 12,
                crop_size=DISPLAY_SIZE - 24,
                size_value=gt_size_proxy or 0.0,
                color=GT_SIZE_RULER_COLOR,
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
            f"pred_visible={visible} vis_conf={vis_conf:.3f} thresh={vis_thresh:.2f}",
            f"pred_x={x_value:+.3f} pred_size={size_value:.3f}",
            gt_line,
            f"head={decoded_summary.get('head_type')} model={decoded_summary.get('model_type')}",
        ]
        _fit_text(header, lines, PANEL_PADDING, 18, line_gap=10)

        canvas.save(output_path)
        return {
            "overlay_path": str(output_path),
            "gt_x_offset": gt_x_offset,
            "gt_size_proxy": gt_size_proxy,
        }


def build_contact_sheet(overlay_paths: list[Path], output_path: Path) -> None:
    if not overlay_paths:
        return

    thumbs = []
    thumb_width = 320
    for path in overlay_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            aspect = image.height / max(1, image.width)
            thumb_height = int(round(thumb_width * aspect))
            thumbs.append(image.resize((thumb_width, thumb_height), RESAMPLE_BILINEAR))

    columns = 2
    rows = int(math.ceil(len(thumbs) / float(columns)))
    row_heights = []
    for row_index in range(rows):
        start = row_index * columns
        end = min(len(thumbs), start + columns)
        row_heights.append(max(image.height for image in thumbs[start:end]) + 32)

    canvas_width = (columns * thumb_width) + ((columns + 1) * PANEL_PADDING)
    canvas_height = sum(row_heights) + ((rows + 1) * PANEL_PADDING)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y = PANEL_PADDING
    for row_index in range(rows):
        x = PANEL_PADDING
        start = row_index * columns
        end = min(len(thumbs), start + columns)
        for overlay_path, image in zip(overlay_paths[start:end], thumbs[start:end]):
            canvas.paste(image, (x, y))
            draw.text((x, y + image.height + 6), overlay_path.stem, fill=TEXT_COLOR, font=font)
            x += thumb_width + PANEL_PADDING
        y += row_heights[row_index]

    canvas.save(output_path)


def make_eval_row(
    *,
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    annotations: AnnotationIndex,
    annotations_path: Path,
    model_type: str,
    head_type: str | None,
    image_size: tuple[int, int],
    output_dir: Path,
    vis_thresh: float,
    index: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
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
    input_tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(input_tensor).detach().cpu()

    decoded_summary = follow_runtime_decode_summary(
        raw_output[0],
        head_type=head_type,
        model_type=model_type,
        vis_thresh=vis_thresh,
    )
    decoded_summary["head_type"] = head_type
    decoded_summary["model_type"] = model_type

    overlay_path = output_dir / f"{index:04d}_{image_path.stem}.png"
    overlay_metadata = build_overlay(
        image_path=image_path,
        output_path=overlay_path,
        annotations_path=annotations_path,
        decoded_summary=decoded_summary,
        vis_thresh=vis_thresh,
    )

    row = {
        "index": index,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "visible": bool(int(decoded_summary["target_visible"])),
        "visibility_confidence": float(decoded_summary["visibility_confidence"]),
        "x_value": float(decoded_summary["x_value"]),
        "size_value": float(decoded_summary["size_value"]),
        "gt_visible": bool(float(follow_target[2].item()) > 0.5),
        "gt_x_offset": float(follow_target[0].item()) if float(follow_target[2].item()) > 0.5 else None,
        "gt_size_proxy": float(follow_target[1].item()) if float(follow_target[2].item()) > 0.5 else None,
        "x_error": (
            abs(float(decoded_summary["x_value"]) - float(follow_target[0].item()))
            if float(follow_target[2].item()) > 0.5
            else None
        ),
        "size_error": (
            abs(float(decoded_summary["size_value"]) - float(follow_target[1].item()))
            if float(follow_target[2].item()) > 0.5
            else None
        ),
        "head_type": head_type,
        "model_type": model_type,
        "overlay_gt_x_offset": overlay_metadata["gt_x_offset"],
        "overlay_gt_size_proxy": overlay_metadata["gt_size_proxy"],
    }
    return (
        row,
        raw_output.squeeze(0),
        follow_target.to(dtype=torch.float32),
        transformed["true_no_person"].view(-1).to(dtype=torch.int64),
    )


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()

    ensure_output_dir(output_dir, args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint_payload(ckpt_path, device)
    model_type = str(payload.get("model_type"))
    head_type = payload.get("follow_head_type")
    image_size = (int(payload.get("height", 128)), int(payload.get("width", 128)))
    model = build_follow_model_from_checkpoint(ckpt_path, device).eval()

    annotations = AnnotationIndex(annotations_path)
    image_paths = discover_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    rows: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    no_person: list[torch.Tensor] = []
    overlay_paths: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        row, output, follow_target, true_no_person = make_eval_row(
            model=model,
            device=device,
            image_path=image_path,
            annotations=annotations,
            annotations_path=annotations_path,
            model_type=model_type,
            head_type=head_type,
            image_size=image_size,
            output_dir=output_dir,
            vis_thresh=float(args.vis_thresh),
            index=index,
        )
        rows.append(row)
        outputs.append(output)
        targets.append(follow_target)
        no_person.append(true_no_person)
        overlay_paths.append(Path(row["overlay_path"]))

    metrics = compute_follow_metrics(
        torch.stack(outputs, dim=0),
        torch.stack(targets, dim=0),
        head_type=head_type,
        model_type=model_type,
        vis_thresh=float(args.vis_thresh),
        true_no_person=torch.cat(no_person, dim=0),
    )

    contact_sheet_path = output_dir / "contact_sheet.png"
    build_contact_sheet(overlay_paths, contact_sheet_path)
    write_predictions_csv(output_dir / "predictions.csv", rows)

    summary = {
        "checkpoint_path": str(ckpt_path),
        "model_type": model_type,
        "follow_head_type": head_type,
        "dataset_label": str(args.dataset_label),
        "images_dir": str(images_dir),
        "annotations": str(annotations_path),
        "image_count": len(rows),
        "metrics": metrics,
        "contact_sheet": str(contact_sheet_path),
        "overlay_paths": [str(path) for path in overlay_paths],
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

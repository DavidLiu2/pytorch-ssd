#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ANN = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_Q_SCALE = 32768.0
DEFAULT_VIS_THRESHOLD = 0.5
MODEL_INPUT_SIZE = 128
PANEL_PADDING = 24
HEADER_HEIGHT = 150
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (250, 248, 244)
CROP_COLOR = (230, 173, 46)
PRED_COLOR = (220, 53, 69)
GT_COLOR = (35, 138, 65)
CENTER_COLOR = (120, 120, 120)
RULER_COLOR = (63, 81, 181)

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass
class CropGeometry:
    left: float
    top: float
    size: float


@dataclass
class FollowPrediction:
    raw_values: list[int]
    x_offset: float
    size_proxy: float
    visibility_logit: float
    confidence: float
    visible: bool


@dataclass
class OverlayResult:
    sample_dir: str
    overlay_path: str
    image_path: str
    raw_values: list[int]
    x_offset: float
    size_proxy: float
    visibility_confidence: float
    visible: bool
    gt_x_offset: float | None
    gt_size_proxy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_dir": self.sample_dir,
            "overlay_path": self.overlay_path,
            "image_path": self.image_path,
            "raw_values": self.raw_values,
            "x_offset": self.x_offset,
            "size_proxy": self.size_proxy,
            "visibility_confidence": self.visibility_confidence,
            "visible": self.visible,
            "gt_x_offset": self.gt_x_offset,
            "gt_size_proxy": self.gt_size_proxy,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a hybrid_follow final tensor on the source image by drawing the "
            "predicted x-position and optional COCO ground-truth person box."
        )
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        help=(
            "Validation sample directory containing comparison.json and "
            "gvsoc_final_tensor.json."
        ),
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANN),
        help="Optional COCO annotations json used to draw the largest person box.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Defaults to <sample-dir>/prediction_overlay.png.",
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


def resolve_repo_path(path_str: str | Path) -> Path:
    path_str = str(path_str)
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.resolve()

    wsl_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path_str)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        rest = wsl_match.group(2).replace("/", "\\")
        candidate = Path(f"{drive}:\\{rest}")
        if candidate.exists():
            return candidate.resolve()

    return Path(path_str)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_prediction(values: list[int], q_scale: float, vis_threshold: float) -> FollowPrediction:
    if len(values) != 3:
        raise ValueError(f"Expected 3 follow values, got {len(values)}")

    x_offset = max(-1.0, min(1.0, float(values[0]) / q_scale))
    size_proxy = max(0.0, min(1.0, float(values[1]) / q_scale))
    visibility_logit = float(values[2]) / q_scale
    confidence = 1.0 / (1.0 + math.exp(-visibility_logit))
    return FollowPrediction(
        raw_values=[int(v) for v in values],
        x_offset=x_offset,
        size_proxy=size_proxy,
        visibility_logit=visibility_logit,
        confidence=confidence,
        visible=confidence >= vis_threshold,
    )


def _compute_crop_geometry(width: int, height: int) -> CropGeometry:
    crop_size = float(min(width, height))
    crop_left = float((width - crop_size) // 2)
    crop_top = float((height - crop_size) // 2)
    return CropGeometry(left=crop_left, top=crop_top, size=crop_size)


def _extract_image_id(image_path: Path) -> int | None:
    stem = image_path.stem
    match = re.search(r"(\d{12})$", stem)
    if match:
        return int(match.group(1))
    return None


def _load_largest_person_box(annotations_path: Path, image_id: int) -> list[float] | None:
    if not annotations_path.is_file():
        return None

    data = _load_json(annotations_path)
    person_ids = {int(cat["id"]) for cat in data.get("categories", []) if cat.get("name") == "person"}
    if not person_ids:
        return None

    best_box = None
    best_area = -1.0
    for ann in data.get("annotations", []):
        if int(ann.get("image_id", -1)) != image_id:
            continue
        if int(ann.get("category_id", -1)) not in person_ids:
            continue
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        area = float(w) * float(h)
        if w <= 0 or h <= 0 or area <= best_area:
            continue
        best_area = area
        best_box = [float(x), float(y), float(x + w), float(y + h)]
    return best_box


def _crop_box_to_square(box_xyxy: list[float], crop: CropGeometry) -> list[float] | None:
    x1, y1, x2, y2 = box_xyxy
    x1 = min(max(x1 - crop.left, 0.0), crop.size)
    y1 = min(max(y1 - crop.top, 0.0), crop.size)
    x2 = min(max(x2 - crop.left, 0.0), crop.size)
    y2 = min(max(y2 - crop.top, 0.0), crop.size)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _gt_follow_target(crop_box_xyxy: list[float], crop_size: float) -> tuple[float, float]:
    x1, y1, x2, y2 = crop_box_xyxy
    center_x = 0.5 * (x1 + x2)
    height = max(0.0, y2 - y1)
    x_offset = max(-1.0, min(1.0, (2.0 * (center_x / crop_size)) - 1.0))
    size_proxy = max(0.0, min(1.0, height / crop_size))
    return x_offset, size_proxy


def _draw_box(draw: ImageDraw.ImageDraw, box_xyxy: list[float], color: tuple[int, int, int], width: int) -> None:
    draw.rectangle(box_xyxy, outline=color, width=width)


def _draw_vertical_line(
    draw: ImageDraw.ImageDraw,
    x: float,
    y1: float,
    y2: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line([(x, y1), (x, y2)], fill=color, width=width)


def _annotate_original_panel(
    image: Image.Image,
    crop: CropGeometry,
    prediction: FollowPrediction,
    gt_box_xyxy: list[float] | None,
) -> Image.Image:
    panel = image.convert("RGB")
    draw = ImageDraw.Draw(panel)

    crop_rect = [crop.left, crop.top, crop.left + crop.size, crop.top + crop.size]
    _draw_box(draw, crop_rect, CROP_COLOR, width=4)

    pred_x = crop.left + ((prediction.x_offset + 1.0) * 0.5 * crop.size)
    _draw_vertical_line(draw, pred_x, crop.top, crop.top + crop.size, PRED_COLOR, width=4)

    if gt_box_xyxy is not None:
        _draw_box(draw, gt_box_xyxy, GT_COLOR, width=4)
        gt_center_x = 0.5 * (gt_box_xyxy[0] + gt_box_xyxy[2])
        _draw_vertical_line(draw, gt_center_x, gt_box_xyxy[1], gt_box_xyxy[3], GT_COLOR, width=3)

    return panel


def _annotate_crop_panel(
    image: Image.Image,
    crop: CropGeometry,
    prediction: FollowPrediction,
    gt_crop_box_xyxy: list[float] | None,
) -> Image.Image:
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
    panel = square.convert("RGB").resize((display_size, display_size), RESAMPLE_NEAREST)
    draw = ImageDraw.Draw(panel)

    center_x = 0.5 * display_size
    pred_x = ((prediction.x_offset + 1.0) * 0.5 * display_size)
    pred_height = prediction.size_proxy * display_size

    _draw_vertical_line(draw, center_x, 0.0, float(display_size), CENTER_COLOR, width=2)
    _draw_vertical_line(draw, pred_x, 0.0, float(display_size), PRED_COLOR, width=4)

    if gt_crop_box_xyxy is not None:
        scale = display_size / crop.size
        gt_box_display = [coord * scale for coord in gt_crop_box_xyxy]
        _draw_box(draw, gt_box_display, GT_COLOR, width=4)
        gt_center_x = 0.5 * (gt_box_display[0] + gt_box_display[2])
        _draw_vertical_line(draw, gt_center_x, gt_box_display[1], gt_box_display[3], GT_COLOR, width=3)

    ruler_x = display_size - 16
    ruler_y2 = display_size - 12
    ruler_y1 = max(12.0, ruler_y2 - pred_height)
    _draw_vertical_line(draw, ruler_x, ruler_y1, ruler_y2, RULER_COLOR, width=5)
    draw.line([(ruler_x - 8, ruler_y1), (ruler_x + 8, ruler_y1)], fill=RULER_COLOR, width=3)
    draw.line([(ruler_x - 8, ruler_y2), (ruler_x + 8, ruler_y2)], fill=RULER_COLOR, width=3)

    return panel


def _fit_text(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, line_gap: int = 4) -> None:
    font = ImageFont.load_default()
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=TEXT_COLOR, font=font)
        bbox = draw.textbbox((x, current_y), line, font=font)
        current_y = int(bbox[3] + line_gap)


def _compose_output(
    image_path: Path,
    original_panel: Image.Image,
    crop_panel: Image.Image,
    prediction: FollowPrediction,
    crop: CropGeometry,
    gt_box_xyxy: list[float] | None,
    gt_crop_box_xyxy: list[float] | None,
) -> Image.Image:
    canvas_width = original_panel.width + crop_panel.width + (PANEL_PADDING * 3)
    canvas_height = HEADER_HEIGHT + max(original_panel.height, crop_panel.height) + PANEL_PADDING
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    canvas.paste(original_panel, (PANEL_PADDING, HEADER_HEIGHT))
    canvas.paste(crop_panel, (original_panel.width + (PANEL_PADDING * 2), HEADER_HEIGHT))

    gt_text = "gt=not found"
    if gt_crop_box_xyxy is not None:
        gt_x_offset, gt_size_proxy = _gt_follow_target(gt_crop_box_xyxy, crop.size)
        gt_text = f"gt_x={gt_x_offset:+.3f} gt_scale={gt_size_proxy:.3f}"

    lines = [
        f"image={image_path.name}",
        (
            f"pred raw={prediction.raw_values} x={prediction.x_offset:+.3f} "
            f"scale={prediction.size_proxy:.3f} conf={prediction.confidence:.4f} "
            f"visible={int(prediction.visible)}"
        ),
        (
            f"crop square: left={crop.left:.0f} top={crop.top:.0f} "
            f"size={crop.size:.0f}  pred_x_px={crop.left + ((prediction.x_offset + 1.0) * 0.5 * crop.size):.1f}"
        ),
        gt_text,
        "legend: red=predicted x, yellow=model crop region, green=largest GT person, blue=predicted height proxy",
    ]
    _fit_text(draw, lines, PANEL_PADDING, 18)

    label_y = HEADER_HEIGHT - 24
    draw.text((PANEL_PADDING, label_y), "original image", fill=TEXT_COLOR, font=ImageFont.load_default())
    draw.text(
        (original_panel.width + (PANEL_PADDING * 2), label_y),
        "model crop (grayscale, 128x128 view)",
        fill=TEXT_COLOR,
        font=ImageFont.load_default(),
    )

    return canvas


def generate_overlay_for_sample(
    sample_dir: str | Path,
    annotations_path: str | Path = DEFAULT_ANN,
    output_path: str | Path | None = None,
    q_scale: float = DEFAULT_Q_SCALE,
    vis_threshold: float = DEFAULT_VIS_THRESHOLD,
) -> OverlayResult:
    sample_dir_path = resolve_repo_path(sample_dir)
    comparison_path = sample_dir_path / "comparison.json"
    tensor_path = sample_dir_path / "gvsoc_final_tensor.json"
    metadata_path = sample_dir_path / "metadata.json"
    resolved_output_path = (
        resolve_repo_path(output_path)
        if output_path is not None
        else (sample_dir_path / "prediction_overlay.png")
    )

    if not comparison_path.is_file():
        raise FileNotFoundError(f"comparison.json not found: {comparison_path}")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"gvsoc_final_tensor.json not found: {tensor_path}")

    comparison = _load_json(comparison_path)
    tensor_json = _load_json(tensor_path)
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}

    image_path = resolve_repo_path(str(comparison.get("image_path") or metadata.get("image_path")))
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    prediction = _decode_prediction(
        values=[int(v) for v in tensor_json.get("values", [])],
        q_scale=q_scale,
        vis_threshold=vis_threshold,
    )

    gt_x_offset = None
    gt_size_proxy = None

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        crop = _compute_crop_geometry(image.width, image.height)

        gt_box_xyxy = None
        gt_crop_box_xyxy = None
        image_id = _extract_image_id(image_path)
        resolved_annotations_path = resolve_repo_path(annotations_path)
        if image_id is not None and resolved_annotations_path.is_file():
            gt_box_xyxy = _load_largest_person_box(resolved_annotations_path, image_id)
            if gt_box_xyxy is not None:
                gt_crop_box_xyxy = _crop_box_to_square(gt_box_xyxy, crop)
                if gt_crop_box_xyxy is not None:
                    gt_x_offset, gt_size_proxy = _gt_follow_target(gt_crop_box_xyxy, crop.size)

        original_panel = _annotate_original_panel(
            image=image,
            crop=crop,
            prediction=prediction,
            gt_box_xyxy=gt_box_xyxy,
        )
        crop_panel = _annotate_crop_panel(
            image=image,
            crop=crop,
            prediction=prediction,
            gt_crop_box_xyxy=gt_crop_box_xyxy,
        )
        canvas = _compose_output(
            image_path=image_path,
            original_panel=original_panel,
            crop_panel=crop_panel,
            prediction=prediction,
            crop=crop,
            gt_box_xyxy=gt_box_xyxy,
            gt_crop_box_xyxy=gt_crop_box_xyxy,
        )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_output_path)

    return OverlayResult(
        sample_dir=str(sample_dir_path),
        overlay_path=str(resolved_output_path),
        image_path=str(image_path),
        raw_values=list(prediction.raw_values),
        x_offset=float(prediction.x_offset),
        size_proxy=float(prediction.size_proxy),
        visibility_confidence=float(prediction.confidence),
        visible=bool(prediction.visible),
        gt_x_offset=None if gt_x_offset is None else float(gt_x_offset),
        gt_size_proxy=None if gt_size_proxy is None else float(gt_size_proxy),
    )


def main() -> int:
    args = parse_args()

    result = generate_overlay_for_sample(
        sample_dir=args.sample_dir,
        annotations_path=args.annotations,
        output_path=args.output,
        q_scale=args.q_scale,
        vis_threshold=args.vis_threshold,
    )

    print(f"saved_overlay={result.overlay_path}")
    print(f"image_path={result.image_path}")
    print(f"raw_values={result.raw_values}")
    print(f"x_offset={result.x_offset:+.6f}")
    print(f"size_proxy={result.size_proxy:.6f}")
    print(f"visibility_confidence={result.visibility_confidence:.6f}")
    if result.gt_x_offset is not None:
        print(f"gt_x_offset={result.gt_x_offset:+.6f}")
        print(f"gt_size_proxy={result.gt_size_proxy:.6f}")
    else:
        print("gt_x_offset=unavailable")
        print("gt_size_proxy=unavailable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

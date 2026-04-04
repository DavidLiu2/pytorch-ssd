#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.coco_follow_regression import compute_follow_target  # noqa: E402
from utils.follow_task import (  # noqa: E402
    FOLLOW_MODEL_TYPES,
    SIZE_BUCKET4_EDGES,
    XBIN9_EDGES,
    follow_model_default_head_type,
    get_follow_head_spec,
    resolve_follow_head_type,
)
from utils.transforms import get_val_transforms  # noqa: E402


DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_IMAGE_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ordered calibration manifest for follow models using deployment-matched "
            "center-crop preprocessing plus simple hard-negative, lighting, and decision-boundary heuristics."
        )
    )
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-type",
        choices=list(FOLLOW_MODEL_TYPES),
        default="plain_follow",
    )
    parser.add_argument("--follow-head-type", default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1])
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-quota", type=float, default=0.25)
    parser.add_argument("--boundary-quota", type=float, default=0.35)
    parser.add_argument("--lighting-quota", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def discover_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def extract_image_id(image_path: Path) -> int:
    match = re.search(r"(\d{12})", image_path.stem)
    if match:
        return int(match.group(1))
    digits = re.sub(r"\D+", "", image_path.stem)
    if digits:
        return int(digits)
    raise ValueError(f"Could not extract COCO image id from {image_path.name}")


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


def largest_valid_box(boxes: torch.Tensor) -> torch.Tensor | None:
    if boxes.numel() == 0:
        return None
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    areas = widths * heights
    valid = areas > 0.0
    if not torch.any(valid):
        return None
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    best_local = int(torch.argmax(areas[valid]).item())
    return boxes[int(valid_indices[best_local].item())].to(dtype=torch.float32)


def box_area(box: torch.Tensor | None) -> float:
    if box is None or box.numel() == 0:
        return 0.0
    width = max(float(box[2].item() - box[0].item()), 0.0)
    height = max(float(box[3].item() - box[1].item()), 0.0)
    return float(width * height)


def tensor_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, float(percentile) / 100.0))


def normalized_boundary_proximity(value: float, edges: tuple[float, ...]) -> float:
    if len(edges) < 3:
        return 0.0
    internal_edges = [float(edge) for edge in edges[1:-1]]
    if not internal_edges:
        return 0.0
    min_distance = min(abs(float(value) - edge) for edge in internal_edges)
    min_width = min(float(edges[idx + 1] - edges[idx]) for idx in range(len(edges) - 1))
    half_width = max(min_width * 0.5, 1e-12)
    return float(max(0.0, 1.0 - (min_distance / half_width)))


def lighting_scores(image_values: np.ndarray) -> dict[str, float]:
    if image_values.size == 0:
        return {
            "brightness_extreme": 0.0,
            "low_contrast": 0.0,
            "high_contrast": 0.0,
            "dark_fraction": 0.0,
            "bright_fraction": 0.0,
            "difficulty": 0.0,
        }
    mean = float(np.mean(image_values))
    std = float(np.std(image_values))
    dark_fraction = float(np.mean(image_values <= 0.05))
    bright_fraction = float(np.mean(image_values >= 0.95))
    brightness_extreme = float(min(abs(mean - 0.5) / 0.35, 1.0))
    low_contrast = float(min(max(0.12 - std, 0.0) / 0.12, 1.0))
    high_contrast = float(min(max(std - 0.32, 0.0) / 0.32, 1.0))
    difficulty = max(
        brightness_extreme,
        low_contrast,
        high_contrast,
        dark_fraction,
        bright_fraction,
    )
    return {
        "brightness_extreme": brightness_extreme,
        "low_contrast": low_contrast,
        "high_contrast": high_contrast,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "difficulty": float(difficulty),
    }


def build_sample_row(
    image_path: Path,
    annotations: AnnotationIndex,
    *,
    model_type: str,
    follow_head_type: str,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    follow_head_spec = get_follow_head_spec(follow_head_type, model_type=model_type)
    image_id = extract_image_id(image_path)
    original_boxes = annotations.boxes_for_image(image_id)
    original_best_box = largest_valid_box(original_boxes)

    target = {
        "boxes": original_boxes.clone(),
        "labels": torch.ones((original_boxes.shape[0],), dtype=torch.int64),
        "area": torch.zeros((original_boxes.shape[0],), dtype=torch.float32),
        "iscrowd": torch.zeros((original_boxes.shape[0],), dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "true_no_person": torch.tensor([1 if original_boxes.numel() == 0 else 0], dtype=torch.int64),
    }
    if original_boxes.numel() > 0:
        widths = (original_boxes[:, 2] - original_boxes[:, 0]).clamp_min(0.0)
        heights = (original_boxes[:, 3] - original_boxes[:, 1]).clamp_min(0.0)
        target["area"] = widths * heights

    transform = get_val_transforms(
        model_type=model_type,
        input_channels=1,
        image_size=image_size,
    )
    with Image.open(image_path) as image:
        x_float, transformed = transform(image.convert("L"), target)

    follow_target, cropped_best_box = compute_follow_target(
        transformed["boxes"],
        image_height=int(x_float.shape[-2]),
        image_width=int(x_float.shape[-1]),
    )
    image_values = np.asarray(x_float.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    stats = {
        "mean": float(np.mean(image_values)) if image_values.size else 0.0,
        "std": float(np.std(image_values)) if image_values.size else 0.0,
        "min": float(np.min(image_values)) if image_values.size else 0.0,
        "max": float(np.max(image_values)) if image_values.size else 0.0,
        "p01": tensor_percentile(image_values, 1.0),
        "p99": tensor_percentile(image_values, 99.0),
        "p99_9": tensor_percentile(image_values, 99.9),
    }
    lighting = lighting_scores(image_values)

    visible = bool(float(follow_target[2].item()) > 0.5)
    true_no_person = bool(int(transformed["true_no_person"].view(-1)[0].item()) > 0)
    crop_negative = bool((not true_no_person) and (not visible) and original_best_box is not None)

    x_boundary_proximity = 0.0
    size_boundary_proximity = 0.0
    if visible:
        x_boundary_proximity = normalized_boundary_proximity(float(follow_target[0].item()), XBIN9_EDGES)
        if follow_head_spec.size_mode == "size_bucket4":
            size_boundary_proximity = normalized_boundary_proximity(float(follow_target[1].item()), SIZE_BUCKET4_EDGES)

    visibility_boundary_proximity = 0.0
    if original_best_box is not None:
        original_area = box_area(original_best_box)
        cropped_area = box_area(cropped_best_box if visible else None)
        if crop_negative:
            visibility_boundary_proximity = 1.0
        elif visible and original_area > 0.0:
            visibility_boundary_proximity = float(
                min(max(1.0 - (cropped_area / max(original_area, 1e-12)), 0.0), 1.0)
            )

    priority_score = 1.0
    if true_no_person:
        priority_score += 2.0
    if crop_negative:
        priority_score += 2.5
    if visible:
        priority_score += 0.25
    priority_score += 2.5 * float(x_boundary_proximity)
    priority_score += 1.8 * float(size_boundary_proximity)
    priority_score += 1.6 * float(visibility_boundary_proximity)
    priority_score += 1.2 * float(lighting["difficulty"])
    priority_score += 0.6 * float(max(lighting["dark_fraction"], lighting["bright_fraction"]))

    tags: list[str] = ["deployment_crop"]
    if true_no_person:
        tags.append("true_no_person")
    if crop_negative:
        tags.append("crop_negative")
    if visible:
        tags.append("visible")
    if x_boundary_proximity >= 0.50:
        tags.append("x_boundary")
    if size_boundary_proximity >= 0.50:
        tags.append("size_boundary")
    if visibility_boundary_proximity >= 0.50:
        tags.append("visibility_boundary")
    if lighting["difficulty"] >= 0.50:
        tags.append("difficult_lighting")
    if lighting["dark_fraction"] >= 0.10 or stats["mean"] <= 0.20:
        tags.append("low_light")
    if lighting["bright_fraction"] >= 0.10 or stats["mean"] >= 0.80:
        tags.append("high_light")
    if lighting["low_contrast"] >= 0.50:
        tags.append("low_contrast")
    if lighting["high_contrast"] >= 0.50:
        tags.append("high_contrast")

    top_reasons = []
    if crop_negative:
        top_reasons.append("crop_negative")
    if true_no_person:
        top_reasons.append("true_no_person")
    if x_boundary_proximity >= 0.50:
        top_reasons.append("x_boundary")
    if size_boundary_proximity >= 0.50:
        top_reasons.append("size_boundary")
    if visibility_boundary_proximity >= 0.50:
        top_reasons.append("visibility_boundary")
    if lighting["difficulty"] >= 0.50:
        top_reasons.append("difficult_lighting")

    return {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "source_path": str(image_path.resolve()),
        "image_id": int(image_id),
        "follow_target": [
            float(follow_target[0].item()),
            float(follow_target[1].item()),
            float(follow_target[2].item()),
        ],
        "true_no_person": true_no_person,
        "crop_negative": crop_negative,
        "visible": visible,
        "priority_score": float(priority_score),
        "selection_reason": ", ".join(top_reasons) if top_reasons else "general_coverage",
        "decision_boundary_scores": {
            "x_boundary_proximity": float(x_boundary_proximity),
            "size_boundary_proximity": float(size_boundary_proximity),
            "visibility_boundary_proximity": float(visibility_boundary_proximity),
        },
        "image_stats": {
            **stats,
            **lighting,
        },
        "tags": tags,
    }


def pick_ordered_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    negative_quota: float,
    boundary_quota: float,
    lighting_quota: float,
) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            str(row.get("image_name") or row.get("source_path") or ""),
        ),
    )
    used_paths: set[str] = set()
    selected: list[dict[str, Any]] = []

    def take(predicate, count: int) -> None:
        if count <= 0:
            return
        for row in sorted_rows:
            if len(selected) >= target_count:
                return
            source_path = str(row.get("source_path"))
            if source_path in used_paths or not predicate(row):
                continue
            selected.append(row)
            used_paths.add(source_path)
            if sum(1 for item in selected if predicate(item)) >= count:
                return

    negative_count = int(math.ceil(float(target_count) * max(float(negative_quota), 0.0)))
    boundary_count = int(math.ceil(float(target_count) * max(float(boundary_quota), 0.0)))
    lighting_count = int(math.ceil(float(target_count) * max(float(lighting_quota), 0.0)))

    take(lambda row: bool(row.get("true_no_person")) or bool(row.get("crop_negative")), negative_count)
    take(
        lambda row: max(
            float(((row.get("decision_boundary_scores") or {}).get("x_boundary_proximity")) or 0.0),
            float(((row.get("decision_boundary_scores") or {}).get("size_boundary_proximity")) or 0.0),
            float(((row.get("decision_boundary_scores") or {}).get("visibility_boundary_proximity")) or 0.0),
        ) >= 0.50,
        boundary_count,
    )
    take(lambda row: float(((row.get("image_stats") or {}).get("difficulty")) or 0.0) >= 0.50, lighting_count)

    for row in sorted_rows:
        if len(selected) >= target_count:
            break
        source_path = str(row.get("source_path"))
        if source_path in used_paths:
            continue
        selected.append(row)
        used_paths.add(source_path)

    ordered = list(selected)
    ordered.extend(row for row in sorted_rows if str(row.get("source_path")) not in used_paths)
    for index, row in enumerate(ordered):
        row["selected_rank"] = int(index) if index < len(selected) else None
    return ordered


def build_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Follow Calibration Manifest",
        "",
        f"- model_type: `{payload.get('model_type')}`",
        f"- follow_head_type: `{payload.get('follow_head_type')}`",
        f"- image_dir: `{payload.get('image_dir')}`",
        f"- annotations: `{payload.get('annotations')}`",
        f"- total_images: `{payload.get('total_images')}`",
        f"- target_count: `{payload.get('target_count')}`",
        "",
        "## Selected Tag Counts",
    ]
    tag_counts = payload.get("selected_tag_counts") or {}
    for tag, count in sorted(tag_counts.items()):
        lines.append(f"- {tag}: `{count}`")
    lines.extend(
        [
            "",
            "## Top Samples",
        ]
    )
    for row in (payload.get("ordered_samples") or [])[: min(12, int(payload.get("target_count") or 0))]:
        lines.append(
            "- rank=`{}` priority=`{:.4f}` tags=`{}` image=`{}`".format(
                row.get("selected_rank"),
                float(row.get("priority_score") or 0.0),
                ",".join(row.get("tags") or []),
                row.get("image_name"),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    requested_head_type = args.follow_head_type or follow_model_default_head_type(args.model_type)
    follow_head_type = resolve_follow_head_type(requested_head_type, model_type=args.model_type)
    image_size = (int(args.height), int(args.width))
    annotations = AnnotationIndex(annotations_path)
    image_paths = discover_images(image_dir)
    if int(args.max_images) > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {image_dir}")
    random.Random(int(args.seed)).shuffle(image_paths)

    rows = [
        build_sample_row(
            path,
            annotations,
            model_type=args.model_type,
            follow_head_type=follow_head_type,
            image_size=image_size,
        )
        for path in image_paths
    ]
    ordered_rows = pick_ordered_rows(
        rows,
        target_count=min(int(args.target_count), len(rows)),
        negative_quota=float(args.negative_quota),
        boundary_quota=float(args.boundary_quota),
        lighting_quota=float(args.lighting_quota),
    )

    selected_rows = [
        row
        for row in ordered_rows
        if row.get("selected_rank") is not None and int(row["selected_rank"]) < int(args.target_count)
    ]
    selected_tag_counts: dict[str, int] = {}
    for row in selected_rows:
        for tag in row.get("tags") or []:
            selected_tag_counts[str(tag)] = int(selected_tag_counts.get(str(tag), 0)) + 1

    payload = {
        "model_type": args.model_type,
        "follow_head_type": follow_head_type,
        "image_dir": str(image_dir),
        "annotations": str(annotations_path),
        "image_size": [int(args.height), int(args.width)],
        "input_channels": int(args.input_channels),
        "total_images": len(rows),
        "target_count": min(int(args.target_count), len(rows)),
        "selection_policy": {
            "negative_quota": float(args.negative_quota),
            "boundary_quota": float(args.boundary_quota),
            "lighting_quota": float(args.lighting_quota),
            "seed": int(args.seed),
        },
        "selected_tag_counts": dict(sorted(selected_tag_counts.items())),
        "ordered_samples": ordered_rows,
    }
    write_json(output_path, payload)
    if output_path.suffix.lower() == ".json":
        write_markdown(output_path.with_suffix(".md"), build_markdown_summary(payload))


if __name__ == "__main__":
    main()

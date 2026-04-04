#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import onnxruntime as ort
import torch

import nemo
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

from models.follow_model_factory import build_follow_model_from_checkpoint, load_checkpoint_payload  # noqa: E402
from models.quant_native_follow_net import (  # noqa: E402
    ConvBN,
    ConvBNReLU,
    DelayedActivationStem,
    ResidualDownsampleStage,
    SingleConvStage,
    StraightStage,
)
from export_nemo_quant_core import (  # noqa: E402
    apply_activation_alpha_overrides,
    bind_safe_set_eps_in,
    build_eps_dict_from_modules,
    compare_arrays_rich,
    collect_calib_samples,
    collect_module_output_samples,
    flatten_sample_tensors,
    make_json_ready,
    module_output_eps_with_source,
    precision_bits_from_value,
    resolve_dotted_module,
    seed_bn_eps_for_id,
    select_activation_alpha_by_mse,
    select_activation_alpha_by_percentile,
    simulate_activation_quantization,
    summarize_calibration_samples,
    tensor_scalar,
    tensor_stats,
    export_fp_onnx,
    maybe_fuse_quant_native_follow_for_export,
    normalize_integer_requant_tensors,
    patch_model_to_graph_compat,
    prepare_quant_native_follow_qd,
    run_activation_calibration,
    semantic_output,
)
from utils.coco_follow_regression import compute_follow_target  # noqa: E402
from utils.follow_task import (  # noqa: E402
    compute_follow_metrics,
    follow_output_metadata,
    summarize_follow_bin_preservation,
)
from utils.transforms import get_val_transforms  # noqa: E402


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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MODULE_CAPTURE_ORDER = (
    "stem",
    "stage1",
    "stage1_refine",
    "stage2",
    "stage2_refine",
    "stage3",
    "global_pool",
    "output_head",
)
EARLY_ACTIVATION_CAPTURE_ORDER = (
    "stem",
    "stage1",
    "stage2",
    "output_head",
)
QD_ID_OPERATOR_DRIFT_THRESHOLD = 0.01
DEFAULT_STEM_ACTIVATION_MODULE_CANDIDATES = (
    "stem.relu",
    "stem.post.relu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate quant-native follow checkpoints through float, FQ, QD, ID, and ONNX "
            "with a simplified earliest-bad-boundary workflow."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs and reports.")
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--calib-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--calib-manifest", default=None)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--id-explicit-eps-dict", action="store_true")
    parser.add_argument("--id-local-scale-module", default=None)
    parser.add_argument("--id-local-scale-factor", type=float, default=None)
    parser.add_argument(
        "--stem-activation-module",
        default="auto",
        help="PACT activation module to treat as the stem-sensitive calibration target. Use `auto` to pick the last stem ReLU.",
    )
    parser.add_argument(
        "--stem-activation-policy",
        choices=["none", "percentile", "mse"],
        default="none",
        help="Optional stem-specific activation override built from the calibration set.",
    )
    parser.add_argument(
        "--stem-activation-percentile",
        type=float,
        default=99.9,
        help="Percentile used when --stem-activation-policy=percentile.",
    )
    parser.add_argument("--qd-id-operator-threshold", type=float, default=QD_ID_OPERATOR_DRIFT_THRESHOLD)
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
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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

        self.images = {int(item["id"]): item for item in payload.get("images", [])}
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


def make_eval_sample(
    image_path: Path,
    annotations: AnnotationIndex,
    *,
    image_size: tuple[int, int],
    model_type: str,
) -> dict[str, Any]:
    image_id = extract_image_id(image_path)
    boxes = annotations.boxes_for_image(image_id)
    target = {
        "boxes": boxes,
        "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
        "area": torch.zeros((boxes.shape[0],), dtype=torch.float32),
        "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "true_no_person": torch.tensor([1 if boxes.numel() == 0 else 0], dtype=torch.int64),
    }
    if boxes.numel() > 0:
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        target["area"] = widths * heights

    transform = get_val_transforms(
        model_type=model_type,
        input_channels=1,
        image_size=image_size,
    )
    with Image.open(image_path) as image:
        x_float, transformed = transform(image.convert("L"), target)
    follow_target, largest_box = compute_follow_target(
        transformed["boxes"],
        image_height=int(x_float.shape[-2]),
        image_width=int(x_float.shape[-1]),
    )
    x_float = x_float.unsqueeze(0).to(dtype=torch.float32)
    x_staged = torch.round(torch.clamp(x_float, 0.0, 1.0) * 255.0).to(dtype=torch.float32)
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "image_id": int(image_id),
        "float_input": x_float,
        "staged_input": x_staged,
        "follow_target": follow_target.unsqueeze(0),
        "true_no_person": transformed["true_no_person"].view(1),
        "largest_box": largest_box,
    }


def build_eval_samples(
    image_dir: Path,
    annotations: AnnotationIndex,
    *,
    image_size: tuple[int, int],
    model_type: str,
) -> list[dict[str, Any]]:
    return [
        make_eval_sample(path, annotations, image_size=image_size, model_type=model_type)
        for path in discover_images(image_dir)
    ]


def build_calib_args(args: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        model_type=str(metadata["model_type"]),
        calib_dir=(str(args.calib_dir) if args.calib_dir else None),
        calib_manifest=(str(args.calib_manifest) if args.calib_manifest else None),
        calib_tensor=None,
        calib_batches=int(args.calib_batches),
        calib_seed=int(args.calib_seed),
        mean=None,
        std=None,
        input_channels=int(metadata["input_channels"]),
        eps_in=float(args.eps_in),
    )


def build_stem_activation_audit(
    model_q: torch.nn.Module,
    calib_samples: list[dict[str, Any]],
    *,
    module_name: str,
) -> dict[str, Any]:
    sample_map = collect_module_output_samples(
        model_q,
        calib_samples,
        [module_name],
        statistics_act=False,
    )
    tensors = sample_map.get(module_name) or []
    if not tensors:
        return {
            "available": False,
            "module_name": module_name,
            "reason": "No calibration samples were captured for the requested stem activation module.",
        }

    module = resolve_dotted_module(model_q, module_name)
    values = flatten_sample_tensors(tensors)
    if values.size == 0:
        return {
            "available": False,
            "module_name": module_name,
            "reason": "Captured stem activation tensors were empty.",
        }

    alpha = float(tensor_scalar(getattr(module, "alpha", None)) or 0.0)
    bits = int(precision_bits_from_value(getattr(module, "precision", None)) or 8)
    p95 = float(np.quantile(values, 0.95))
    p99 = float(np.quantile(values, 0.99))
    p99_9 = float(np.quantile(values, 0.999))
    p100 = float(np.max(values))
    outlier_ratio = None if p99 <= 0.0 else float(p100 / max(p99, 1e-12))
    current_report = (
        simulate_activation_quantization(values, alpha, bits, symmetric=False)
        if alpha > 0.0
        else None
    )
    mse_pick = select_activation_alpha_by_mse(tensors, bits, symmetric=False)
    percentile_pick = float(select_activation_alpha_by_percentile(tensors, 99.9, symmetric=False))

    return {
        "available": True,
        "module_name": module_name,
        "module_class": module.__class__.__name__,
        "precision_bits": bits,
        "current_alpha": alpha,
        "percentiles": {
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": p95,
            "p99": p99,
            "p99_9": p99_9,
            "p100": p100,
        },
        "outlier_diagnostics": {
            "max_to_p99_ratio": outlier_ratio,
            "tail_above_p99_fraction": float(np.mean(values > p99)),
            "tail_above_p99_9_fraction": float(np.mean(values > p99_9)),
            "dominant_outlier_warning": bool(
                (outlier_ratio is not None and outlier_ratio > 2.0)
                or (current_report is not None and float(current_report["clip_fraction"]) > 0.02)
            ),
        },
        "current_quantization_report": (
            None
            if current_report is None
            else {
                "mse": float(current_report["mse"]),
                "mean_abs_error": float(current_report["mean_abs_error"]),
                "max_abs_error": float(current_report["max_abs_error"]),
                "clip_fraction": float(current_report["clip_fraction"]),
                "eps_out": (
                    float(current_report["eps_out"])
                    if current_report.get("eps_out") is not None
                    else None
                ),
            }
        ),
        "recommended_alpha_candidates": {
            "mse_alpha": float(mse_pick["alpha"]),
            "percentile_99_9_alpha": percentile_pick,
        },
    }


def build_stem_activation_override(
    model_q: torch.nn.Module,
    calib_samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    module_name = str(args.stem_activation_module)
    policy = str(args.stem_activation_policy)
    if policy == "none":
        return {}, None
    sample_map = collect_module_output_samples(
        model_q,
        calib_samples,
        [module_name],
        statistics_act=False,
    )
    tensors = sample_map.get(module_name) or []
    if not tensors:
        raise RuntimeError(f"No calibration samples captured for activation module: {module_name}")
    module = resolve_dotted_module(model_q, module_name)
    bits = precision_bits_from_value(getattr(module, "precision", None)) or 8
    values = flatten_sample_tensors(tensors)
    if policy == "percentile":
        alpha = float(select_activation_alpha_by_percentile(tensors, float(args.stem_activation_percentile), symmetric=False))
        quant_report = simulate_activation_quantization(values, alpha, int(bits), symmetric=False)
        config = {
            "alpha": alpha,
            "policy_name": f"stem_percentile_{float(args.stem_activation_percentile):g}",
            "symmetric": False,
            "percentile": float(args.stem_activation_percentile),
            "quantization_report": {
                "mse": float(quant_report["mse"]),
                "mean_abs_error": float(quant_report["mean_abs_error"]),
                "max_abs_error": float(quant_report["max_abs_error"]),
                "clip_fraction": float(quant_report["clip_fraction"]),
                "eps_out": (
                    float(quant_report["eps_out"])
                    if quant_report.get("eps_out") is not None
                    else None
                ),
            },
        }
    elif policy == "mse":
        search_report = select_activation_alpha_by_mse(tensors, int(bits), symmetric=False)
        best_report = dict(search_report.get("best_report") or {})
        config = {
            "alpha": float(search_report["alpha"]),
            "policy_name": "stem_mse_alpha_search",
            "symmetric": False,
            "search_method": "mse",
            "quantization_report": {
                "mse": float(best_report.get("mse") or 0.0),
                "mean_abs_error": float(best_report.get("mean_abs_error") or 0.0),
                "max_abs_error": float(best_report.get("max_abs_error") or 0.0),
                "clip_fraction": float(best_report.get("clip_fraction") or 0.0),
                "eps_out": (
                    float(best_report["eps_out"])
                    if best_report.get("eps_out") is not None
                    else None
                ),
            },
            "search": make_json_ready(search_report.get("search") or []),
        }
    else:
        raise ValueError(f"Unsupported stem activation policy: {policy}")
    return {module_name: config}, deepcopy(config)


def resolve_stem_conv_module_name(model: torch.nn.Module) -> str:
    available = {name for name, _ in model.named_modules()}
    for candidate in ("stem.conv", "stem.pre.conv", "stem.post.conv"):
        if candidate in available:
            return candidate
    raise RuntimeError("Could not resolve a stem convolution module for the current model.")


def stem_per_channel_support_report(
    model_fq: torch.nn.Module,
    model_id: torch.nn.Module,
) -> dict[str, Any]:
    operator_name = resolve_stem_conv_module_name(model_fq)
    fq_module = resolve_dotted_module(model_fq, operator_name)
    id_module = resolve_dotted_module(model_id, operator_name)
    out_channels = int(getattr(fq_module, "out_channels", 0))
    fq_w_alpha = getattr(fq_module, "W_alpha", None)
    id_w_alpha = getattr(id_module, "W_alpha", None)
    id_eps_out_static = getattr(id_module, "eps_out_static", None)

    supported = True
    reasons: list[str] = []
    if fq_w_alpha is None or int(fq_w_alpha.numel()) != out_channels:
        supported = False
        reasons.append(
            f"PACT_Conv2d still exposes scalar W_alpha for {operator_name} in the current NeMO path."
        )
    if id_eps_out_static is None or int(id_eps_out_static.numel()) != out_channels:
        supported = False
        reasons.append(
            f"The integerized {operator_name} export path stores scalar eps_out_static instead of one output scale per channel."
        )

    return {
        "supported": supported,
        "operator_name": operator_name,
        "observed": {
            "out_channels": out_channels,
            "fq_W_alpha_shape": None if fq_w_alpha is None else list(fq_w_alpha.shape),
            "id_W_alpha_shape": None if id_w_alpha is None else list(id_w_alpha.shape),
            "id_eps_out_static_shape": None if id_eps_out_static is None else list(id_eps_out_static.shape),
        },
        "reason_if_unsupported": reasons,
    }


def build_quantized_models(
    ckpt_path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    device = torch.device("cpu")
    image_size = (int(metadata["height"]), int(metadata["width"]))
    model_fp = build_follow_model_from_checkpoint(ckpt_path, device).eval()
    model_fp = maybe_fuse_quant_native_follow_for_export(model_fp)

    patch_model_to_graph_compat()
    dummy_input = torch.randn(1, 1, image_size[0], image_size[1], device=device)
    calib_args = build_calib_args(args, metadata)
    calib_samples = collect_calib_samples(calib_args, image_size, device)
    calibration_summary = summarize_calibration_samples(
        calib_samples,
        calib_dir=(str(args.calib_dir) if args.calib_dir else None),
        calib_manifest=(str(args.calib_manifest) if args.calib_manifest else None),
        calib_tensor=None,
        calib_batches=int(args.calib_batches),
        calib_seed=int(args.calib_seed),
    )

    def calibrated_quant_model() -> torch.nn.Module:
        model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
        model_q.to(device).eval()
        model_q.change_precision(bits=int(args.bits), scale_weights=True, scale_activations=True)
        run_activation_calibration(model_q, calib_samples)
        prepare_quant_native_follow_qd(model_q, calib_samples=calib_samples)
        return model_q.eval()

    model_fq = calibrated_quant_model()
    stem_activation_module_name = resolve_stem_activation_module_name(
        model_fq,
        getattr(args, "stem_activation_module", None),
    )
    stem_activation_audit = build_stem_activation_audit(
        model_fq,
        calib_samples,
        module_name=stem_activation_module_name,
    )
    override_args = argparse.Namespace(**vars(args))
    override_args.stem_activation_module = stem_activation_module_name
    stem_override_map, stem_override_config = build_stem_activation_override(model_fq, calib_samples, override_args)
    applied_stem_override = []
    if stem_override_map:
        applied_stem_override = apply_activation_alpha_overrides(model_fq, stem_override_map)

    def calibrated_quant_model_with_overrides() -> torch.nn.Module:
        model_q = calibrated_quant_model()
        if stem_override_map:
            apply_activation_alpha_overrides(model_q, stem_override_map)
        return model_q.eval()

    model_qd = calibrated_quant_model_with_overrides()
    model_qd_qd_kwargs = prepare_quant_native_follow_qd(model_qd, calib_samples=calib_samples)
    model_qd.qd_stage(eps_in=float(args.eps_in), **model_qd_qd_kwargs)

    model_id = calibrated_quant_model_with_overrides()
    model_id_qd_kwargs = prepare_quant_native_follow_qd(model_id, calib_samples=calib_samples)
    model_id.qd_stage(eps_in=float(args.eps_in), **model_id_qd_kwargs)
    id_stage_eps = None
    explicit_eps_report: dict[str, Any] = {
        "enabled": bool(args.id_explicit_eps_dict),
        "entry_count": 0,
        "local_scale_module": args.id_local_scale_module,
        "local_scale_factor": args.id_local_scale_factor,
    }
    if bool(args.id_explicit_eps_dict) or args.id_local_scale_module:
        id_stage_eps = build_eps_dict_from_modules(model_id)
        explicit_eps_report["entry_count"] = int(len(id_stage_eps))
        if args.id_local_scale_module:
            id_stage_eps = apply_local_eps_dict_scale(
                id_stage_eps,
                module_name=str(args.id_local_scale_module),
                scale_factor=float(args.id_local_scale_factor or 1.0),
            )
        bind_safe_set_eps_in(model_id)
        seed_bn_eps_for_id(model_id)
    if id_stage_eps is None:
        model_id.id_stage()
    else:
        model_id.id_stage(eps_in=id_stage_eps)
    normalize_integer_requant_tensors(model_id)
    for param in model_id.parameters():
        param.requires_grad_(False)

    build_context = {
        "id_stage_config": explicit_eps_report,
        "calibration_summary": calibration_summary,
        "stem_activation_audit": stem_activation_audit,
        "stem_activation_override": {
            "module_name": stem_activation_module_name,
            "policy": str(args.stem_activation_policy),
            "config": make_json_ready(stem_override_config),
            "applied_report": make_json_ready(applied_stem_override),
        },
        "stem_per_channel_support": stem_per_channel_support_report(model_fq, model_id),
        "preprocessing_contract": {
            "transform": "center_crop_square -> resize(128x128) -> grayscale",
            "staged_input": "round(clamp(x_float, 0, 1) * 255)",
            "input_channels": int(metadata["input_channels"]),
            "image_size": list(image_size),
        },
    }
    return model_fp, model_fq.eval(), model_qd.eval(), model_id.eval(), build_context


def eps_dict_aliases(module_name: str) -> tuple[str, ...]:
    base = str(module_name)
    return (
        base,
        f"ssd.{base}",
        f"module.{base}",
        f"ssd.module.{base}",
    )


def apply_local_eps_dict_scale(
    eps_dict: dict[str, torch.Tensor],
    *,
    module_name: str,
    scale_factor: float,
) -> dict[str, torch.Tensor]:
    scaled = {
        key: (value.detach().clone() if torch.is_tensor(value) else torch.tensor(float(value), dtype=torch.float32))
        for key, value in eps_dict.items()
    }
    matched = False
    for key in eps_dict_aliases(module_name):
        if key not in scaled:
            continue
        tensor = scaled[key]
        scaled[key] = torch.as_tensor(
            float(tensor_scalar(tensor) or 0.0) * float(scale_factor),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        matched = True
    if not matched:
        raise KeyError(f"Module `{module_name}` does not appear in the ID eps_dict.")
    return scaled


def plain_follow_operator_module_names(model: torch.nn.Module) -> list[str]:
    available = {name for name, _ in model.named_modules()}
    ordered: list[str] = []

    def extend_conv_block(prefix: str) -> None:
        for suffix in ("conv", "bn", "relu"):
            candidate = f"{prefix}.{suffix}"
            if candidate in available:
                ordered.append(candidate)

    for stem_name in (
        "stem.conv",
        "stem.bn",
        "stem.relu",
        "stem.pre.conv",
        "stem.pre.bn",
        "stem.post.conv",
        "stem.post.bn",
        "stem.post.relu",
    ):
        if stem_name in available:
            ordered.append(stem_name)
    for stage_name in ("stage1", "stage2", "stage3"):
        extend_conv_block(f"{stage_name}.conv")
        extend_conv_block(f"{stage_name}.downsample")
        extend_conv_block(f"{stage_name}.refine")
        extend_conv_block(f"{stage_name}.main_conv1")
        for suffix in ("conv", "bn"):
            candidate = f"{stage_name}.main_conv2.{suffix}"
            if candidate in available:
                ordered.append(candidate)
        for suffix in ("conv", "bn"):
            candidate = f"{stage_name}.skip_proj.{suffix}"
            if candidate in available:
                ordered.append(candidate)
    for stage_name in ("stage1_refine", "stage2_refine"):
        extend_conv_block(stage_name)
    for tail_name in ("global_pool", "output_head"):
        if tail_name in available:
            ordered.append(tail_name)
    return ordered


def resolve_stem_activation_module_name(
    model: torch.nn.Module,
    requested_module_name: str | None,
) -> str:
    available = {name for name, _ in model.named_modules()}
    requested = str(requested_module_name or "").strip()
    if requested and requested.lower() != "auto":
        if requested not in available:
            raise KeyError(f"Stem activation module `{requested}` does not exist in the current model.")
        return requested

    for candidate in DEFAULT_STEM_ACTIVATION_MODULE_CANDIDATES:
        if candidate in available:
            return candidate

    fallback = sorted(
        name
        for name in available
        if name.startswith("stem.") and name.endswith(".relu")
    )
    if fallback:
        return fallback[-1]
    raise RuntimeError("Could not resolve a stem activation module for the current model.")


def run_plain_follow_pytorch_probe(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    module_names: list[str],
) -> dict[str, np.ndarray]:
    captures: dict[str, np.ndarray] = {}
    handles = []
    name_to_module = dict(model.named_modules())

    def capture_input(alias: str):
        def hook(_module, inputs):
            if inputs and f"{alias}__input" not in captures:
                captures[f"{alias}__input"] = np.asarray(inputs[0].detach().cpu().numpy(), dtype=np.float64)

        return hook

    def capture_output(alias: str):
        def hook(_module, _inputs, output):
            captures[f"{alias}__output"] = np.asarray(output.detach().cpu().numpy(), dtype=np.float64)

        return hook

    for module_name in module_names:
        module = name_to_module[module_name]
        handles.append(module.register_forward_pre_hook(capture_input(module_name)))
        handles.append(module.register_forward_hook(capture_output(module_name)))

    with torch.no_grad():
        output = model(input_tensor)

    for handle in handles:
        handle.remove()

    captures["model_output"] = np.asarray(output.detach().cpu().numpy(), dtype=np.float64)
    return captures


def shift_from_divisor(divisor: float | None) -> int | None:
    if divisor in (None, 0):
        return None
    value = float(divisor)
    if value <= 0.0:
        return None
    rounded = round(math.log2(value))
    if abs((2.0**rounded) - value) > 1e-6:
        return None
    return int(rounded)


def plain_follow_operator_context(
    model: torch.nn.Module,
    *,
    input_eps: float,
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    current_eps = float(input_eps)
    for module_name in plain_follow_operator_module_names(model):
        module = resolve_dotted_module(model, module_name)
        explicit_eps_in = tensor_scalar(getattr(module, "eps_in", None))
        eps_in = explicit_eps_in if explicit_eps_in is not None else current_eps
        eps_out_report = module_output_eps_with_source(module, eps_in)
        eps_out = None if eps_out_report is None else float(eps_out_report["value"])
        eps_out_source = None if eps_out_report is None else str(eps_out_report["source"])
        if eps_out is None and isinstance(module, torch.nn.AvgPool2d):
            eps_out = float(eps_in)
            eps_out_source = "avg_pool_passthrough_fallback"
        if eps_out is None and module_name == "output_head":
            eps_out = 1.0 / 32768.0
            eps_out_source = "final_output_fallback"
        context[module_name] = {
            "module_name": module_name,
            "module_class": module.__class__.__name__,
            "eps_in": eps_in,
            "eps_out": eps_out,
            "eps_out_source": eps_out_source,
            "D": tensor_scalar(getattr(module, "D", None)),
            "shift": shift_from_divisor(tensor_scalar(getattr(module, "D", None))),
            "mul": make_json_ready(getattr(module, "mul", None)),
        }
        if eps_out not in (None, 0.0):
            current_eps = float(eps_out)
    return context


def scale_control_module_name(
    model: torch.nn.Module,
    module_name: str,
    ordered_names: list[str],
) -> str | None:
    module = resolve_dotted_module(model, module_name)
    if hasattr(module, "eps_in") or hasattr(module, "eps_in_list"):
        return module_name
    try:
        index = ordered_names.index(module_name)
    except ValueError:
        return None
    for next_name in ordered_names[index + 1 :]:
        next_module = resolve_dotted_module(model, next_name)
        if hasattr(next_module, "eps_in") or hasattr(next_module, "eps_in_list"):
            return next_name
    return None


def qd_operator_output_semantic(raw_output: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(raw_output, dtype=np.float64)
    module_class = str(ctx.get("module_class") or "")
    if module_class in {"PACT_Conv2d", "PACT_Conv1d", "PACT_Linear", "PACT_QuantizedBatchNormNd"}:
        eps_in = ctx.get("eps_in")
        if eps_in not in (None, 0.0):
            return arr * float(eps_in)
    return arr


def id_operator_output_semantic(raw_output: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(raw_output, dtype=np.float64)
    eps_out = ctx.get("eps_out")
    if eps_out not in (None, 0.0):
        return arr * float(eps_out)
    return arr


def build_qd_id_operator_report(
    model_qd: torch.nn.Module,
    model_id: torch.nn.Module,
    samples: list[dict[str, Any]],
    *,
    input_eps: float,
    threshold: float,
) -> dict[str, Any]:
    module_names = plain_follow_operator_module_names(model_id)
    context_map = plain_follow_operator_context(model_id, input_eps=float(input_eps))
    module_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in module_names}
    worst_sample = None
    worst_sample_score = -1.0

    for sample in samples:
        qd_probe = run_plain_follow_pytorch_probe(model_qd, sample["staged_input"], module_names)
        id_probe = run_plain_follow_pytorch_probe(model_id, sample["staged_input"], module_names)
        qd_output = np.asarray(semantic_output(qd_probe["model_output"], "qd"), dtype=np.float64)
        id_output = np.asarray(semantic_output(id_probe["model_output"], "id"), dtype=np.float64)
        final_output_drift = compare_arrays_rich(qd_output, id_output)
        sample_score = float(final_output_drift.get("mean_abs_diff") or 0.0)
        if sample_score > worst_sample_score:
            worst_sample_score = sample_score
            worst_sample = {
                "image_name": sample["image_name"],
                "image_path": sample["image_path"],
                "final_output_drift": final_output_drift,
            }

        qd_semantic_outputs = {
            module_name: qd_operator_output_semantic(qd_probe[f"{module_name}__output"], context_map[module_name])
            for module_name in module_names
        }
        id_semantic_outputs = {
            module_name: id_operator_output_semantic(id_probe[f"{module_name}__output"], context_map[module_name])
            for module_name in module_names
        }
        previous_qd = np.asarray(sample["staged_input"].detach().cpu().numpy(), dtype=np.float64) * float(input_eps)
        previous_id = previous_qd.copy()
        for module_name in module_names:
            qd_output_semantic = qd_semantic_outputs[module_name]
            id_output_semantic = id_semantic_outputs[module_name]
            input_drift = compare_arrays_rich(previous_qd, previous_id)
            output_drift = compare_arrays_rich(qd_output_semantic, id_output_semantic)
            module_rows[module_name].append(
                {
                    "image_name": sample["image_name"],
                    "input_drift": input_drift,
                    "output_drift": output_drift,
                }
            )
            previous_qd = qd_output_semantic
            previous_id = id_output_semantic

    focus_sample_name = None if worst_sample is None else str(worst_sample["image_name"])
    focus_rows = []
    if focus_sample_name is not None:
        focus_sample = next(sample for sample in samples if sample["image_name"] == focus_sample_name)
        qd_probe = run_plain_follow_pytorch_probe(model_qd, focus_sample["staged_input"], module_names)
        id_probe = run_plain_follow_pytorch_probe(model_id, focus_sample["staged_input"], module_names)
        qd_semantic_outputs = {
            module_name: qd_operator_output_semantic(qd_probe[f"{module_name}__output"], context_map[module_name])
            for module_name in module_names
        }
        id_semantic_outputs = {
            module_name: id_operator_output_semantic(id_probe[f"{module_name}__output"], context_map[module_name])
            for module_name in module_names
        }
        previous_qd = np.asarray(focus_sample["staged_input"].detach().cpu().numpy(), dtype=np.float64) * float(input_eps)
        previous_id = previous_qd.copy()
        for module_name in module_names:
            ctx = context_map[module_name]
            qd_output_semantic = qd_semantic_outputs[module_name]
            id_output_semantic = id_semantic_outputs[module_name]
            focus_rows.append(
                {
                    "module_name": module_name,
                    "module_class": ctx["module_class"],
                    "scale_control_module": scale_control_module_name(model_id, module_name, module_names),
                    "eps_in": ctx.get("eps_in"),
                    "eps_out": ctx.get("eps_out"),
                    "eps_out_source": ctx.get("eps_out_source"),
                    "D": ctx.get("D"),
                    "shift": ctx.get("shift"),
                    "mul": ctx.get("mul"),
                    "qd_input_stats": tensor_stats(previous_qd),
                    "id_input_stats": tensor_stats(previous_id),
                    "qd_output_stats": tensor_stats(qd_output_semantic),
                    "id_output_stats": tensor_stats(id_output_semantic),
                    "input_drift": compare_arrays_rich(previous_qd, previous_id),
                    "output_drift": compare_arrays_rich(qd_output_semantic, id_output_semantic),
                }
            )
            previous_qd = qd_output_semantic
            previous_id = id_output_semantic

    rows = []
    for module_name in module_names:
        records = module_rows[module_name]
        ctx = context_map[module_name]
        output_means = [float((item["output_drift"].get("mean_abs_diff")) or 0.0) for item in records]
        input_means = [float((item["input_drift"].get("mean_abs_diff")) or 0.0) for item in records]
        worst_record = max(records, key=lambda item: float((item["output_drift"].get("mean_abs_diff")) or -1.0))
        rows.append(
            {
                "module_name": module_name,
                "module_class": ctx["module_class"],
                "scale_control_module": scale_control_module_name(model_id, module_name, module_names),
                "eps_in": ctx.get("eps_in"),
                "eps_out": ctx.get("eps_out"),
                "eps_out_source": ctx.get("eps_out_source"),
                "D": ctx.get("D"),
                "shift": ctx.get("shift"),
                "mul": ctx.get("mul"),
                "input_mean_abs_diff_mean": float(np.mean(input_means)) if input_means else None,
                "output_mean_abs_diff_mean": float(np.mean(output_means)) if output_means else None,
                "drift_gain_mean_abs_diff": (
                    float(np.mean(output_means) - np.mean(input_means)) if output_means and input_means else None
                ),
                "worst_sample": {
                    "image_name": worst_record["image_name"],
                    "output_drift": worst_record["output_drift"],
                },
            }
        )

    first_bad = None
    for row in rows:
        output_mean = float(row.get("output_mean_abs_diff_mean") or 0.0)
        input_mean = float(row.get("input_mean_abs_diff_mean") or 0.0)
        gain = output_mean - input_mean
        if output_mean >= float(threshold) and gain >= float(threshold) * 0.25:
            first_bad = row
            break
    if first_bad is None and rows:
        first_bad = max(rows, key=lambda row: float(row.get("output_mean_abs_diff_mean") or -1.0))

    return {
        "operator_order": module_names,
        "operator_threshold": float(threshold),
        "focus_sample": worst_sample,
        "rows": rows,
        "focus_rows": focus_rows,
        "first_bad_operator": first_bad,
    }


def build_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def export_id_onnx(model_id: torch.nn.Module, output_path: Path, opset_version: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randint(0, 256, size=(1, 1, 128, 128), dtype=torch.int32).to(dtype=torch.float32)
    export_fp_onnx(
        model=model_id,
        dummy_input=dummy_input,
        out_path=output_path,
        opset_version=opset_version,
    )


def run_cleanup_pipeline(src_onnx: Path, output_dir: Path) -> Path:
    no_affine = output_dir / "model_id_noaffine.onnx"
    no_transpose = output_dir / "model_id_notranspose.onnx"
    no_min = output_dir / "model_id_nomin.onnx"
    no_fake_quant = output_dir / "model_id_nofakequant.onnx"
    final_onnx = output_dir / "model_id_dory.onnx"
    scripts = [
        (PROJECT_DIR / "export" / "strip_affine_mul_add.py", src_onnx, no_affine),
        (PROJECT_DIR / "export" / "strip_transpose.py", no_affine, no_transpose),
        (PROJECT_DIR / "export" / "strip_min.py", no_transpose, no_min),
        (PROJECT_DIR / "export" / "strip_fake_quant.py", no_min, no_fake_quant),
        (PROJECT_DIR / "export" / "simplify_onnx.py", no_fake_quant, final_onnx),
    ]
    for script, src, dst in scripts:
        subprocess.run(
            [sys.executable, str(script), str(src), str(dst)],
            cwd=str(PROJECT_DIR),
            check=True,
        )
    return final_onnx


def run_compatibility_check(
    *,
    ckpt_path: Path,
    output_dir: Path,
    model_type: str,
    onnx_path: Path,
    dory_onnx_path: Path,
    calib_dir: Path | None,
    calib_manifest: Path | None,
) -> dict[str, Any]:
    report_path = output_dir / "compatibility_report.json"
    command = [
        sys.executable,
        str(PROJECT_DIR / "export" / "check_model_compatibility.py"),
        "--mode",
        "both",
        "--model-type",
        model_type,
        "--ckpt",
        str(ckpt_path),
        "--onnx",
        str(onnx_path),
        "--dory-onnx",
        str(dory_onnx_path),
        "--calib-batches",
        "4",
        "--compat-calib-batches",
        "4",
        "--report-json",
        str(report_path),
    ]
    if calib_dir is not None:
        command.extend(["--calib-dir", str(calib_dir)])
    if calib_manifest is not None:
        command.extend(["--calib-manifest", str(calib_manifest)])
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report_path": str(report_path),
    }
    if report_path.is_file():
        payload["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return payload


def named_capture_modules(model: torch.nn.Module) -> list[str]:
    available = {
        name
        for name, module in model.named_modules()
        if name
        and isinstance(
            module,
            (
                ConvBN,
                ConvBNReLU,
                DelayedActivationStem,
                SingleConvStage,
                StraightStage,
                ResidualDownsampleStage,
                torch.nn.AvgPool2d,
                torch.nn.Linear,
            ),
        )
    }
    return [name for name in MODULE_CAPTURE_ORDER if name in available]


def early_activation_capture_modules(model: torch.nn.Module) -> list[str]:
    available = {name for name, _module in model.named_modules()}
    return [name for name in EARLY_ACTIVATION_CAPTURE_ORDER if name in available]


def capture_named_outputs(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    module_names: list[str],
) -> dict[str, np.ndarray]:
    captures: dict[str, np.ndarray] = {}
    handles = []
    name_to_module = dict(model.named_modules())

    for module_name in module_names:
        module = name_to_module[module_name]

        def hook_factory(alias: str):
            def hook(_module, _inputs, output):
                captures[alias] = np.asarray(output.detach().cpu().numpy(), dtype=np.float64)

            return hook

        handles.append(module.register_forward_hook(hook_factory(module_name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for handle in handles:
        handle.remove()
    return captures


def early_activation_context(
    model: torch.nn.Module,
    module_names: list[str],
    *,
    input_eps: float,
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    current_eps = float(input_eps)
    for module_name in module_names:
        module = resolve_dotted_module(model, module_name)
        explicit_eps_in = tensor_scalar(getattr(module, "eps_in", None))
        eps_in = explicit_eps_in if explicit_eps_in is not None else current_eps
        eps_out_report = module_output_eps_with_source(module, eps_in)
        eps_out = None if eps_out_report is None else float(eps_out_report["value"])
        eps_out_source = None if eps_out_report is None else str(eps_out_report["source"])
        if eps_out is None and module_name == "output_head":
            eps_out = 1.0 / 32768.0
            eps_out_source = "final_output_fallback"
        context[module_name] = {
            "module_name": module_name,
            "module_class": module.__class__.__name__,
            "eps_in": eps_in,
            "eps_out": eps_out,
            "eps_out_source": eps_out_source,
        }
        if eps_out not in (None, 0.0):
            current_eps = float(eps_out)
    return context


def early_activation_semantic_output(
    raw_output: np.ndarray,
    *,
    stage_name: str,
    module_name: str,
    context: dict[str, Any],
) -> np.ndarray:
    arr = np.asarray(raw_output, dtype=np.float64)
    if module_name == "output_head":
        semantic_stage = "id" if stage_name in {"qd", "id", "onnx"} else stage_name
        return np.asarray(semantic_output(arr, semantic_stage), dtype=np.float64)
    if stage_name in {"fp", "fq"}:
        return arr
    scale = context.get("eps_out")
    if scale in (None, 0.0):
        scale = context.get("eps_in")
    if scale in (None, 0.0):
        return arr
    return arr * float(scale)


def flatten_capture_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    flattened: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=np.float64).reshape(-1)
        if arr.size:
            flattened.append(arr)
    if not flattened:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(flattened)


def compare_capture_lists(
    left_arrays: list[np.ndarray],
    right_arrays: list[np.ndarray],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = compare_arrays(
        flatten_capture_arrays(left_arrays),
        flatten_capture_arrays(right_arrays),
    )
    worst_sample = None
    worst_score = -1.0
    for index, (left_array, right_array) in enumerate(zip(left_arrays, right_arrays)):
        sample_drift = compare_arrays(left_array, right_array)
        score = float(sample_drift.get("mean_abs_diff") or 0.0)
        if score > worst_score:
            sample_name = None
            if index < len(samples):
                sample_name = str(samples[index].get("image_name"))
            worst_score = score
            worst_sample = {
                "index": int(index),
                "image_name": sample_name,
                "drift": sample_drift,
            }
    return {
        "aggregate": aggregate,
        "sample_count": int(min(len(left_arrays), len(right_arrays))),
        "worst_sample": worst_sample,
    }


def summarize_stage_arrays(arrays: list[np.ndarray]) -> dict[str, Any]:
    flattened = flatten_capture_arrays(arrays)
    return {
        "sample_count": int(len(arrays)),
        "stats": None if flattened.size == 0 else tensor_stats(flattened),
    }


def build_activation_sensitivity_report(
    model_fp: torch.nn.Module,
    model_fq: torch.nn.Module,
    model_qd: torch.nn.Module,
    model_id: torch.nn.Module,
    samples: list[dict[str, Any]],
    *,
    input_eps: float,
) -> dict[str, Any]:
    module_names = early_activation_capture_modules(model_fp)
    stage_specs = {
        "fp": (model_fp, "float_input"),
        "fq": (model_fq, "float_input"),
        "qd": (model_qd, "staged_input"),
        "id": (model_id, "staged_input"),
    }
    stage_contexts = {
        "fp": {},
        "fq": {},
        "qd": early_activation_context(model_qd, module_names, input_eps=float(input_eps)),
        "id": early_activation_context(model_id, module_names, input_eps=float(input_eps)),
    }
    stage_captures: dict[str, dict[str, list[np.ndarray]]] = {
        stage_name: {module_name: [] for module_name in module_names}
        for stage_name in stage_specs
    }

    for sample in samples:
        for stage_name, (model, input_key) in stage_specs.items():
            raw_outputs = capture_named_outputs(model, sample[input_key], module_names)
            for module_name in module_names:
                context = stage_contexts.get(stage_name, {}).get(module_name, {})
                stage_captures[stage_name][module_name].append(
                    early_activation_semantic_output(
                        raw_outputs[module_name],
                        stage_name=stage_name,
                        module_name=module_name,
                        context=context,
                    )
                )

    rows = []
    boundary_order = ("fp_to_fq", "fq_to_qd", "qd_to_id")
    for module_name in module_names:
        boundary_reports = {
            "fp_to_fq": compare_capture_lists(
                stage_captures["fp"][module_name],
                stage_captures["fq"][module_name],
                samples,
            ),
            "fq_to_qd": compare_capture_lists(
                stage_captures["fq"][module_name],
                stage_captures["qd"][module_name],
                samples,
            ),
            "qd_to_id": compare_capture_lists(
                stage_captures["qd"][module_name],
                stage_captures["id"][module_name],
                samples,
            ),
        }
        rows.append(
            {
                "module_name": module_name,
                "fp": summarize_stage_arrays(stage_captures["fp"][module_name]),
                "fq": summarize_stage_arrays(stage_captures["fq"][module_name]),
                "qd": summarize_stage_arrays(stage_captures["qd"][module_name]),
                "id": summarize_stage_arrays(stage_captures["id"][module_name]),
                **boundary_reports,
            }
        )

    boundary_rollups = {}
    for boundary_name in boundary_order:
        worst_row = None
        worst_score = -1.0
        for row in rows:
            aggregate = (row.get(boundary_name) or {}).get("aggregate") or {}
            score = float(aggregate.get("mean_abs_diff") or 0.0)
            if score > worst_score:
                worst_score = score
                worst_row = {
                    "module_name": row["module_name"],
                    "aggregate": aggregate,
                    "worst_sample": (row.get(boundary_name) or {}).get("worst_sample"),
                }
        boundary_rollups[boundary_name] = {"worst_module": worst_row}

    return {
        "module_order": module_names,
        "stage_order": list(stage_specs.keys()),
        "boundary_order": list(boundary_order),
        "rows": rows,
        "boundary_rollups": boundary_rollups,
    }


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.shape != right_arr.shape:
        return {
            "shape_left": list(left_arr.shape),
            "shape_right": list(right_arr.shape),
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "cosine_similarity": None,
        }

    diff = np.abs(left_arr - right_arr)
    cosine = None
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = float(np.dot(left_arr, right_arr) / (left_norm * right_norm))
    return {
        "shape": list(left_arr.shape),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "cosine_similarity": cosine,
    }


def boundary_threshold(boundary_name: str) -> tuple[float, float]:
    if boundary_name == "fp_to_fq":
        return 0.01, 0.999
    return 0.5, 0.995


def first_bad_module_report(
    reference_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    boundary_name: str,
) -> dict[str, Any]:
    module_names = named_capture_modules(reference_model)
    left = capture_named_outputs(reference_model, input_tensor, module_names)
    right = capture_named_outputs(candidate_model, input_tensor, module_names)
    mean_abs_threshold, cosine_threshold = boundary_threshold(boundary_name)

    rows = []
    first_bad = None
    for module_name in module_names:
        drift = compare_arrays(left[module_name], right[module_name])
        rows.append({"module_name": module_name, "drift": drift})
        cosine = drift.get("cosine_similarity")
        mean_abs = drift.get("mean_abs_diff")
        is_bad = (
            mean_abs is not None
            and mean_abs > mean_abs_threshold
            or cosine is not None
            and cosine < cosine_threshold
        )
        if first_bad is None and is_bad:
            first_bad = {"module_name": module_name, "drift": drift}

    worst = None
    if rows:
        worst = max(
            rows,
            key=lambda row: float((row["drift"].get("mean_abs_diff") or 0.0)),
        )
    return {
        "boundary_name": boundary_name,
        "module_order": module_names,
        "rows": rows,
        "first_bad": first_bad,
        "worst": worst,
    }


def boundary_is_bad(metrics: dict[str, Any]) -> bool:
    if float(metrics.get("x_value_mae") or 0.0) > 0.05:
        return True
    if float(metrics.get("size_value_mae") or 0.0) > 0.05:
        return True
    if float(metrics.get("visibility_gate_agreement") or 0.0) < 0.95:
        return True
    exact = metrics.get("x_bin_exact_match_rate")
    if exact is not None and float(exact) < 0.9:
        return True
    coarse = metrics.get("x_coarse_exact_match_rate")
    if coarse is not None and float(coarse) < 0.9:
        return True
    return False


def stage_metrics_for_subset(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    no_person: torch.Tensor,
    *,
    model_type: str,
    head_type: str,
    vis_thresh: float,
) -> dict[str, Any]:
    return compute_follow_metrics(
        outputs,
        targets,
        head_type=head_type,
        model_type=model_type,
        vis_thresh=vis_thresh,
        true_no_person=no_person,
    )


def tensor_rows(stage_rows: list[np.ndarray]) -> torch.Tensor:
    return torch.tensor(np.asarray(stage_rows, dtype=np.float32))


def build_summary_markdown(summary: dict[str, Any]) -> str:
    float_metrics = summary["float_validation"]
    rep16 = summary["datasets"]["rep16"]["onnx"]
    hard_case = summary["datasets"]["hard_case"]["onnx"]
    earliest = summary["quant_fidelity"]["earliest_bad_boundary"]
    qd_id_operator = (summary["quant_fidelity"].get("qd_to_id_operator_report") or {}).get("first_bad_operator") or {}
    pipeline = summary["pipeline_complexity"]
    calibration = summary.get("calibration") or {}
    calib_summary = calibration.get("summary") or {}
    stem_audit = calibration.get("stem_activation_audit") or {}
    stem_override = calibration.get("stem_activation_override") or {}
    stem_per_channel = calibration.get("stem_per_channel_support") or {}
    activation_sensitivity = summary.get("activation_sensitivity") or {}
    sensitivity_rollups = activation_sensitivity.get("boundary_rollups") or {}

    lines = [
        f"# {summary['candidate_name']}",
        "",
        "## Float Validation",
        f"- follow_score: `{float_metrics.get('follow_score')}`",
        f"- x_mae: `{float_metrics.get('x_mae')}`",
        f"- size_mae: `{float_metrics.get('size_mae')}`",
        f"- no_person_fp_rate: `{float_metrics.get('no_person_fp_rate')}`",
        "",
        "## Quantized Rep16",
        f"- onnx follow_score: `{rep16.get('follow_score')}`",
        f"- onnx x_mae: `{rep16.get('x_mae')}`",
        f"- onnx size_mae: `{rep16.get('size_mae')}`",
        f"- onnx no_person_fp_rate: `{rep16.get('no_person_fp_rate')}`",
        "",
        "## Hard Case",
        f"- onnx follow_score: `{hard_case.get('follow_score')}`",
        f"- onnx x_mae: `{hard_case.get('x_mae')}`",
        f"- onnx size_mae: `{hard_case.get('size_mae')}`",
        "",
        "## Quant Fidelity",
        f"- earliest bad boundary: `{(earliest or {}).get('boundary_name')}`",
        f"- earliest bad op: `{((earliest or {}).get('first_bad') or {}).get('module_name')}`",
        f"- first-bad local drift: `{(((earliest or {}).get('first_bad') or {}).get('drift') or {})}`",
        f"- qd->id first bad operator: `{qd_id_operator.get('module_name')}`",
        f"- qd->id scale control module: `{qd_id_operator.get('scale_control_module')}`",
        f"- qd->id operator output mean abs drift: `{qd_id_operator.get('output_mean_abs_diff_mean')}`",
        f"- float->onnx bin preservation: `{summary['quant_fidelity']['float_to_onnx_bin_preservation']}`",
        f"- early activation worst fp->fq: `{((sensitivity_rollups.get('fp_to_fq') or {}).get('worst_module') or {}).get('module_name')}`",
        f"- early activation worst fq->qd: `{((sensitivity_rollups.get('fq_to_qd') or {}).get('worst_module') or {}).get('module_name')}`",
        f"- early activation worst qd->id: `{((sensitivity_rollups.get('qd_to_id') or {}).get('worst_module') or {}).get('module_name')}`",
        "",
        "## ID Config",
        f"- explicit eps_dict: `{(summary.get('id_stage_config') or {}).get('enabled')}`",
        f"- eps_dict entries: `{(summary.get('id_stage_config') or {}).get('entry_count')}`",
        f"- local scale module: `{(summary.get('id_stage_config') or {}).get('local_scale_module')}`",
        f"- local scale factor: `{(summary.get('id_stage_config') or {}).get('local_scale_factor')}`",
        "",
        "## Calibration",
        f"- calibration source kinds: `{calib_summary.get('source_kinds')}`",
        f"- calibration tag counts: `{calib_summary.get('tag_counts')}`",
        f"- stem activation module: `{stem_audit.get('module_name')}`",
        f"- stem current alpha: `{stem_audit.get('current_alpha')}`",
        f"- stem outlier warning: `{((stem_audit.get('outlier_diagnostics') or {}).get('dominant_outlier_warning'))}`",
        f"- stem max/p99 ratio: `{((stem_audit.get('outlier_diagnostics') or {}).get('max_to_p99_ratio'))}`",
        f"- stem override policy: `{stem_override.get('policy')}`",
        f"- stem override config: `{stem_override.get('config')}`",
        f"- stem per-channel supported cleanly: `{stem_per_channel.get('supported')}`",
        f"- stem per-channel note: `{stem_per_channel.get('reason_if_unsupported')}`",
        "",
        "## Pipeline Complexity",
        f"- custom export patches required: `{pipeline['custom_export_patch_count']}`",
        f"- graph repair needed: `{pipeline['graph_repair_needed']}`",
        f"- head collapse needed: `{pipeline['head_collapse_needed']}`",
        f"- residual rescue needed: `{pipeline['residual_rescue_needed']}`",
        f"- onnx cleanup steps: `{pipeline['onnx_cleanup_steps']}`",
        f"- compatibility return code: `{pipeline['compatibility_returncode']}`",
    ]
    return "\n".join(lines)


def build_activation_sensitivity_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Plain Follow Early Activation Sensitivity",
        "",
        "## Boundary Rollups",
    ]
    for boundary_name in report.get("boundary_order") or []:
        worst = (((report.get("boundary_rollups") or {}).get(boundary_name) or {}).get("worst_module") or {})
        lines.append(
            f"- {boundary_name}: module=`{worst.get('module_name')}` aggregate=`{worst.get('aggregate')}`"
        )
    lines.extend(
        [
            "",
            "## Aggregate Rows",
            "",
            "| Module | fp->fq mean abs | fq->qd mean abs | qd->id mean abs | fp abs_max | fq abs_max | qd abs_max | id abs_max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("rows") or []:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                row.get("module_name"),
                float((((row.get("fp_to_fq") or {}).get("aggregate") or {}).get("mean_abs_diff")) or 0.0),
                float((((row.get("fq_to_qd") or {}).get("aggregate") or {}).get("mean_abs_diff")) or 0.0),
                float((((row.get("qd_to_id") or {}).get("aggregate") or {}).get("mean_abs_diff")) or 0.0),
                float((((row.get("fp") or {}).get("stats") or {}).get("abs_max")) or 0.0),
                float((((row.get("fq") or {}).get("stats") or {}).get("abs_max")) or 0.0),
                float((((row.get("qd") or {}).get("stats") or {}).get("abs_max")) or 0.0),
                float((((row.get("id") or {}).get("stats") or {}).get("abs_max")) or 0.0),
            )
        )
    return "\n".join(lines)


def build_qd_id_operator_markdown(report: dict[str, Any]) -> str:
    first_bad = report.get("first_bad_operator") or {}
    focus_sample = report.get("focus_sample") or {}
    lines = [
        "# Plain Follow QD -> ID Operator Report",
        "",
        f"- First bad operator: `{first_bad.get('module_name')}`",
        f"- Scale control module: `{first_bad.get('scale_control_module')}`",
        f"- Operator output mean abs drift: `{first_bad.get('output_mean_abs_diff_mean')}`",
        f"- Focus sample: `{focus_sample.get('image_name')}`",
        f"- Focus sample final output drift: `{focus_sample.get('final_output_drift')}`",
        "",
        "## Aggregate Rows",
        "",
        "| Module | Control Module | Input mean abs | Output mean abs | Gain | eps_in | eps_out |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {} | {} |".format(
                row.get("module_name"),
                row.get("scale_control_module"),
                float(row.get("input_mean_abs_diff_mean") or 0.0),
                float(row.get("output_mean_abs_diff_mean") or 0.0),
                float(row.get("drift_gain_mean_abs_diff") or 0.0),
                row.get("eps_in"),
                row.get("eps_out"),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    rep16_dir = Path(args.rep16_dir).expanduser().resolve()
    hard_case_dir = Path(args.hard_case_dir).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()

    metadata = load_checkpoint_payload(ckpt_path, torch.device("cpu"))
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata is not a dict.")
    model_type = str(metadata["model_type"])
    follow_head_type = str(metadata.get("follow_head_type") or "legacy_regression")
    image_size = (int(metadata["height"]), int(metadata["width"]))
    candidate_name = str(args.candidate_name or f"{model_type}_{follow_head_type}")

    annotations = AnnotationIndex(annotations_path)
    rep16_samples = build_eval_samples(rep16_dir, annotations, image_size=image_size, model_type=model_type)
    hard_case_names = {path.name for path in discover_images(hard_case_dir)}

    model_fp, model_fq, model_qd, model_id, quant_build_context = build_quantized_models(ckpt_path, args, metadata)
    parameter_count = int(sum(param.numel() for param in model_fp.parameters()))

    id_onnx_path = output_dir / "model_id.onnx"
    export_id_onnx(model_id, id_onnx_path, int(args.opset_version))
    dory_onnx_path = run_cleanup_pipeline(id_onnx_path, output_dir)
    compat = run_compatibility_check(
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        model_type=model_type,
        onnx_path=id_onnx_path,
        dory_onnx_path=dory_onnx_path,
        calib_dir=(Path(args.calib_dir).expanduser().resolve() if args.calib_dir else None),
        calib_manifest=(Path(args.calib_manifest).expanduser().resolve() if args.calib_manifest else None),
    )
    onnx_session = build_onnx_session(id_onnx_path)
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name

    stage_rows: dict[str, list[np.ndarray]] = {
        "fp": [],
        "fq": [],
        "qd": [],
        "id": [],
        "onnx": [],
    }
    target_rows: list[np.ndarray] = []
    no_person_rows: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample in rep16_samples:
            fp_out = model_fp(sample["float_input"]).detach().cpu().numpy()
            fq_out = model_fq(sample["float_input"]).detach().cpu().numpy()
            qd_out = model_qd(sample["staged_input"]).detach().cpu().numpy()
            id_out = model_id(sample["staged_input"]).detach().cpu().numpy()
            onnx_out = np.asarray(
                onnx_session.run(
                    [onnx_output_name],
                    {onnx_input_name: sample["staged_input"].detach().cpu().numpy()},
                )[0],
                dtype=np.float32,
            )

            stage_rows["fp"].append(np.asarray(semantic_output(fp_out, "fp"), dtype=np.float32))
            stage_rows["fq"].append(np.asarray(semantic_output(fq_out, "fq"), dtype=np.float32))
            stage_rows["qd"].append(np.asarray(semantic_output(qd_out, "qd"), dtype=np.float32))
            stage_rows["id"].append(np.asarray(semantic_output(id_out, "id"), dtype=np.float32))
            stage_rows["onnx"].append(np.asarray(semantic_output(onnx_out, "id"), dtype=np.float32))
            target_rows.append(sample["follow_target"].detach().cpu().numpy().reshape(-1))
            no_person_rows.append(sample["true_no_person"].detach().cpu().numpy().reshape(-1))
            sample_rows.append(
                {
                    "image_name": sample["image_name"],
                    "image_path": sample["image_path"],
                    "hard_case": sample["image_name"] in hard_case_names,
                }
            )

    target_tensor = torch.tensor(np.asarray(target_rows, dtype=np.float32))
    no_person_tensor = torch.tensor(np.asarray(no_person_rows, dtype=np.int64)).view(-1, 1)
    stage_tensors = {key: tensor_rows(rows) for key, rows in stage_rows.items()}

    dataset_views: dict[str, Any] = {}
    for dataset_name, index_mask in (
        ("rep16", [True] * len(sample_rows)),
        ("hard_case", [row["hard_case"] for row in sample_rows]),
    ):
        selected = [idx for idx, keep in enumerate(index_mask) if keep]
        stage_metrics = {}
        for stage_name, outputs in stage_tensors.items():
            if not selected:
                stage_metrics[stage_name] = {}
                continue
            index_tensor = torch.tensor(selected, dtype=torch.long)
            stage_metrics[stage_name] = stage_metrics_for_subset(
                outputs.index_select(0, index_tensor),
                target_tensor.index_select(0, index_tensor),
                no_person_tensor.index_select(0, index_tensor),
                model_type=model_type,
                head_type=follow_head_type,
                vis_thresh=float(args.vis_thresh),
            )
        dataset_views[dataset_name] = stage_metrics

    boundary_reports = {
        "fp_to_fq": {
            "semantic": summarize_follow_bin_preservation(
                stage_tensors["fp"],
                stage_tensors["fq"],
                head_type=follow_head_type,
                model_type=model_type,
                vis_thresh=float(args.vis_thresh),
            ),
            "local": first_bad_module_report(
                model_fp,
                model_fq,
                rep16_samples[0]["float_input"],
                boundary_name="fp_to_fq",
            ),
        },
        "qd_to_id": {
            "semantic": summarize_follow_bin_preservation(
                stage_tensors["qd"],
                stage_tensors["id"],
                head_type=follow_head_type,
                model_type=model_type,
                vis_thresh=float(args.vis_thresh),
            ),
            "local": first_bad_module_report(
                model_qd,
                model_id,
                rep16_samples[0]["staged_input"],
                boundary_name="qd_to_id",
            ),
        },
        "id_to_onnx": {
            "semantic": summarize_follow_bin_preservation(
                stage_tensors["id"],
                stage_tensors["onnx"],
                head_type=follow_head_type,
                model_type=model_type,
                vis_thresh=float(args.vis_thresh),
            ),
            "local": {
                "boundary_name": "id_to_onnx",
                "module_order": ["model_output"],
                "rows": [
                    {
                        "module_name": "model_output",
                        "drift": compare_arrays(
                            stage_tensors["id"].detach().cpu().numpy(),
                            stage_tensors["onnx"].detach().cpu().numpy(),
                        ),
                    }
                ],
                "first_bad": None,
                "worst": None,
            },
        },
    }
    boundary_reports["id_to_onnx"]["local"]["worst"] = boundary_reports["id_to_onnx"]["local"]["rows"][0]
    if boundary_is_bad(boundary_reports["id_to_onnx"]["semantic"]):
        boundary_reports["id_to_onnx"]["local"]["first_bad"] = boundary_reports["id_to_onnx"]["local"]["rows"][0]

    qd_id_operator_report = build_qd_id_operator_report(
        model_qd,
        model_id,
        rep16_samples,
        input_eps=float(args.eps_in),
        threshold=float(args.qd_id_operator_threshold),
    )
    activation_sensitivity_report = build_activation_sensitivity_report(
        model_fp,
        model_fq,
        model_qd,
        model_id,
        rep16_samples,
        input_eps=float(args.eps_in),
    )

    earliest_bad = None
    for boundary_name in ("fp_to_fq", "qd_to_id", "id_to_onnx"):
        report = boundary_reports[boundary_name]
        if boundary_is_bad(report["semantic"]):
            earliest_bad = {
                "boundary_name": boundary_name,
                "semantic": report["semantic"],
                "first_bad": report["local"].get("first_bad"),
                "worst": report["local"].get("worst"),
            }
            break

    float_validation = dict(metadata.get("val_stats") or {})
    summary = {
        "candidate_name": candidate_name,
        "checkpoint_path": str(ckpt_path),
        "model_type": model_type,
        "follow_head_type": follow_head_type,
        "output_metadata": follow_output_metadata(model_type=model_type, head_type=follow_head_type),
        "parameter_count": parameter_count,
        "float_validation": float_validation,
        "datasets": dataset_views,
        "quant_fidelity": {
            "boundaries": boundary_reports,
            "earliest_bad_boundary": earliest_bad,
            "qd_to_id_operator_report": qd_id_operator_report,
            "float_to_onnx_bin_preservation": summarize_follow_bin_preservation(
                stage_tensors["fp"],
                stage_tensors["onnx"],
                head_type=follow_head_type,
                model_type=model_type,
                vis_thresh=float(args.vis_thresh),
            ),
        },
        "id_stage_config": quant_build_context["id_stage_config"],
        "calibration": {
            "summary": quant_build_context["calibration_summary"],
            "stem_activation_audit": quant_build_context["stem_activation_audit"],
            "stem_activation_override": quant_build_context["stem_activation_override"],
            "stem_per_channel_support": quant_build_context["stem_per_channel_support"],
            "preprocessing_contract": quant_build_context["preprocessing_contract"],
        },
        "activation_sensitivity": activation_sensitivity_report,
        "pipeline_complexity": {
            "custom_export_patch_count": 0,
            "graph_repair_needed": False,
            "head_collapse_needed": False,
            "residual_rescue_needed": False,
            "onnx_cleanup_steps": 5,
            "exporter_path": "generic_nemo_quantize_pact_qd_id",
            "deployment_artifacts_cleaner_than_hybrid": True,
            "compatibility_returncode": int(compat["returncode"]),
            "compatibility_status": ((compat.get("report") or {}).get("status")),
            "compatibility_report_path": compat.get("report_path"),
        },
        "artifacts": {
            "onnx": str(id_onnx_path),
            "dory_onnx": str(dory_onnx_path),
            "compatibility": compat,
        },
        "sample_rows": sample_rows,
    }

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    write_markdown(output_dir / "summary.md", build_summary_markdown(summary))
    write_json(output_dir / "qd_id_operator_report.json", qd_id_operator_report)
    write_markdown(output_dir / "qd_id_operator_report.md", build_qd_id_operator_markdown(qd_id_operator_report))
    write_json(output_dir / "activation_sensitivity_report.json", activation_sensitivity_report)
    write_markdown(
        output_dir / "activation_sensitivity_report.md",
        build_activation_sensitivity_markdown(activation_sensitivity_report),
    )
    write_json(output_dir / "calibration_summary.json", summary["calibration"])


if __name__ == "__main__":
    main()

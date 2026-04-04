#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import nemo
import numpy as np
import onnxruntime as ort
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_nemo_quant import (  # noqa: E402
    HYBRID_FOLLOW_CONV_BIAS_ROUNDING,
    HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE,
    HYBRID_FOLLOW_INTEGER_ADD_POLICY_CANDIDATES,
    HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
    apply_activation_alpha_overrides,
    collect_calib_samples,
    collect_integer_add_branch_samples,
    collect_module_output_samples,
    compare_arrays,
    compare_arrays_rich,
    compare_decoded_hybrid_follow_outputs,
    hybrid_follow_image_to_tensor,
    hybrid_follow_output_to_decoded,
    integer_add_scale_selection_scope,
    integerize_deploy_conv_biases,
    make_json_ready,
    normalize_integer_requant_tensors,
    patch_model_to_graph_compat,
    prepare_model_fp,
    repair_hybrid_follow_fused_quant_graph,
    resolve_dotted_module,
    resolve_hybrid_follow_head_input_eps,
    run_activation_calibration,
    run_hybrid_follow_integer_add_audit,
    run_hybrid_follow_pytorch_probe,
    scalar_from_value,
    select_activation_alpha_by_mse,
    select_activation_alpha_by_percentile,
    simulate_activation_quantization,
    saturation_stats,
    semantic_output,
    stage4_1_path_quant_context,
    tensor_stats,
)

DEFAULT_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_follow_score.pth"
DEFAULT_CALIB_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"
DEFAULT_KNOWN_BAD_IMAGE = DEFAULT_CALIB_DIR / "000000493613.jpg"
DEFAULT_EVAL_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "export" / "hybrid_follow" / "quant_operator_sweep"
FINAL_OUTPUT_SCALE = 32768.0

DEFAULT_THRESHOLDS = {
    "x_abs_diff": 0.05,
    "size_abs_diff": 0.05,
    "vis_conf_abs_diff": 0.10,
}

DEPLOY_ONNX_THRESHOLDS = {
    "x_abs_diff": 0.005,
    "size_abs_diff": 0.005,
    "vis_conf_abs_diff": 0.01,
}

ACTIVATION_REGION_SPECS = {
    "conv1": {
        "label": "stage4.1.conv1 input activation",
        "activation_module": "stage4.0.out_relu",
        "operator_name": "stage4.1.conv1",
        "target_label": "stage4.1.conv1",
        "group_name": "conv1_activation",
        "patched_region": "conv1",
    },
    "conv2": {
        "label": "stage4.1.conv2 input activation",
        "activation_module": "stage4.1.relu1",
        "operator_name": "stage4.1.conv2",
        "target_label": "stage4.1.conv2",
        "group_name": "conv2_activation",
        "patched_region": "conv2",
    },
    "add_activation": {
        "label": "stage4.1.add output activation",
        "activation_module": "stage4.1.out_relu",
        "operator_name": "stage4.1.add",
        "target_label": "stage4.1.add post-requant",
        "group_name": "add_activation",
        "patched_region": "add",
    },
}

MICROBLOCK_SWEEP_LAYOUT = [
    ("conv1_only", ["conv1"]),
    ("conv2_only", ["conv2"]),
    ("add_only", ["add"]),
    ("conv1_add", ["conv1", "add"]),
    ("conv1_conv2_add", ["conv1", "conv2", "add"]),
]


@dataclass
class PolicySpec:
    name: str
    description: str
    operator_name: str
    family: str
    group_name: str = "default"
    target_label: str | None = None
    patched_regions: list[str] = field(default_factory=list)
    activation_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    integer_add_operator_overrides: dict[str, Any] = field(default_factory=dict)
    conv_bias_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_patch_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostic_only: bool = False
    search_context: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hybrid_follow two-loop quant drift sweep: identify the earliest "
            "material FQ->ID drift on a known sample, compare a small operator-specific "
            "policy set locally, and then score the winner on a representative batch."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--known-bad-image", default=str(DEFAULT_KNOWN_BAD_IMAGE))
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-val-summary", default=None,
                        help="Optional run_val summary.json or results dir for application outputs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--hard-case-count", type=int, default=4)
    parser.add_argument("--operator", default="auto",
                        choices=["auto", "stage4.1.add"])
    parser.add_argument("--material-mean-abs-diff", type=float, default=0.01)
    parser.add_argument("--warn-x-abs-diff", type=float, default=DEFAULT_THRESHOLDS["x_abs_diff"])
    parser.add_argument("--warn-size-abs-diff", type=float, default=DEFAULT_THRESHOLDS["size_abs_diff"])
    parser.add_argument("--warn-vis-conf-abs-diff", type=float, default=DEFAULT_THRESHOLDS["vis_conf_abs_diff"])
    parser.add_argument("--score-weight-x", type=float, default=1.0)
    parser.add_argument("--score-weight-size", type=float, default=1.0)
    parser.add_argument("--score-weight-vis", type=float, default=1.0)
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


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._")
    return cleaned or "item"


def discover_images(images_dir: Path, limit: Optional[int]) -> list[Path]:
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    images = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts)
    if limit is not None:
        return images[:limit]
    return images


def module_precision_bits(module) -> int:
    precision = getattr(module, "precision", None)
    if precision is not None and hasattr(precision, "get_bits"):
        return int(precision.get_bits())
    return 8


def activation_policy_reports(
    tensors: list[np.ndarray],
    module,
) -> dict[str, dict[str, Any]]:
    bits = module_precision_bits(module)
    baseline_alpha = max(float(scalar_from_value(getattr(module, "alpha", None)) or 0.0), 1e-12)
    flattened = [np.asarray(tensor, dtype=np.float64).reshape(-1) for tensor in tensors if np.asarray(tensor).size]
    values = np.concatenate(flattened, axis=0) if flattened else np.asarray([], dtype=np.float64)
    baseline_report = simulate_activation_quantization(values, baseline_alpha, bits, symmetric=False)
    reports: dict[str, dict[str, Any]] = {
        "baseline": {
            "policy_name": "baseline",
            "alpha": float(baseline_alpha),
            "symmetric": False,
            "quantization_report": {
                "mse": float(baseline_report["mse"]),
                "mean_abs_error": float(baseline_report["mean_abs_error"]),
                "max_abs_error": float(baseline_report["max_abs_error"]),
                "clip_fraction": float(baseline_report["clip_fraction"]),
                "eps_out": baseline_report["eps_out"],
            },
        }
    }

    mse_pick = select_activation_alpha_by_mse(tensors, bits, symmetric=False)
    reports["mse"] = {
        "policy_name": "mse",
        "alpha": float(mse_pick["alpha"]),
        "symmetric": False,
        "quantization_report": mse_pick["best_report"],
        "search": mse_pick["search"],
    }
    for percentile in (99.9, 99.5, 99.0):
        key = f"percentile_{str(percentile).replace('.', '_')}"
        alpha = select_activation_alpha_by_percentile(tensors, percentile, symmetric=False)
        report = simulate_activation_quantization(values, alpha, bits, symmetric=False)
        reports[key] = {
            "policy_name": key,
            "alpha": float(alpha),
            "symmetric": False,
            "percentile": float(percentile),
            "quantization_report": {
                "mse": float(report["mse"]),
                "mean_abs_error": float(report["mean_abs_error"]),
                "max_abs_error": float(report["max_abs_error"]),
                "clip_fraction": float(report["clip_fraction"]),
                "eps_out": report["eps_out"],
            },
        }

    if values.size and float(np.min(values)) < -1e-9:
        sym_pick = select_activation_alpha_by_mse(tensors, bits, symmetric=True)
        reports["symmetric_mse"] = {
            "policy_name": "symmetric_mse",
            "alpha": float(sym_pick["alpha"]),
            "symmetric": True,
            "quantization_report": sym_pick["best_report"],
            "search": sym_pick["search"],
        }

    return reports


def build_activation_policy_specs(
    region_key: str,
    region_spec: dict[str, Any],
    policy_reports: dict[str, dict[str, Any]],
) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    activation_module = str(region_spec["activation_module"])
    for policy_key, report in policy_reports.items():
        if policy_key == "baseline":
            continue
        alpha = float(report["alpha"])
        policy_name = str(report["policy_name"])
        specs.append(
            PolicySpec(
                name=f"{region_key}_{policy_name}",
                description=(
                    f"{region_spec['label']} with {policy_name} alpha={alpha:.8f}"
                ),
                operator_name=str(region_spec["operator_name"]),
                family="activation",
                group_name=str(region_spec.get("group_name") or f"{region_key}_activation"),
                target_label=str(region_spec["target_label"]),
                patched_regions=[str(region_spec.get("patched_region") or region_key)],
                activation_overrides={
                    activation_module: {
                        "alpha": alpha,
                        "policy_name": policy_name,
                        "symmetric": bool(report.get("symmetric", False)),
                        "quantization_report": deepcopy(report.get("quantization_report")),
                    }
                },
                search_context=deepcopy(report),
            )
        )
    return specs


def simulate_integer_add_semantic(
    branch_samples: dict[str, list[np.ndarray]],
    eps_in_list: list[float],
    eps_out: float,
    *,
    requantization_factor: int = 32,
) -> dict[str, Any]:
    if len(eps_in_list) < 2:
        raise ValueError("stage4.1.add expects two branch eps values.")
    if eps_out <= 0.0:
        raise ValueError(f"eps_out must be positive, got {eps_out}")

    min_eps = min(float(value) for value in eps_in_list if value is not None)
    exponent = int(np.ceil(np.log2(float(requantization_factor) * float(eps_out) / max(min_eps, 1e-12))))
    divisor = 2 ** max(exponent, 0)
    mul = [int(round(float(divisor) * float(eps) / float(eps_out))) for eps in eps_in_list[:2]]

    predicted = []
    for main_tensor, skip_tensor in zip(branch_samples.get("main", []), branch_samples.get("skip", [])):
        main_arr = np.asarray(main_tensor, dtype=np.float64)
        skip_arr = np.asarray(skip_tensor, dtype=np.float64)
        main_raw = np.rint(main_arr / float(eps_in_list[0])).astype(np.int64)
        skip_raw = np.rint(skip_arr / float(eps_in_list[1])).astype(np.int64)
        pre_raw = (main_raw * np.int64(mul[0])) + (skip_raw * np.int64(mul[1]))
        post_raw = np.floor(pre_raw.astype(np.float64) / float(divisor)).astype(np.int64)
        predicted.append(post_raw.astype(np.float64) * float(eps_out))

    target_flattened = [
        np.asarray(tensor, dtype=np.float64).reshape(-1)
        for tensor in branch_samples.get("output", [])
        if np.asarray(tensor).size
    ]
    target = np.concatenate(target_flattened, axis=0) if target_flattened else np.asarray([], dtype=np.float64)
    predicted_flattened = [
        np.asarray(tensor, dtype=np.float64).reshape(-1)
        for tensor in predicted
        if np.asarray(tensor).size
    ]
    predicted_flat = (
        np.concatenate(predicted_flattened, axis=0)
        if predicted_flattened
        else np.asarray([], dtype=np.float64)
    )
    diff = compare_arrays_rich(target, predicted_flat) if target.size and predicted_flat.size else None
    return {
        "eps_out": float(eps_out),
        "D": int(divisor),
        "mul": mul,
        "semantic_compare": diff,
    }


def build_add_scale_policy_reports(
    branch_samples: dict[str, list[np.ndarray]],
    eps_in_list: list[float],
) -> dict[str, dict[str, Any]]:
    if len(eps_in_list) < 2:
        raise ValueError(f"Expected two eps_in values for stage4.1.add, got {eps_in_list}")
    max_eps = float(max(eps_in_list))
    min_eps = float(min(eps_in_list))
    joint_balanced_eps = float(np.sqrt(max_eps * min_eps))
    reports: dict[str, dict[str, Any]] = {
        "baseline": {
            "policy_name": "baseline",
            "override": None,
            "scale_report": simulate_integer_add_semantic(branch_samples, eps_in_list, max_eps),
        },
        "max_branch": {
            "policy_name": "max_branch",
            "override": "max_branch",
            "scale_report": simulate_integer_add_semantic(branch_samples, eps_in_list, max_eps),
        },
        "joint_balanced": {
            "policy_name": "joint_balanced",
            "override": "joint_balanced",
            "scale_report": simulate_integer_add_semantic(branch_samples, eps_in_list, joint_balanced_eps),
        },
    }

    search_candidates = np.unique(
        np.concatenate(
            [
                np.geomspace(max(min_eps * 0.5, 1e-12), max_eps * 2.0, 80),
                np.asarray([min_eps, joint_balanced_eps, max_eps], dtype=np.float64),
            ]
        )
    )
    search_rows = []
    best_report = None
    best_key = None
    for candidate in search_candidates:
        scale_report = simulate_integer_add_semantic(branch_samples, eps_in_list, float(candidate))
        drift = scale_report.get("semantic_compare") or {}
        row = {
            "eps_out": float(candidate),
            "D": scale_report["D"],
            "mul": scale_report["mul"],
            "mean_abs_diff": float(drift.get("mean_abs_diff") or 0.0),
            "max_abs_diff": float(drift.get("max_abs_diff") or 0.0),
            "cosine_similarity": drift.get("cosine_similarity"),
            "abs_mean_ratio": float(drift.get("abs_mean_ratio") or 0.0),
        }
        search_rows.append(row)
        key = (
            float(row["mean_abs_diff"]),
            float(row["max_abs_diff"]),
            -float(candidate),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_report = scale_report

    if best_report is None:
        best_report = simulate_integer_add_semantic(branch_samples, eps_in_list, max_eps)

    reports["mse_selected_joint"] = {
        "policy_name": "mse_selected_joint",
        "override": {
            "eps_out": float(best_report["eps_out"]),
            "policy_name": "mse_selected_joint",
            "metadata": {
                "search_rows": search_rows,
                "semantic_compare": deepcopy(best_report.get("semantic_compare")),
            },
        },
        "scale_report": best_report,
        "search": search_rows,
    }
    return reports


def build_add_scale_policy_specs(add_reports: dict[str, dict[str, Any]]) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for policy_key, report in add_reports.items():
        if policy_key == "baseline":
            continue
        override = deepcopy(report.get("override"))
        specs.append(
            PolicySpec(
                name=f"add_scale_{policy_key}",
                description=f"stage4.1.add residual balancing with {policy_key}",
                operator_name="stage4.1.add",
                family="integer_add",
                group_name="add_scale",
                target_label="stage4.1.add post-requant",
                patched_regions=["add"],
                integer_add_operator_overrides=(
                    {"stage4.1.add": override}
                    if override is not None
                    else {}
                ),
                search_context=deepcopy(report),
            )
        )
    return specs


def build_export_args(args: argparse.Namespace, ckpt_path: Path, calib_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="hybrid_follow",
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        height=args.height,
        width=args.width,
        input_channels=args.input_channels,
        ckpt=str(ckpt_path),
        out=str(DEFAULT_OUTPUT_DIR / "unused.onnx"),
        opset_version=13,
        bits=args.bits,
        eps_in=args.eps_in,
        stage="id",
        strict_stage=True,
        stage_report=None,
        force_cpu=True,
        calib_dir=str(calib_dir),
        calib_tensor=None,
        calib_batches=args.calib_batches,
        calib_seed=args.calib_seed,
        mean=None,
        std=None,
        disable_conv_bn_fusion=False,
        disable_hybrid_follow_head_collapse=False,
        debug_quant_drift_dir=None,
        clamp_dory_weights=False,
        round_export_params=False,
    )


def load_hybrid_follow_sample(path: Path, hw: tuple[int, int], device: torch.device) -> dict[str, Any]:
    x_float = hybrid_follow_image_to_tensor(path=path, hw=hw, device=device)
    x_staged = torch.round(torch.clamp(x_float, 0.0, 1.0) * 255.0).to(dtype=torch.float32)
    return {
        "image_name": path.name,
        "image_path": str(path),
        "float": x_float,
        "staged": x_staged,
    }


def output_payload(output: Any, stage: str) -> dict[str, Any]:
    raw = np.asarray(output, dtype=np.float64).reshape(-1)
    decoded = hybrid_follow_output_to_decoded(raw, stage)
    return {
        "stage": stage,
        "raw_native": make_json_ready(raw.tolist()),
        "raw_semantic": make_json_ready(semantic_output(raw, stage).tolist()),
        "decoded": make_json_ready(decoded),
    }


def final_output_score(drift: dict[str, Any] | None, weights: dict[str, float]) -> float | None:
    if drift is None:
        return None
    return (
        weights["x"] * float(drift["x_abs_diff"])
        + weights["size"] * float(drift["size_abs_diff"])
        + weights["vis"] * float(drift["vis_conf_abs_diff"])
    )


def x_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def ordering_agreement(fp_values: np.ndarray, quant_values: np.ndarray) -> float | None:
    if fp_values.size < 2 or quant_values.size != fp_values.size:
        return None
    agree = 0
    total = 0
    for left_idx in range(fp_values.size):
        for right_idx in range(left_idx + 1, fp_values.size):
            fp_delta = float(fp_values[left_idx] - fp_values[right_idx])
            if abs(fp_delta) < 1e-9:
                continue
            quant_delta = float(quant_values[left_idx] - quant_values[right_idx])
            total += 1
            if x_sign(fp_delta) == x_sign(quant_delta):
                agree += 1
    if total == 0:
        return None
    return float(agree) / float(total)


def summarize_anti_collapse(fp_values: list[float], exported_values: list[float]) -> dict[str, Any]:
    if not fp_values or not exported_values or len(fp_values) != len(exported_values):
        return {
            "count": 0,
            "sign_flip_rate": None,
            "correlation": None,
            "slope": None,
            "collapsed_fraction": None,
            "left_right_ordering_agreement": None,
        }

    fp = np.asarray(fp_values, dtype=np.float64)
    exported = np.asarray(exported_values, dtype=np.float64)
    sign_flip_rate = float(
        np.mean([x_sign(float(a)) != x_sign(float(b)) for a, b in zip(fp, exported)])
    )

    correlation = None
    if fp.size >= 2 and float(np.std(fp)) > 0.0 and float(np.std(exported)) > 0.0:
        correlation = float(np.corrcoef(fp, exported)[0, 1])

    slope = None
    denom = float(np.dot(fp, fp))
    if denom > 0.0:
        slope = float(np.dot(fp, exported) / denom)

    collapse_mask = np.abs(fp) > 0.5
    collapsed_fraction = None
    if np.any(collapse_mask):
        collapsed_fraction = float(np.mean(np.abs(exported[collapse_mask]) < 0.25))

    return {
        "count": int(fp.size),
        "sign_flip_rate": sign_flip_rate,
        "correlation": correlation,
        "slope": slope,
        "collapsed_fraction": collapsed_fraction,
        "left_right_ordering_agreement": ordering_agreement(fp, exported),
    }


def anti_collapse_sort_key(metrics: dict[str, Any] | None) -> tuple[float, float, float, float, float]:
    metrics = metrics or {}
    sign_flip_rate = float(metrics.get("sign_flip_rate") or 1.0)
    collapsed_fraction = float(metrics.get("collapsed_fraction") or 1.0)
    correlation = float(metrics.get("correlation") or -1.0)
    slope = float(metrics.get("slope") or 0.0)
    ordering = float(metrics.get("left_right_ordering_agreement") or 0.0)
    return (
        sign_flip_rate,
        collapsed_fraction,
        -correlation,
        abs(slope - 1.0),
        -ordering,
    )


def compare_decoded_payloads(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    return {
        "x_abs_diff": abs(float(left["x_offset"]) - float(right["x_offset"])),
        "size_abs_diff": abs(float(left["size_proxy"]) - float(right["size_proxy"])),
        "vis_logit_abs_diff": abs(float(left["visibility_logit"]) - float(right["visibility_logit"])),
        "vis_conf_abs_diff": abs(
            float(left["visibility_confidence"]) - float(right["visibility_confidence"])
        ),
        "left": make_json_ready(left),
        "right": make_json_ready(right),
    }


def transition_status(report: dict[str, Any] | None, thresholds: dict[str, float]) -> str:
    if not isinstance(report, dict):
        return "skipped"
    if (
        float(report.get("x_abs_diff", 0.0)) > thresholds["x_abs_diff"]
        or float(report.get("size_abs_diff", 0.0)) > thresholds["size_abs_diff"]
        or float(report.get("vis_conf_abs_diff", 0.0)) > thresholds["vis_conf_abs_diff"]
    ):
        return "warn"
    return "ok"


def first_material_transition(pairwise: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    ordered = [
        ("fp_to_fq", "FP -> FQ"),
        ("fq_to_id", "FQ -> ID"),
        ("id_to_onnx", "ID -> ONNX"),
        ("onnx_to_application", "ONNX -> application"),
    ]
    for key, label in ordered:
        report = pairwise.get(key)
        if transition_status(report, thresholds) == "warn":
            return {
                "transition_key": key,
                "transition_label": label,
                "status": "warn",
            }
    return {
        "transition_key": None,
        "transition_label": None,
        "status": "ok",
    }


def decoded_triplet(decoded: dict[str, Any] | None) -> str:
    if not isinstance(decoded, dict):
        return "n/a"
    return "{:+.3f} / {:+.3f} / {:.3f}".format(
        float(decoded["x_offset"]),
        float(decoded["size_proxy"]),
        float(decoded["visibility_confidence"]),
    )


def load_application_outputs(run_val_summary: Path | None) -> dict[str, dict[str, Any]]:
    if run_val_summary is None:
        return {}

    summary_path = run_val_summary
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run_val summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        image_name = row.get("image_name")
        if not image_name:
            continue
        gvsoc = (row.get("stage_outputs") or {}).get("gvsoc")
        if not isinstance(gvsoc, dict):
            continue
        mapping[image_name] = {
            "stage": gvsoc.get("stage_tag", "id"),
            "raw_native": make_json_ready(gvsoc.get("raw_native")),
            "raw_semantic": make_json_ready(gvsoc.get("raw_semantic")),
            "decoded": {
                "x_offset": float(gvsoc["x_offset_raw"]),
                "size_proxy": float(gvsoc["size_proxy"]),
                "visibility_logit": float(gvsoc["visibility_logit"]),
                "visibility_confidence": float(gvsoc["visibility_confidence"]),
            },
        }
    return mapping


def build_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def run_onnx_output(session: ort.InferenceSession, input_tensor: torch.Tensor) -> np.ndarray:
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    output = session.run([output_name], {input_name: input_tensor.detach().cpu().numpy()})[0]
    return np.asarray(output)


def export_integer_model_onnx(
    model_id,
    onnx_path: Path,
    *,
    input_channels: int,
    height: int,
    width: int,
) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    nemo.utils.export_onnx(
        str(onnx_path),
        model_id,
        model_id,
        (input_channels, height, width),
        round_params=False,
        batch_size=1,
    )
    return onnx_path


def shift_from_divisor(divisor: Any) -> int | None:
    if divisor in (None, 0):
        return None
    rounded = int(round(float(divisor)))
    if rounded <= 0 or (rounded & (rounded - 1)) != 0:
        return None
    return int(np.log2(rounded))


def channel_outlier_report(value: Any) -> dict[str, Any] | None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim < 2:
        return None
    reduce_axes = tuple(idx for idx in range(arr.ndim) if idx != 1)
    channel_abs_mean = np.mean(np.abs(arr), axis=reduce_axes)
    if channel_abs_mean.size == 0:
        return None
    sorted_indices = np.argsort(channel_abs_mean)[::-1]
    median_value = float(np.median(channel_abs_mean))
    return {
        "channel_abs_mean_stats": tensor_stats(channel_abs_mean),
        "top_channels": [
            {"channel": int(idx), "abs_mean": float(channel_abs_mean[idx])}
            for idx in sorted_indices[:5]
        ],
        "max_to_median_abs_mean_ratio": float(channel_abs_mean[sorted_indices[0]]) / max(median_value, 1e-12),
    }


def _conv_output_channel_abs_max(module) -> np.ndarray:
    weight = np.asarray(module.weight.detach().cpu().numpy(), dtype=np.float64)
    channel_abs_max = np.max(np.abs(weight), axis=(1, 2, 3))
    if getattr(module, "bias", None) is not None:
        bias = np.asarray(module.bias.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        channel_abs_max = np.maximum(channel_abs_max, np.abs(bias))
    return channel_abs_max


def _conv_input_channel_abs_max(module) -> np.ndarray:
    weight = np.asarray(module.weight.detach().cpu().numpy(), dtype=np.float64)
    return np.max(np.abs(weight), axis=(0, 2, 3))


def _channel_max_ratio(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    safe_left = np.maximum(np.asarray(left, dtype=np.float64), 1e-12)
    safe_right = np.maximum(np.asarray(right, dtype=np.float64), 1e-12)
    ratio = np.maximum(safe_left / safe_right, safe_right / safe_left)
    return {
        "stats": tensor_stats(ratio),
        "max": float(np.max(ratio)),
        "median": float(np.median(ratio)),
        "mean": float(np.mean(ratio)),
    }


def stage4_1_add_branch_alignment_report(
    model,
    calib_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    branch_samples = collect_integer_add_branch_samples(model, calib_samples, "stage4.1.add")
    main_tensor = np.concatenate(branch_samples["main"], axis=0) if branch_samples.get("main") else np.asarray([], dtype=np.float64)
    skip_tensor = np.concatenate(branch_samples["skip"], axis=0) if branch_samples.get("skip") else np.asarray([], dtype=np.float64)
    output_tensor = (
        np.concatenate(branch_samples["output"], axis=0)
        if branch_samples.get("output")
        else np.asarray([], dtype=np.float64)
    )
    main_abs_mean = float(np.mean(np.abs(main_tensor))) if main_tensor.size else None
    skip_abs_mean = float(np.mean(np.abs(skip_tensor))) if skip_tensor.size else None
    main_abs_max = float(np.max(np.abs(main_tensor))) if main_tensor.size else None
    skip_abs_max = float(np.max(np.abs(skip_tensor))) if skip_tensor.size else None
    return {
        "main_stats": tensor_stats(main_tensor),
        "skip_stats": tensor_stats(skip_tensor),
        "output_stats": tensor_stats(output_tensor),
        "main_to_skip_abs_mean_ratio": (
            None
            if main_abs_mean is None or skip_abs_mean is None
            else float(main_abs_mean) / max(float(skip_abs_mean), 1e-12)
        ),
        "main_to_skip_abs_max_ratio": (
            None
            if main_abs_max is None or skip_abs_max is None
            else float(main_abs_max) / max(float(skip_abs_max), 1e-12)
        ),
        "main_outlier_channels": channel_outlier_report(main_tensor),
        "skip_outlier_channels": channel_outlier_report(skip_tensor),
    }


def stage4_1_conv_pair_equalization_report(model) -> dict[str, Any]:
    conv1 = resolve_dotted_module(model, "stage4.1.conv1")
    conv2 = resolve_dotted_module(model, "stage4.1.conv2")
    if not isinstance(conv1, torch.nn.Conv2d) or not isinstance(conv2, torch.nn.Conv2d):
        raise TypeError("stage4.1 conv pair equalization expects Conv2d modules for conv1 and conv2.")
    conv1_out = _conv_output_channel_abs_max(conv1)
    conv2_in = _conv_input_channel_abs_max(conv2)
    scales = np.sqrt(np.maximum(conv2_in, 1e-12) / np.maximum(conv1_out, 1e-12))
    return {
        "pair": ["stage4.1.conv1", "stage4.1.conv2"],
        "conv1_output_channel_abs_max": tensor_stats(conv1_out),
        "conv2_input_channel_abs_max": tensor_stats(conv2_in),
        "channel_max_ratio_before": _channel_max_ratio(conv1_out, conv2_in),
        "suggested_scale_stats": tensor_stats(scales),
        "top_channels_by_mismatch": [
            {
                "channel": int(idx),
                "conv1_out_abs_max": float(conv1_out[idx]),
                "conv2_in_abs_max": float(conv2_in[idx]),
                "scale": float(scales[idx]),
                "ratio": float(
                    max(
                        float(conv1_out[idx]) / max(float(conv2_in[idx]), 1e-12),
                        float(conv2_in[idx]) / max(float(conv1_out[idx]), 1e-12),
                    )
                ),
            }
            for idx in np.argsort(
                np.maximum(
                    np.maximum(conv1_out, 1e-12) / np.maximum(conv2_in, 1e-12),
                    np.maximum(conv2_in, 1e-12) / np.maximum(conv1_out, 1e-12),
                )
            )[::-1][:5]
        ],
    }


def apply_stage4_1_conv_pair_cross_layer_equalization(
    model,
    *,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> dict[str, Any]:
    conv1 = resolve_dotted_module(model, "stage4.1.conv1")
    conv2 = resolve_dotted_module(model, "stage4.1.conv2")
    if not isinstance(conv1, torch.nn.Conv2d) or not isinstance(conv2, torch.nn.Conv2d):
        raise TypeError("stage4.1 conv pair equalization expects Conv2d modules for conv1 and conv2.")

    conv1_out_before = _conv_output_channel_abs_max(conv1)
    conv2_in_before = _conv_input_channel_abs_max(conv2)
    raw_scales = np.sqrt(np.maximum(conv2_in_before, 1e-12) / np.maximum(conv1_out_before, 1e-12))
    clipped_scales = np.clip(raw_scales, float(scale_min), float(scale_max))

    scale_tensor = torch.as_tensor(
        clipped_scales,
        dtype=conv1.weight.dtype,
        device=conv1.weight.device,
    )
    with torch.no_grad():
        conv1.weight.mul_(scale_tensor.view(-1, 1, 1, 1))
        if conv1.bias is not None:
            conv1.bias.mul_(scale_tensor)
        conv2.weight.div_(scale_tensor.view(1, -1, 1, 1))

    conv1_out_after = _conv_output_channel_abs_max(conv1)
    conv2_in_after = _conv_input_channel_abs_max(conv2)
    return {
        "patch_name": "stage4_1_conv_pair_cross_layer_equalization",
        "pair": ["stage4.1.conv1", "stage4.1.conv2"],
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "raw_scale_stats": tensor_stats(raw_scales),
        "applied_scale_stats": tensor_stats(clipped_scales),
        "channel_max_ratio_before": _channel_max_ratio(conv1_out_before, conv2_in_before),
        "channel_max_ratio_after": _channel_max_ratio(conv1_out_after, conv2_in_after),
        "top_channels_by_applied_scale": [
            {
                "channel": int(idx),
                "scale": float(clipped_scales[idx]),
                "raw_scale": float(raw_scales[idx]),
                "conv1_out_abs_max_before": float(conv1_out_before[idx]),
                "conv2_in_abs_max_before": float(conv2_in_before[idx]),
                "conv1_out_abs_max_after": float(conv1_out_after[idx]),
                "conv2_in_abs_max_after": float(conv2_in_after[idx]),
            }
            for idx in np.argsort(np.abs(np.log2(np.maximum(clipped_scales, 1e-12))))[::-1][:5]
        ],
    }


def apply_model_patch_overrides(
    model,
    policy_overrides: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    overrides = deepcopy(policy_overrides or {})
    if not overrides:
        return []

    reports: list[dict[str, Any]] = []
    if "stage4_1_conv_pair_cross_layer_equalization" in overrides:
        config = deepcopy(overrides.pop("stage4_1_conv_pair_cross_layer_equalization") or {})
        reports.append(
            apply_stage4_1_conv_pair_cross_layer_equalization(
                model,
                scale_min=float(config.get("scale_min", 0.25)),
                scale_max=float(config.get("scale_max", 4.0)),
            )
        )

    if overrides:
        raise ValueError(f"Unsupported model patch overrides: {sorted(overrides.keys())}")
    return reports


def tap_context_from_model(model_id, head_eps_in: float | None) -> dict[str, dict[str, Any]]:
    context = stage4_1_path_quant_context(model_id)
    global_pool = resolve_dotted_module(model_id, "global_pool")
    head_name = "head" if hasattr(model_id, "head") else "head_x"
    head_module = resolve_dotted_module(model_id, head_name)
    global_pool_eps_out = scalar_from_value(getattr(global_pool, "eps_out", None))
    context["global_pool"] = {
        "module_class": global_pool.__class__.__name__,
        "eps_in": scalar_from_value(getattr(global_pool, "eps_in", None)),
        "eps_out": global_pool_eps_out if global_pool_eps_out is not None else head_eps_in,
        "D": scalar_from_value(getattr(global_pool, "D", None)),
        "shift": shift_from_divisor(scalar_from_value(getattr(global_pool, "D", None))),
        "mul": make_json_ready(getattr(global_pool, "mul", None)),
        "zero_point": None,
    }
    context["head"] = {
        "module_class": head_module.__class__.__name__,
        "eps_in": head_eps_in,
        "eps_out": 1.0 / FINAL_OUTPUT_SCALE,
        "D": scalar_from_value(getattr(head_module, "D", None)),
        "shift": shift_from_divisor(scalar_from_value(getattr(head_module, "D", None))),
        "mul": make_json_ready(getattr(head_module, "mul", None)),
        "zero_point": None,
    }
    return context


def build_local_tap_records(
    fq_probe: dict[str, Any],
    id_probe: dict[str, Any],
    id_add_audit: dict[str, Any],
    context_map: dict[str, dict[str, Any]],
    *,
    head_eps_in: float | None,
) -> list[dict[str, Any]]:
    stage4_add_tensors = id_add_audit["tensors"].get("stage4.1.add", {})
    add_scale = (id_add_audit["reports"].get("stage4.1.add") or {}).get("scale_selection") or {}

    fq_add = np.asarray(fq_probe["tensors"]["stage4_1_add"], dtype=np.float64)
    fq_next_input = np.asarray(fq_probe["tensors"]["stage4_1_out_relu"], dtype=np.float64)
    fq_global = np.asarray(fq_probe["tensors"]["global_pool_post_requant"], dtype=np.float64)
    fq_head = np.asarray(fq_probe["tensors"]["head_input"], dtype=np.float64)

    add_ctx = context_map.get("stage4.1.add") or {}
    next_ctx = context_map.get("stage4.1.out_relu") or {}
    global_ctx = context_map.get("global_pool") or {}
    head_ctx = context_map.get("head") or {}

    global_eps = global_ctx.get("eps_out") if global_ctx.get("eps_out") not in (None, 0.0) else head_eps_in

    tap_rows = [
        {
            "label": "stage4.1.add pre-requant",
            "operator_name": "stage4.1.add",
            "fq_tensor": fq_add,
            "id_tensor": np.asarray(stage4_add_tensors["pre_requant_semantic"], dtype=np.float64),
            "operator_context": add_ctx or add_scale,
            "clip_bounds": make_json_ready(add_scale.get("clip_bounds_semantic_equivalent")),
            "factor_audit": {
                "branch_eps_ratio": add_scale.get("branch_eps_ratio"),
                "output_lsb_per_input_lsb": add_scale.get("output_lsb_per_input_lsb"),
                "input_lsb_per_output_lsb": add_scale.get("input_lsb_per_output_lsb"),
                "branch_inputs": make_json_ready((id_add_audit["reports"].get("stage4.1.add") or {}).get("branch_inputs")),
            },
        },
        {
            "label": "stage4.1.add post-requant",
            "operator_name": "stage4.1.add",
            "fq_tensor": fq_add,
            "id_tensor": np.asarray(stage4_add_tensors["post_requant_semantic"], dtype=np.float64),
            "operator_context": add_scale,
            "clip_bounds": make_json_ready(add_scale.get("clip_bounds_semantic_equivalent")),
            "factor_audit": {
                "branch_eps_ratio": add_scale.get("branch_eps_ratio"),
                "output_lsb_per_input_lsb": add_scale.get("output_lsb_per_input_lsb"),
                "input_lsb_per_output_lsb": add_scale.get("input_lsb_per_output_lsb"),
                "requant_rounding_loss": compare_arrays_rich(
                    np.asarray(stage4_add_tensors["pre_requant_semantic"], dtype=np.float64),
                    np.asarray(stage4_add_tensors["post_requant_semantic"], dtype=np.float64),
                ),
                "branch_inputs": make_json_ready((id_add_audit["reports"].get("stage4.1.add") or {}).get("branch_inputs")),
            },
        },
        {
            "label": "stage4.1.out_relu input",
            "operator_name": "stage4.1.out_relu",
            "fq_tensor": fq_next_input,
            "id_tensor": np.asarray(id_probe["tensors"]["stage4_1_out_relu"], dtype=np.float64) * float(next_ctx.get("eps_out") or 1.0),
            "operator_context": next_ctx,
            "clip_bounds": None,
            "factor_audit": {},
        },
        {
            "label": "global pool",
            "operator_name": "global_pool",
            "fq_tensor": fq_global,
            "id_tensor": np.asarray(id_probe["tensors"]["global_pool_post_requant"], dtype=np.float64) * float(global_eps or 1.0),
            "operator_context": global_ctx,
            "clip_bounds": None,
            "factor_audit": {
                "outlier_channels": channel_outlier_report(fq_global),
            },
        },
        {
            "label": "head input",
            "operator_name": "head",
            "fq_tensor": fq_head,
            "id_tensor": np.asarray(id_probe["tensors"]["head_input"], dtype=np.float64) * float(head_eps_in or 1.0),
            "operator_context": head_ctx,
            "clip_bounds": None,
            "factor_audit": {
                "outlier_channels": channel_outlier_report(fq_head),
            },
        },
    ]

    records = []
    for row in tap_rows:
        clip_bounds = row["clip_bounds"] or {}
        op_ctx = row["operator_context"] or {}
        drift = compare_arrays_rich(row["fq_tensor"], row["id_tensor"])
        records.append(
            {
                "label": row["label"],
                "operator_name": row["operator_name"],
                "eps_in": op_ctx.get("eps_in") if "eps_in" in op_ctx else op_ctx.get("eps_in_list"),
                "eps_out": op_ctx.get("eps_out"),
                "zero_point": op_ctx.get("zero_point"),
                "mul": make_json_ready(op_ctx.get("mul")),
                "shift": op_ctx.get("shift"),
                "D": op_ctx.get("D"),
                "fq_stats": tensor_stats(row["fq_tensor"]),
                "id_stats": tensor_stats(row["id_tensor"]),
                "fq_to_id": drift,
                "saturation": saturation_stats(
                    row["id_tensor"],
                    min_value=clip_bounds.get("min"),
                    max_value=clip_bounds.get("max"),
                    quantum=op_ctx.get("eps_out"),
                ),
                "factor_audit": make_json_ready(row["factor_audit"]),
            }
        )
    return records


def first_bad_tap(tap_records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    for tap in tap_records:
        if float((tap.get("fq_to_id") or {}).get("mean_abs_diff") or 0.0) >= threshold:
            return tap
    return max(
        tap_records,
        key=lambda tap: float((tap.get("fq_to_id") or {}).get("mean_abs_diff") or -1.0),
    )


def policy_specs_for_operator(operator_name: str) -> list[PolicySpec]:
    baseline = PolicySpec(
        name="current",
        description="Current exporter defaults",
        operator_name=operator_name,
        family="baseline",
    )
    if operator_name in {"stage4.1.conv1", "stage4.1.conv2"}:
        return [
            baseline,
            PolicySpec(
                name="eps_static_nearest_even",
                description="Use eps_static for deploy conv-bias integerization",
                operator_name=operator_name,
                family="conv_bias",
                conv_bias_overrides={
                    operator_name: {
                        "scale_source": "eps_static",
                        "rounding_mode": "nearest_even",
                    }
                },
            ),
            PolicySpec(
                name="eps_out_static_half_away_from_zero",
                description="Keep eps_out_static but switch bias rounding mode",
                operator_name=operator_name,
                family="conv_bias",
                conv_bias_overrides={
                    operator_name: {
                        "scale_source": "eps_out_static",
                        "rounding_mode": "half_away_from_zero",
                    }
                },
            ),
            PolicySpec(
                name="eps_static_half_away_from_zero",
                description="Use eps_static with half-away-from-zero bias rounding",
                operator_name=operator_name,
                family="conv_bias",
                conv_bias_overrides={
                    operator_name: {
                        "scale_source": "eps_static",
                        "rounding_mode": "half_away_from_zero",
                    }
                },
            ),
        ]

    add_candidates = [
        policy
        for policy in HYBRID_FOLLOW_INTEGER_ADD_POLICY_CANDIDATES
        if policy != HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY
    ]
    specs = [baseline]
    for policy in add_candidates[:3]:
        specs.append(
            PolicySpec(
                name=policy,
                description=f"Use {policy} only for stage4.1.add output eps selection",
                operator_name=operator_name,
                family="integer_add",
                integer_add_operator_overrides={"stage4.1.add": policy},
            )
        )
    return specs


def build_models_for_policy(
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    spec: PolicySpec,
):
    with integer_add_scale_selection_scope(
        HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
        spec.integer_add_operator_overrides,
    ):
        model_fp = prepare_model_fp(export_args, device)
        model_patch_report = apply_model_patch_overrides(
            model_fp,
            spec.model_patch_overrides,
        )
        dummy_input = torch.randn(
            1,
            export_args.input_channels,
            export_args.height,
            export_args.width,
            device=device,
        )

        def build_quant_base():
            model_quant = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
            model_quant.to(device).eval()
            repair_hybrid_follow_fused_quant_graph(model_quant)
            model_quant.change_precision(bits=export_args.bits, scale_weights=True, scale_activations=True)
            run_activation_calibration(model_quant, calib_samples)
            activation_override_report = apply_activation_alpha_overrides(
                model_quant,
                spec.activation_overrides,
            )
            model_quant._sweep_activation_override_report = activation_override_report
            return model_quant

        model_fq = build_quant_base()

        model_id = build_quant_base()
        try:
            model_id.reset_alpha_weights()
        except Exception:
            pass
        repair_hybrid_follow_fused_quant_graph(model_id)
        model_id.qd_stage(eps_in=export_args.eps_in)
        repair_hybrid_follow_fused_quant_graph(model_id)
        model_id.id_stage()
        normalize_integer_requant_tensors(model_id)
        conv_bias_report = integerize_deploy_conv_biases(
            model_id,
            default_scale_source=HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE,
            default_rounding_mode=HYBRID_FOLLOW_CONV_BIAS_ROUNDING,
            policy_overrides=spec.conv_bias_overrides,
            collect_reports=True,
        )
        model_id.eval()

        head_eps_in = resolve_hybrid_follow_head_input_eps(model_id)
        context_map = tap_context_from_model(model_id, head_eps_in)

        return {
            "model_fp": model_fp,
            "model_fq": model_fq,
            "model_id": model_id,
            "head_eps_in": head_eps_in,
            "context_map": context_map,
            "conv_bias_report": conv_bias_report,
            "model_patch_report": make_json_ready(model_patch_report),
            "activation_override_report": make_json_ready(
                getattr(model_id, "_sweep_activation_override_report", [])
            ),
        }


def evaluate_policy_trial(
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    spec: PolicySpec,
    *,
    known_bad_sample: dict[str, Any],
    batch_samples: list[dict[str, Any]],
    application_map: dict[str, dict[str, Any]],
    output_dir: Path,
    thresholds: dict[str, float],
    weights: dict[str, float],
) -> dict[str, Any]:
    trial_dir = output_dir / "candidates" / sanitize_name(spec.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial: dict[str, Any] = {
        "policy": spec.name,
        "description": spec.description,
        "operator_name": spec.operator_name,
        "family": spec.family,
        "group_name": spec.group_name,
        "target_label": spec.target_label,
        "patched_regions": spec.patched_regions,
        "search_context": make_json_ready(spec.search_context),
    }

    try:
        models = build_models_for_policy(export_args, device, calib_samples, spec)
        model_fp = models["model_fp"]
        model_fq = models["model_fq"]
        model_id = models["model_id"]

        fq_probe = run_hybrid_follow_pytorch_probe(model_fq, known_bad_sample["float"])
        id_probe = run_hybrid_follow_pytorch_probe(model_id, known_bad_sample["staged"])
        id_add_audit = run_hybrid_follow_integer_add_audit(model_id, known_bad_sample["staged"])

        local_taps = build_local_tap_records(
            fq_probe,
            id_probe,
            id_add_audit,
            models["context_map"],
            head_eps_in=models["head_eps_in"],
        )
        known_drift = compare_decoded_hybrid_follow_outputs(
            fq_probe["tensors"]["model_output"],
            "fq",
            id_probe["tensors"]["model_output"],
            "id",
        )
        known_score = final_output_score(known_drift, weights)
        first_bad = first_bad_tap(local_taps, threshold=thresholds["material_mean_abs_diff"])

        onnx_path = export_integer_model_onnx(
            model_id,
            trial_dir / "model_id.onnx",
            input_channels=export_args.input_channels,
            height=export_args.height,
            width=export_args.width,
        )
        onnx_session = build_onnx_session(onnx_path)

        batch_rows = []
        deploy_vs_onnx_warn_count = 0
        with torch.no_grad():
            for sample in batch_samples:
                fp_output = model_fp(sample["float"]).detach().cpu().numpy()
                fq_output = model_fq(sample["float"]).detach().cpu().numpy()
                id_output = model_id(sample["staged"]).detach().cpu().numpy()
                onnx_output = run_onnx_output(onnx_session, sample["staged"])

                fp_payload = output_payload(fp_output, "fp")
                fq_payload = output_payload(fq_output, "fq")
                id_payload = output_payload(id_output, "id")
                onnx_payload = output_payload(onnx_output, "id")
                app_payload = application_map.get(sample["image_name"])

                pairwise = {
                    "fp_to_fq": compare_decoded_hybrid_follow_outputs(fp_output, "fp", fq_output, "fq"),
                    "fq_to_id": compare_decoded_hybrid_follow_outputs(fq_output, "fq", id_output, "id"),
                    "id_to_onnx": compare_decoded_hybrid_follow_outputs(id_output, "id", onnx_output, "id"),
                    "onnx_to_application": compare_decoded_payloads(
                        onnx_payload["decoded"],
                        None if app_payload is None else app_payload.get("decoded"),
                    ),
                }
                first_transition = first_material_transition(pairwise, thresholds)
                final_drift = pairwise["fq_to_id"]
                score = final_output_score(final_drift, weights)
                x_sign = None
                if final_drift is not None:
                    left = final_drift["left"]
                    right = final_drift["right"]
                    x_sign = int(np.sign(float(right["x_offset"]) - float(left["x_offset"])))

                raw_match = compare_arrays(
                    np.rint(np.asarray(id_output).reshape(-1)).astype(np.int64),
                    np.rint(np.asarray(onnx_output).reshape(-1)).astype(np.int64),
                )
                id_to_onnx_semantic_ok = transition_status(pairwise["id_to_onnx"], DEPLOY_ONNX_THRESHOLDS) == "ok"
                if not id_to_onnx_semantic_ok:
                    deploy_vs_onnx_warn_count += 1

                batch_rows.append(
                    {
                        "image_name": sample["image_name"],
                        "image_path": sample["image_path"],
                        "score_final_output": score,
                        "x_sign_error": x_sign,
                        "final_output_drift": make_json_ready(final_drift),
                        "per_head_abs_error": (
                            {
                                "x_abs_diff": float(final_drift["x_abs_diff"]),
                                "size_abs_diff": float(final_drift["size_abs_diff"]),
                                "vis_conf_abs_diff": float(final_drift["vis_conf_abs_diff"]),
                            }
                            if final_drift is not None
                            else None
                        ),
                        "stage_outputs": {
                            "fp": fp_payload,
                            "fq": fq_payload,
                            "id": id_payload,
                            "onnx": onnx_payload,
                            "application": app_payload,
                        },
                        "pairwise": make_json_ready(pairwise),
                        "first_material_transition": first_transition,
                        "id_to_onnx_raw_compare": raw_match,
                        "id_to_onnx_semantic_ok": id_to_onnx_semantic_ok,
                    }
                )

        score_values = [float(row["score_final_output"]) for row in batch_rows if row["score_final_output"] is not None]
        x_errors = [float(row["per_head_abs_error"]["x_abs_diff"]) for row in batch_rows if row["per_head_abs_error"]]
        size_errors = [float(row["per_head_abs_error"]["size_abs_diff"]) for row in batch_rows if row["per_head_abs_error"]]
        vis_errors = [float(row["per_head_abs_error"]["vis_conf_abs_diff"]) for row in batch_rows if row["per_head_abs_error"]]
        fp_x_values = [
            float(((row.get("stage_outputs") or {}).get("fp") or {}).get("decoded", {}).get("x_offset"))
            for row in batch_rows
            if ((row.get("stage_outputs") or {}).get("fp") or {}).get("decoded") is not None
        ]
        onnx_x_values = [
            float(((row.get("stage_outputs") or {}).get("onnx") or {}).get("decoded", {}).get("x_offset"))
            for row in batch_rows
            if ((row.get("stage_outputs") or {}).get("onnx") or {}).get("decoded") is not None
        ]
        application_x_values = [
            float(((row.get("stage_outputs") or {}).get("application") or {}).get("decoded", {}).get("x_offset"))
            for row in batch_rows
            if ((row.get("stage_outputs") or {}).get("application") or {}).get("decoded") is not None
        ]
        anti_collapse = {
            "onnx": summarize_anti_collapse(fp_x_values, onnx_x_values),
            "application": summarize_anti_collapse(fp_x_values, application_x_values),
        }

        trial.update(
            {
                "local_report": {
                    "known_bad_image": known_bad_sample["image_name"],
                    "first_bad_tap": make_json_ready(first_bad),
                    "tap_records": make_json_ready(local_taps),
                    "score_final_output": known_score,
                    "final_output_drift": make_json_ready(known_drift),
                    "activation_overrides": make_json_ready(models.get("activation_override_report")),
                    "conv_bias_integerization": make_json_ready(models["conv_bias_report"]),
                    "integer_add_scale_selection": make_json_ready(
                        (id_add_audit["reports"].get("stage4.1.add") or {}).get("scale_selection")
                    ),
                },
                "batch_report": {
                    "aggregate": {
                        "count": len(batch_rows),
                        "score_mean": float(np.mean(score_values)) if score_values else None,
                        "score_median": float(median(score_values)) if score_values else None,
                        "score_max": float(np.max(score_values)) if score_values else None,
                        "x_abs_diff_mean": float(np.mean(x_errors)) if x_errors else None,
                        "size_abs_diff_mean": float(np.mean(size_errors)) if size_errors else None,
                        "vis_conf_abs_diff_mean": float(np.mean(vis_errors)) if vis_errors else None,
                        "deploy_vs_onnx_semantic_ok_count": len(batch_rows) - deploy_vs_onnx_warn_count,
                        "deploy_vs_onnx_warn_count": deploy_vs_onnx_warn_count,
                    },
                    "anti_collapse": make_json_ready(anti_collapse),
                    "images": make_json_ready(batch_rows),
                    "onnx_path": str(onnx_path),
                },
            }
        )
        return trial
    except Exception as exc:
        trial["error"] = f"{type(exc).__name__}: {exc}"
        return trial


def operator_from_first_bad(first_bad: dict[str, Any], requested_operator: str) -> str:
    if requested_operator != "auto":
        return requested_operator
    return "stage4.1.add"


def add_baseline_comparisons(trials: dict[str, dict[str, Any]], baseline_name: str) -> None:
    baseline_images = {
        row["image_name"]: row
        for row in ((trials.get(baseline_name) or {}).get("batch_report") or {}).get("images", [])
    }
    baseline_scores = {
        image_name: float(row["score_final_output"])
        for image_name, row in baseline_images.items()
        if row.get("score_final_output") is not None
    }
    baseline_signs = {
        image_name: row.get("x_sign_error")
        for image_name, row in baseline_images.items()
    }

    for policy_name, trial in trials.items():
        batch_report = trial.get("batch_report") or {}
        images = batch_report.get("images") or []
        wins = 0
        losses = 0
        ties = 0
        sign_flip_count = 0
        deltas = []
        for row in images:
            image_name = row["image_name"]
            baseline_score = baseline_scores.get(image_name)
            score = row.get("score_final_output")
            if baseline_score is None or score is None:
                continue
            delta = float(score) - float(baseline_score)
            deltas.append(delta)
            if delta < -1e-12:
                wins += 1
            elif delta > 1e-12:
                losses += 1
            else:
                ties += 1
            baseline_sign = baseline_signs.get(image_name)
            if baseline_sign is not None and row.get("x_sign_error") is not None and baseline_sign != row.get("x_sign_error"):
                sign_flip_count += 1

        batch_report["vs_baseline"] = {
            "baseline_policy": baseline_name,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_flip_count_x": sign_flip_count,
            "mean_score_delta": float(np.mean(deltas)) if deltas else None,
            "median_score_delta": float(median(deltas)) if deltas else None,
        }


def rank_candidates(
    trials: dict[str, dict[str, Any]],
    baseline_name: str,
    target_operator_name: str,
) -> list[dict[str, Any]]:
    ranked = []
    for policy_name, trial in trials.items():
        if trial.get("error"):
            continue
        local_report = trial.get("local_report") or {}
        batch_report = trial.get("batch_report") or {}
        first_bad = local_report.get("first_bad_tap") or {}
        tap_records = local_report.get("tap_records") or []
        target_tap = None
        for tap in tap_records:
            if tap.get("operator_name") == target_operator_name:
                target_tap = tap
                break
        if target_tap is None:
            target_tap = first_bad
        aggregate = batch_report.get("aggregate") or {}
        vs_baseline = batch_report.get("vs_baseline") or {}
        ranked.append(
            {
                "policy": policy_name,
                "target_mean_abs_diff": float(((target_tap.get("fq_to_id") or {}).get("mean_abs_diff")) or 0.0),
                "batch_mean_score": float(aggregate.get("score_mean") or 0.0),
                "wins": int(vs_baseline.get("wins") or 0),
                "losses": int(vs_baseline.get("losses") or 0),
                "deploy_vs_onnx_warn_count": int(aggregate.get("deploy_vs_onnx_warn_count") or 0),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["target_mean_abs_diff"],
            item["batch_mean_score"],
            -item["wins"],
            item["deploy_vs_onnx_warn_count"],
        )
    )
    return ranked


def make_recommendation(
    trials: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    *,
    baseline_name: str,
    target_operator_name: str,
) -> dict[str, Any]:
    baseline = trials.get(baseline_name) or {}
    baseline_local = (baseline.get("local_report") or {}).get("first_bad_tap") or {}
    baseline_mean = float((((baseline.get("batch_report") or {}).get("aggregate") or {}).get("score_mean")) or 0.0)
    baseline_target = float(((baseline_local.get("fq_to_id") or {}).get("mean_abs_diff")) or 0.0)

    if not ranking:
        return {
            "action": "reject all tested policies",
            "reason": "No candidate trials completed successfully.",
        }

    top = ranking[0]
    if top["policy"] == baseline_name:
        return {
            "action": "keep baseline",
            "reason": "The current policy remains best after ranking local drift first and batch score second.",
        }

    selected = trials[top["policy"]]
    selected_batch = (selected.get("batch_report") or {}).get("aggregate") or {}
    selected_vs_baseline = (selected.get("batch_report") or {}).get("vs_baseline") or {}

    improved_local = top["target_mean_abs_diff"] < baseline_target - 1e-12
    improved_batch = float(selected_batch.get("score_mean") or 0.0) < baseline_mean - 1e-12
    non_losing_batch = int(selected_vs_baseline.get("wins") or 0) >= int(selected_vs_baseline.get("losses") or 0)
    deploy_matches_onnx = int(selected_batch.get("deploy_vs_onnx_warn_count") or 0) == 0

    if improved_local and improved_batch and non_losing_batch and deploy_matches_onnx:
        return {
            "action": "patch this single operator with this single policy",
            "operator_name": target_operator_name,
            "policy": top["policy"],
            "reason": (
                f"{top['policy']} lowered the earliest local drift at {target_operator_name}, "
                "improved the representative-batch mean final-output score, and preserved deploy==ONNX."
            ),
        }

    if improved_local and not improved_batch:
        return {
            "action": "reject all tested policies",
            "reason": (
                f"{top['policy']} helped the known sample at {target_operator_name} "
                "but lost on the representative batch."
            ),
        }

    return {
        "action": "keep baseline",
        "reason": "No tested policy beat the baseline on both local drift and batch score.",
    }


def local_operator_sweep_markdown(
    trials: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    baseline_name: str,
    target_operator_name: str,
    known_bad_image: str,
) -> str:
    lines = [
        "# Local Operator Sweep",
        "",
        f"- Known bad image: `{known_bad_image}`",
        f"- Operator under test: `{target_operator_name}`",
        f"- Baseline policy: `{baseline_name}`",
        "",
        "## Ranking",
        "",
        "| Policy | Target fq->id mean abs diff | Known-sample score | First bad tap |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in ranking:
        trial = trials[item["policy"]]
        local = trial.get("local_report") or {}
        first_bad = local.get("first_bad_tap") or {}
        lines.append(
            "| `{}` | `{:.6f}` | `{:.6f}` | `{}` |".format(
                item["policy"],
                float(item["target_mean_abs_diff"]),
                float(local.get("score_final_output") or 0.0),
                first_bad.get("label"),
            )
        )

    for policy_name, trial in trials.items():
        lines.extend(["", f"## {policy_name}", ""])
        if trial.get("error"):
            lines.append(f"- error: `{trial['error']}`")
            continue
        local = trial.get("local_report") or {}
        final_drift = local.get("final_output_drift") or {}
        first_bad = local.get("first_bad_tap") or {}
        lines.extend(
            [
                f"- first bad tap: `{first_bad.get('label')}`",
                "- known-sample score: `{:.6f}`".format(float(local.get("score_final_output") or 0.0)),
                "- final drift: x=`{:.6f}` size=`{:.6f}` vis_conf=`{:.6f}`".format(
                    float(final_drift.get("x_abs_diff") or 0.0),
                    float(final_drift.get("size_abs_diff") or 0.0),
                    float(final_drift.get("vis_conf_abs_diff") or 0.0),
                ),
                "",
                "| Tap | Mean abs diff | Max abs diff | Cosine | Abs-mean ratio | eps_in | eps_out | D | shift | mul |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for tap in local.get("tap_records") or []:
            drift = tap.get("fq_to_id") or {}
            lines.append(
                "| `{}` | `{:.6f}` | `{:.6f}` | `{}` | `{:.6f}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                    tap.get("label"),
                    float(drift.get("mean_abs_diff") or 0.0),
                    float(drift.get("max_abs_diff") or 0.0),
                    "n/a" if drift.get("cosine_similarity") is None else f"{float(drift['cosine_similarity']):.6f}",
                    float(drift.get("abs_mean_ratio") or 0.0),
                    tap.get("eps_in"),
                    tap.get("eps_out"),
                    tap.get("D"),
                    tap.get("shift"),
                    tap.get("mul"),
                )
            )
    return "\n".join(lines)


def batch_score_compare_markdown(
    trials: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    baseline_name: str,
    recommendation: dict[str, Any],
) -> str:
    lines = [
        "# Batch Score Compare",
        "",
        f"- Baseline policy: `{baseline_name}`",
        f"- Recommendation: `{recommendation.get('action')}`",
        "",
        "## Aggregate",
        "",
        "| Policy | Mean score | Median score | Wins | Losses | Ties | Sign flips (x) | deploy->onnx warn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in ranking:
        trial = trials[item["policy"]]
        batch = trial.get("batch_report") or {}
        agg = batch.get("aggregate") or {}
        vs_baseline = batch.get("vs_baseline") or {}
        lines.append(
            "| `{}` | `{:.6f}` | `{:.6f}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                item["policy"],
                float(agg.get("score_mean") or 0.0),
                float(agg.get("score_median") or 0.0),
                int(vs_baseline.get("wins") or 0),
                int(vs_baseline.get("losses") or 0),
                int(vs_baseline.get("ties") or 0),
                int(vs_baseline.get("sign_flip_count_x") or 0),
                int(agg.get("deploy_vs_onnx_warn_count") or 0),
            )
        )

    selected_name = recommendation.get("policy") or ranking[0]["policy"]
    selected_trial = trials.get(selected_name) or {}
    lines.extend(
        [
            "",
            f"## Stage Table ({selected_name})",
            "",
            "| Image | FP | FQ | ID | ONNX | Application | First materially wrong |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ((selected_trial.get("batch_report") or {}).get("images") or []):
        stages = row.get("stage_outputs") or {}
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["image_name"],
                decoded_triplet((stages.get("fp") or {}).get("decoded")),
                decoded_triplet((stages.get("fq") or {}).get("decoded")),
                decoded_triplet((stages.get("id") or {}).get("decoded")),
                decoded_triplet((stages.get("onnx") or {}).get("decoded")),
                decoded_triplet((stages.get("application") or {}).get("decoded")),
                (row.get("first_material_transition") or {}).get("transition_label") or "none",
            )
        )
    return "\n".join(lines)


def summary_markdown(summary: dict[str, Any]) -> str:
    recommendation = summary["recommendation"]
    lines = [
        "# Hybrid Follow Quant Drift Sweep",
        "",
        f"- Known bad image: `{summary['known_bad_image']}`",
        f"- Eval image count: `{summary['eval_count']}`",
        f"- Operator under test: `{summary['operator_under_test']}`",
        f"- Baseline policy: `{summary['baseline_policy']}`",
        f"- Recommendation: `{recommendation['action']}`",
    ]
    if recommendation.get("policy"):
        lines.append(f"- Recommended policy: `{recommendation['policy']}`")
    lines.extend(
        [
            f"- Reason: `{recommendation['reason']}`",
            "",
            "## Outputs",
            "",
            f"- Local sweep: `{summary['artifacts']['local_operator_sweep_json']}`",
            f"- Batch compare: `{summary['artifacts']['batch_score_compare_json']}`",
        ]
    )
    return "\n".join(lines)


def target_tap_for_trial(
    trial: dict[str, Any],
    target_operator_name: str,
    *,
    target_label: str | None = None,
) -> dict[str, Any]:
    local_report = trial.get("local_report") or {}
    tap_records = local_report.get("tap_records") or []
    if target_label:
        for tap in tap_records:
            if tap.get("label") == target_label:
                return tap
    for tap in tap_records:
        if tap.get("operator_name") == target_operator_name:
            return tap
    return local_report.get("first_bad_tap") or {}


def select_hard_case_names(batch_rows: list[dict[str, Any]], count: int) -> list[str]:
    scored = [row for row in batch_rows if row.get("score_final_output") is not None]
    scored.sort(key=lambda row: float(row.get("score_final_output") or 0.0), reverse=True)
    return [row["image_name"] for row in scored[: max(int(count), 0)]]


def build_batch_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_values = [float(row["score_final_output"]) for row in rows if row.get("score_final_output") is not None]
    x_errors = [float(row["per_head_abs_error"]["x_abs_diff"]) for row in rows if row.get("per_head_abs_error")]
    size_errors = [float(row["per_head_abs_error"]["size_abs_diff"]) for row in rows if row.get("per_head_abs_error")]
    vis_errors = [float(row["per_head_abs_error"]["vis_conf_abs_diff"]) for row in rows if row.get("per_head_abs_error")]
    warn_count = sum(1 for row in rows if not bool(row.get("id_to_onnx_semantic_ok", False)))
    return {
        "count": len(rows),
        "score_mean": float(np.mean(score_values)) if score_values else None,
        "score_median": float(median(score_values)) if score_values else None,
        "score_max": float(np.max(score_values)) if score_values else None,
        "x_abs_diff_mean": float(np.mean(x_errors)) if x_errors else None,
        "size_abs_diff_mean": float(np.mean(size_errors)) if size_errors else None,
        "vis_conf_abs_diff_mean": float(np.mean(vis_errors)) if vis_errors else None,
        "deploy_vs_onnx_semantic_ok_count": len(rows) - warn_count,
        "deploy_vs_onnx_warn_count": warn_count,
    }


def attach_dataset_views_to_trial(trial: dict[str, Any], hard_case_names: list[str]) -> None:
    batch_report = trial.get("batch_report") or {}
    images = list(batch_report.get("images") or [])
    hard_case_set = set(hard_case_names)
    datasets = {
        "rep16": {
            "aggregate": build_batch_aggregate(images),
            "images": images,
        }
    }
    if hard_case_set:
        hard_rows = [row for row in images if row["image_name"] in hard_case_set]
        datasets["hard_case"] = {
            "aggregate": build_batch_aggregate(hard_rows),
            "images": hard_rows,
        }
    batch_report["datasets"] = make_json_ready(datasets)
    batch_report["aggregate"] = make_json_ready(datasets["rep16"]["aggregate"])
    batch_report["images"] = make_json_ready(images)
    batch_report["hard_case_names"] = list(hard_case_names)


def add_baseline_comparisons(trials: dict[str, dict[str, Any]], baseline_name: str) -> None:
    baseline_trial = trials.get(baseline_name) or {}
    baseline_datasets = ((baseline_trial.get("batch_report") or {}).get("datasets") or {})
    for trial in trials.values():
        batch_report = trial.get("batch_report") or {}
        datasets = batch_report.get("datasets") or {}
        for dataset_name, dataset in datasets.items():
            baseline_images = {
                row["image_name"]: row
                for row in (baseline_datasets.get(dataset_name) or {}).get("images", [])
            }
            baseline_scores = {
                image_name: float(row["score_final_output"])
                for image_name, row in baseline_images.items()
                if row.get("score_final_output") is not None
            }
            baseline_signs = {
                image_name: row.get("x_sign_error")
                for image_name, row in baseline_images.items()
            }
            wins = 0
            losses = 0
            ties = 0
            sign_flip_count = 0
            deltas = []
            for row in dataset.get("images") or []:
                image_name = row["image_name"]
                baseline_score = baseline_scores.get(image_name)
                score = row.get("score_final_output")
                if baseline_score is None or score is None:
                    continue
                delta = float(score) - float(baseline_score)
                deltas.append(delta)
                if delta < -1e-12:
                    wins += 1
                elif delta > 1e-12:
                    losses += 1
                else:
                    ties += 1
                baseline_sign = baseline_signs.get(image_name)
                if baseline_sign is not None and row.get("x_sign_error") is not None and baseline_sign != row.get("x_sign_error"):
                    sign_flip_count += 1
            dataset["vs_baseline"] = {
                "baseline_policy": baseline_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_flip_count_x": sign_flip_count,
                "mean_score_delta": float(np.mean(deltas)) if deltas else None,
                "median_score_delta": float(median(deltas)) if deltas else None,
            }
        if "rep16" in datasets:
            batch_report["vs_baseline"] = make_json_ready(datasets["rep16"].get("vs_baseline"))


def rank_candidates(
    trials: dict[str, dict[str, Any]],
    baseline_name: str,
    target_operator_name: str,
    *,
    target_label: str | None = None,
    dataset_key: str = "rep16",
    policy_names: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    ranked = []
    for policy_name, trial in trials.items():
        if policy_names is not None and policy_name not in policy_names:
            continue
        if trial.get("error"):
            continue
        target_tap = target_tap_for_trial(
            trial,
            target_operator_name,
            target_label=target_label,
        )
        batch_report = trial.get("batch_report") or {}
        dataset = (batch_report.get("datasets") or {}).get(dataset_key) or {}
        hard_case_dataset = (batch_report.get("datasets") or {}).get("hard_case") or {}
        aggregate = dataset.get("aggregate") or {}
        vs_baseline = dataset.get("vs_baseline") or {}
        hard_agg = hard_case_dataset.get("aggregate") or {}
        hard_vs_baseline = hard_case_dataset.get("vs_baseline") or {}
        anti_collapse = batch_report.get("anti_collapse") or {}
        onnx_anti_collapse = anti_collapse.get("onnx") or {}
        ranked.append(
            {
                "policy": policy_name,
                "group_name": trial.get("group_name"),
                "target_mean_abs_diff": float(((target_tap.get("fq_to_id") or {}).get("mean_abs_diff")) or 0.0),
                "batch_mean_score": float(aggregate.get("score_mean") or 0.0),
                "wins": int(vs_baseline.get("wins") or 0),
                "losses": int(vs_baseline.get("losses") or 0),
                "deploy_vs_onnx_warn_count": int(aggregate.get("deploy_vs_onnx_warn_count") or 0),
                "hard_case_mean_score": hard_agg.get("score_mean"),
                "hard_case_wins": int(hard_vs_baseline.get("wins") or 0),
                "hard_case_losses": int(hard_vs_baseline.get("losses") or 0),
                "onnx_anti_collapse": make_json_ready(onnx_anti_collapse),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["target_mean_abs_diff"],
            anti_collapse_sort_key(item.get("onnx_anti_collapse")),
            item["batch_mean_score"],
            -item["wins"],
            item["deploy_vs_onnx_warn_count"],
        )
    )
    return ranked


def build_microblock_specs(
    selected_specs: dict[str, Optional[PolicySpec]],
    *,
    target_operator_name: str,
    target_label: str | None,
) -> list[PolicySpec]:
    region_members = {
        "conv1": [selected_specs.get("conv1_activation")],
        "conv2": [selected_specs.get("conv2_activation")],
        "add": [selected_specs.get("add_activation"), selected_specs.get("add_scale")],
    }
    specs: list[PolicySpec] = []
    for combo_name, region_keys in MICROBLOCK_SWEEP_LAYOUT:
        members = []
        for region_key in region_keys:
            members.extend([spec for spec in region_members.get(region_key, []) if spec is not None])
        activation_overrides: dict[str, dict[str, Any]] = {}
        integer_add_overrides: dict[str, Any] = {}
        conv_bias_overrides: dict[str, dict[str, Any]] = {}
        descriptions = []
        search_context = {}
        patched_regions = []
        for member in members:
            activation_overrides.update(deepcopy(member.activation_overrides))
            integer_add_overrides.update(deepcopy(member.integer_add_operator_overrides))
            conv_bias_overrides.update(deepcopy(member.conv_bias_overrides))
            descriptions.append(member.name)
            search_context[member.group_name] = member.name
            patched_regions.extend(member.patched_regions)
        specs.append(
            PolicySpec(
                name=f"microblock_{combo_name}",
                description=(
                    f"{combo_name.replace('_', '+')} using "
                    + (", ".join(descriptions) if descriptions else "baseline/no-op winners")
                ),
                operator_name=target_operator_name,
                family="microblock",
                group_name="microblock",
                target_label=target_label,
                patched_regions=sorted(set(patched_regions or region_keys)),
                activation_overrides=activation_overrides,
                integer_add_operator_overrides=integer_add_overrides,
                conv_bias_overrides=conv_bias_overrides,
                search_context=search_context,
            )
        )
    return specs


def make_recommendation(
    trials: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    *,
    baseline_name: str,
    target_operator_name: str,
    target_label: str | None = None,
) -> dict[str, Any]:
    baseline = trials.get(baseline_name) or {}
    baseline_target = target_tap_for_trial(baseline, target_operator_name, target_label=target_label)
    baseline_rep16 = (((baseline.get("batch_report") or {}).get("datasets") or {}).get("rep16") or {})
    baseline_hard = (((baseline.get("batch_report") or {}).get("datasets") or {}).get("hard_case") or {})
    baseline_mean = float(((baseline_rep16.get("aggregate") or {}).get("score_mean")) or 0.0)
    baseline_target_mean = float(((baseline_target.get("fq_to_id") or {}).get("mean_abs_diff")) or 0.0)
    baseline_hard_mean = (baseline_hard.get("aggregate") or {}).get("score_mean")

    if not ranking:
        return {
            "action": "reject all tested policies",
            "reason": "No candidate trials completed successfully.",
        }

    top = ranking[0]
    if top["policy"] == baseline_name:
        return {
            "action": "keep baseline",
            "reason": "The current exporter remains best after ranking earliest local drift first, rep16 mean score second, and wins vs baseline third.",
        }

    selected = trials[top["policy"]]
    selected_rep16 = (((selected.get("batch_report") or {}).get("datasets") or {}).get("rep16") or {})
    selected_hard = (((selected.get("batch_report") or {}).get("datasets") or {}).get("hard_case") or {})
    rep16_agg = selected_rep16.get("aggregate") or {}
    rep16_vs_baseline = selected_rep16.get("vs_baseline") or {}
    hard_agg = selected_hard.get("aggregate") or {}
    hard_vs_baseline = selected_hard.get("vs_baseline") or {}
    baseline_anti = (((baseline.get("batch_report") or {}).get("anti_collapse") or {}).get("onnx") or {})
    selected_anti = (((selected.get("batch_report") or {}).get("anti_collapse") or {}).get("onnx") or {})

    improved_local = top["target_mean_abs_diff"] < baseline_target_mean - 1e-12
    improved_rep16 = float(rep16_agg.get("score_mean") or 0.0) < baseline_mean - 1e-12
    non_losing_rep16 = int(rep16_vs_baseline.get("wins") or 0) >= int(rep16_vs_baseline.get("losses") or 0)
    deploy_matches_onnx = int(rep16_agg.get("deploy_vs_onnx_warn_count") or 0) == 0
    improved_anti_collapse = anti_collapse_sort_key(selected_anti) <= anti_collapse_sort_key(baseline_anti)

    improved_hard_case = True
    non_losing_hard_case = True
    if baseline_hard_mean is not None and hard_agg.get("score_mean") is not None:
        improved_hard_case = float(hard_agg.get("score_mean") or 0.0) <= float(baseline_hard_mean) + 1e-12
        non_losing_hard_case = int(hard_vs_baseline.get("wins") or 0) >= int(hard_vs_baseline.get("losses") or 0)

    if improved_local and improved_rep16 and non_losing_rep16 and improved_hard_case and non_losing_hard_case and deploy_matches_onnx and improved_anti_collapse:
        return {
            "action": "patch this single operator with this single policy",
            "operator_name": target_operator_name,
            "policy": top["policy"],
            "reason": (
                f"{top['policy']} lowered the earliest local drift around {target_label or target_operator_name}, "
                "improved rep16 mean final-output score, improved or held x anti-collapse metrics, "
                "held the hard-case subset, and preserved deploy==ONNX."
            ),
        }

    downstream_shift_patch = (
        not improved_local
        and top["target_mean_abs_diff"] <= baseline_target_mean + 1e-12
        and improved_rep16
        and non_losing_rep16
        and improved_hard_case
        and non_losing_hard_case
        and deploy_matches_onnx
        and improved_anti_collapse
        and int(rep16_vs_baseline.get("wins") or 0) > int(rep16_vs_baseline.get("losses") or 0)
        and float(rep16_agg.get("score_mean") or 0.0) <= baseline_mean * 0.95
    )
    if downstream_shift_patch:
        return {
            "action": "patch this single operator with this single policy",
            "operator_name": target_operator_name,
            "policy": top["policy"],
            "reason": (
                f"{top['policy']} did not improve the earliest local drift at {target_label or target_operator_name}, "
                "but it clearly shifted the win downstream: rep16 and the hard-case subset both improved without deploy==ONNX regressions."
            ),
        }

    if improved_local and not improved_rep16:
        return {
            "action": "reject all tested policies",
            "reason": (
                f"{top['policy']} improved the known bad sample near {target_label or target_operator_name} "
                "but regressed on rep16."
            ),
        }

    if improved_rep16 and not improved_hard_case:
        return {
            "action": "keep baseline",
            "reason": (
                f"{top['policy']} improved rep16 mean score but lost on the hard-case subset, "
                "so it is not stable enough to replace baseline."
            ),
        }

    return {
        "action": "keep baseline",
        "reason": "No tested exporter-side patch beat baseline on both local drift and batch stability.",
    }


def local_operator_sweep_markdown(
    trials: dict[str, dict[str, Any]],
    group_rankings: dict[str, list[dict[str, Any]]],
    baseline_name: str,
    known_bad_image: str,
) -> str:
    lines = [
        "# Local Operator Sweep",
        "",
        f"- Known bad image: `{known_bad_image}`",
        f"- Baseline policy: `{baseline_name}`",
    ]
    for group_name, ranking in group_rankings.items():
        lines.extend(
            [
                "",
                f"## {group_name}",
                "",
                "| Policy | Target fq->id mean abs diff | Known-sample score | First bad tap |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for item in ranking:
            trial = trials[item["policy"]]
            local = trial.get("local_report") or {}
            first_bad = local.get("first_bad_tap") or {}
            lines.append(
                "| `{}` | `{:.6f}` | `{:.6f}` | `{}` |".format(
                    item["policy"],
                    float(item["target_mean_abs_diff"]),
                    float(local.get("score_final_output") or 0.0),
                    first_bad.get("label"),
                )
            )
        for item in ranking:
            trial = trials[item["policy"]]
            lines.extend(["", f"### {item['policy']}", ""])
            if trial.get("error"):
                lines.append(f"- error: `{trial['error']}`")
                continue
            local = trial.get("local_report") or {}
            final_drift = local.get("final_output_drift") or {}
            first_bad = local.get("first_bad_tap") or {}
            lines.extend(
                [
                    f"- description: `{trial.get('description')}`",
                    f"- patched regions: `{trial.get('patched_regions')}`",
                    f"- first bad tap: `{first_bad.get('label')}`",
                    "- known-sample score: `{:.6f}`".format(float(local.get("score_final_output") or 0.0)),
                    "- final drift: x=`{:.6f}` size=`{:.6f}` vis_conf=`{:.6f}`".format(
                        float(final_drift.get("x_abs_diff") or 0.0),
                        float(final_drift.get("size_abs_diff") or 0.0),
                        float(final_drift.get("vis_conf_abs_diff") or 0.0),
                    ),
                ]
            )
            add_scale = local.get("integer_add_scale_selection") or {}
            if add_scale:
                lines.append(
                    "- stage4.1.add scale: eps_out=`{}` D=`{}` shift=`{}` mul=`{}`".format(
                        add_scale.get("eps_out"),
                        add_scale.get("D"),
                        add_scale.get("shift"),
                        add_scale.get("mul"),
                    )
                )
            if trial.get("search_context"):
                lines.append(f"- search context: `{trial['search_context']}`")
            lines.extend(
                [
                    "",
                    "| Tap | Mean abs diff | Max abs diff | Cosine | Abs-mean ratio | eps_in | eps_out | D | shift | mul |",
                    "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
                ]
            )
            for tap in local.get("tap_records") or []:
                drift = tap.get("fq_to_id") or {}
                lines.append(
                    "| `{}` | `{:.6f}` | `{:.6f}` | `{}` | `{:.6f}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                        tap.get("label"),
                        float(drift.get("mean_abs_diff") or 0.0),
                        float(drift.get("max_abs_diff") or 0.0),
                        "n/a" if drift.get("cosine_similarity") is None else f"{float(drift['cosine_similarity']):.6f}",
                        float(drift.get("abs_mean_ratio") or 0.0),
                        tap.get("eps_in"),
                        tap.get("eps_out"),
                        tap.get("D"),
                        tap.get("shift"),
                        tap.get("mul"),
                    )
                )
    return "\n".join(lines)


def batch_score_compare_markdown(
    trials: dict[str, dict[str, Any]],
    ranking: list[dict[str, Any]],
    baseline_name: str,
    recommendation: dict[str, Any],
) -> str:
    lines = [
        "# Batch Score Compare",
        "",
        f"- Baseline policy: `{baseline_name}`",
        f"- Recommendation: `{recommendation.get('action')}`",
        "",
        "## Aggregate",
        "",
        "| Policy | rep16 mean | rep16 median | rep16 wins | rep16 losses | hard-case mean | hard-case wins | hard-case losses | Sign flips (x) | deploy->onnx warn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in ranking:
        trial = trials[item["policy"]]
        batch = trial.get("batch_report") or {}
        rep16 = (batch.get("datasets") or {}).get("rep16") or {}
        hard_case = (batch.get("datasets") or {}).get("hard_case") or {}
        agg = rep16.get("aggregate") or {}
        vs_baseline = rep16.get("vs_baseline") or {}
        hard_agg = hard_case.get("aggregate") or {}
        hard_vs_baseline = hard_case.get("vs_baseline") or {}
        lines.append(
            "| `{}` | `{:.6f}` | `{:.6f}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                item["policy"],
                float(agg.get("score_mean") or 0.0),
                float(agg.get("score_median") or 0.0),
                int(vs_baseline.get("wins") or 0),
                int(vs_baseline.get("losses") or 0),
                "n/a" if hard_agg.get("score_mean") is None else f"{float(hard_agg.get('score_mean')):.6f}",
                int(hard_vs_baseline.get("wins") or 0),
                int(hard_vs_baseline.get("losses") or 0),
                int(vs_baseline.get("sign_flip_count_x") or 0),
                int(agg.get("deploy_vs_onnx_warn_count") or 0),
            )
        )

    selected_name = recommendation.get("policy") or (ranking[0]["policy"] if ranking else baseline_name)
    selected_trial = trials.get(selected_name) or {}
    anti_collapse = (selected_trial.get("batch_report") or {}).get("anti_collapse") or {}
    onnx_anti_collapse = anti_collapse.get("onnx") or {}
    lines.extend(
        [
            "",
            "## Anti-Collapse (onnx vs fp x)",
            "",
            "- sign_flip_rate=`{}` corr=`{}` slope=`{}` collapsed_fraction=`{}` left_right_ordering_agreement=`{}`".format(
                onnx_anti_collapse.get("sign_flip_rate"),
                onnx_anti_collapse.get("correlation"),
                onnx_anti_collapse.get("slope"),
                onnx_anti_collapse.get("collapsed_fraction"),
                onnx_anti_collapse.get("left_right_ordering_agreement"),
            ),
            "",
            f"## Stage Table ({selected_name})",
            "",
            "| Image | FP | FQ | ID | ONNX | Application | First materially wrong |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in (((selected_trial.get("batch_report") or {}).get("datasets") or {}).get("rep16") or {}).get("images", []):
        stages = row.get("stage_outputs") or {}
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["image_name"],
                decoded_triplet((stages.get("fp") or {}).get("decoded")),
                decoded_triplet((stages.get("fq") or {}).get("decoded")),
                decoded_triplet((stages.get("id") or {}).get("decoded")),
                decoded_triplet((stages.get("onnx") or {}).get("decoded")),
                decoded_triplet((stages.get("application") or {}).get("decoded")),
                (row.get("first_material_transition") or {}).get("transition_label") or "none",
            )
        )
    return "\n".join(lines)


def summary_markdown(summary: dict[str, Any]) -> str:
    recommendation = summary["recommendation"]
    lines = [
        "# Hybrid Follow Quant Drift Sweep",
        "",
        f"- Known bad image: `{summary['known_bad_image']}`",
        f"- Eval image count: `{summary['eval_count']}`",
        f"- Hard-case subset: `{summary['hard_case_names']}`",
        f"- Earliest operator under test: `{summary['operator_under_test']}`",
        f"- Baseline policy: `{summary['baseline_policy']}`",
        f"- Recommendation: `{recommendation['action']}`",
    ]
    if recommendation.get("policy"):
        lines.append(f"- Recommended policy: `{recommendation['policy']}`")
    lines.append(f"- Reason: `{recommendation['reason']}`")
    training_branch = summary.get("training_branch")
    if training_branch:
        lines.extend(
            [
                "",
                "## Training Fallback",
                "",
                f"- Enabled: `{training_branch.get('enabled')}`",
                f"- Triggered: `{training_branch.get('triggered')}`",
                f"- Command: `{training_branch.get('command')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Local sweep: `{summary['artifacts']['local_operator_sweep_json']}`",
            f"- Batch compare: `{summary['artifacts']['batch_score_compare_json']}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_repo_path(args.ckpt)
    calib_dir = resolve_repo_path(args.calib_dir)
    known_bad_image = resolve_repo_path(args.known_bad_image)
    eval_dir = resolve_repo_path(args.eval_dir)
    run_val_summary = resolve_repo_path(args.run_val_summary) if args.run_val_summary else None
    output_dir = resolve_repo_path(args.output_dir)

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if calib_dir is None or not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {args.calib_dir}")
    if known_bad_image is None or not known_bad_image.is_file():
        raise FileNotFoundError(f"Known bad image not found: {args.known_bad_image}")
    if eval_dir is None or not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: {args.eval_dir}")
    if output_dir is None:
        raise RuntimeError("Could not resolve output directory.")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    patch_model_to_graph_compat()

    device = torch.device("cpu")
    export_args = build_export_args(args, ckpt_path, calib_dir)
    image_size = (args.height, args.width)
    calib_samples = collect_calib_samples(export_args, image_size, device)
    known_bad_sample = load_hybrid_follow_sample(known_bad_image, image_size, device)
    batch_samples = [
        load_hybrid_follow_sample(path, image_size, device)
        for path in discover_images(eval_dir, args.eval_limit)
    ]
    application_map = load_application_outputs(run_val_summary)
    thresholds = {
        "x_abs_diff": args.warn_x_abs_diff,
        "size_abs_diff": args.warn_size_abs_diff,
        "vis_conf_abs_diff": args.warn_vis_conf_abs_diff,
        "material_mean_abs_diff": args.material_mean_abs_diff,
    }
    weights = {
        "x": args.score_weight_x,
        "size": args.score_weight_size,
        "vis": args.score_weight_vis,
    }

    baseline_spec = PolicySpec(
        name="current",
        description="Current exporter defaults",
        operator_name=args.operator if args.operator != "auto" else "auto",
        family="baseline",
    )
    baseline_trial = evaluate_policy_trial(
        export_args,
        device,
        calib_samples,
        baseline_spec,
        known_bad_sample=known_bad_sample,
        batch_samples=batch_samples,
        application_map=application_map,
        output_dir=output_dir,
        thresholds=thresholds,
        weights=weights,
    )
    if baseline_trial.get("error"):
        raise RuntimeError(f"Baseline trial failed: {baseline_trial['error']}")

    baseline_first_bad = (baseline_trial.get("local_report") or {}).get("first_bad_tap") or {}
    operator_under_test = operator_from_first_bad(baseline_first_bad, args.operator)
    target_label = "stage4.1.add post-requant"
    hard_case_names = select_hard_case_names(
        (baseline_trial.get("batch_report") or {}).get("images") or [],
        args.hard_case_count,
    )
    attach_dataset_views_to_trial(baseline_trial, hard_case_names)

    reference_models = build_models_for_policy(export_args, device, calib_samples, baseline_spec)
    activation_modules = [ACTIVATION_REGION_SPECS["add_activation"]["activation_module"]]
    activation_sample_map = collect_module_output_samples(
        reference_models["model_fq"],
        calib_samples,
        activation_modules,
        statistics_act=True,
    )
    add_branch_samples = collect_integer_add_branch_samples(
        reference_models["model_fq"],
        calib_samples,
        "stage4.1.add",
    )
    add_eps_in_list = [
        float(value)
        for value in ((reference_models["context_map"].get("stage4.1.add") or {}).get("eps_in_list") or [])
    ]

    group_configs = {
        "add_activation": {
            "operator_name": "stage4.1.add",
            "target_label": "stage4.1.add post-requant",
            "policy_names": {"current"},
        },
        "add_scale": {
            "operator_name": "stage4.1.add",
            "target_label": "stage4.1.add post-requant",
            "policy_names": {"current"},
        },
    }

    specs_by_name: dict[str, PolicySpec] = {baseline_spec.name: baseline_spec}
    individual_specs: list[PolicySpec] = []
    activation_policy_catalog = {}
    region_key = "add_activation"
    region_spec = ACTIVATION_REGION_SPECS[region_key]
    module = resolve_dotted_module(reference_models["model_fq"], region_spec["activation_module"])
    policy_reports = activation_policy_reports(
        activation_sample_map[region_spec["activation_module"]],
        module,
    )
    activation_policy_catalog[region_key] = make_json_ready(policy_reports)
    specs = build_activation_policy_specs(region_key, region_spec, policy_reports)
    individual_specs.extend(specs)
    for spec in specs:
        specs_by_name[spec.name] = spec
        group_configs[spec.group_name]["policy_names"].add(spec.name)

    add_scale_reports = build_add_scale_policy_reports(add_branch_samples, add_eps_in_list)
    add_scale_catalog = make_json_ready(add_scale_reports)
    for spec in build_add_scale_policy_specs(add_scale_reports):
        individual_specs.append(spec)
        specs_by_name[spec.name] = spec
        group_configs[spec.group_name]["policy_names"].add(spec.name)

    local_trials: dict[str, dict[str, Any]] = {"current": baseline_trial}
    for spec in individual_specs:
        local_trials[spec.name] = evaluate_policy_trial(
            export_args,
            device,
            calib_samples,
            spec,
            known_bad_sample=known_bad_sample,
            batch_samples=batch_samples,
            application_map=application_map,
            output_dir=output_dir,
            thresholds=thresholds,
            weights=weights,
        )
        attach_dataset_views_to_trial(local_trials[spec.name], hard_case_names)

    add_baseline_comparisons(local_trials, baseline_name="current")

    group_rankings: dict[str, list[dict[str, Any]]] = {}
    selected_specs: dict[str, Optional[PolicySpec]] = {
        "add_activation": None,
        "add_scale": None,
    }
    for group_name, config in group_configs.items():
        ranking = rank_candidates(
            local_trials,
            baseline_name="current",
            target_operator_name=str(config["operator_name"]),
            target_label=str(config["target_label"]),
            policy_names=set(config["policy_names"]),
        )
        group_rankings[group_name] = ranking
        winner_name = ranking[0]["policy"] if ranking else "current"
        if winner_name != "current":
            selected_specs[group_name] = specs_by_name[winner_name]

    microblock_specs: list[PolicySpec] = []
    add_activation_override = (
        deepcopy(selected_specs["add_activation"].activation_overrides)
        if selected_specs["add_activation"] is not None
        else {}
    )
    add_scale_override = (
        deepcopy(selected_specs["add_scale"].integer_add_operator_overrides)
        if selected_specs["add_scale"] is not None
        else {}
    )
    if add_activation_override or add_scale_override:
        microblock_specs.append(
            PolicySpec(
                name="microblock_add_only",
                description="Patch stage4.1.add only with the current local winners.",
                operator_name=operator_under_test,
                family="microblock",
                group_name="microblock",
                target_label=target_label,
                patched_regions=["add"],
                activation_overrides=add_activation_override,
                integer_add_operator_overrides=add_scale_override,
                search_context={
                    "add_activation_winner": None if selected_specs["add_activation"] is None else selected_specs["add_activation"].name,
                    "add_scale_winner": None if selected_specs["add_scale"] is None else selected_specs["add_scale"].name,
                },
            )
        )
    matched_scale_ablation_spec = PolicySpec(
        name="ablation_matched_scale_residual_add",
        description="Local matched-scale residual add variant around stage4.1.add only.",
        operator_name=operator_under_test,
        family="architecture_ablation",
        group_name="architecture_ablation",
        target_label=target_label,
        patched_regions=["add"],
        activation_overrides=deepcopy(add_activation_override),
        integer_add_operator_overrides={"stage4.1.add": "max_branch"},
        search_context={"variant": "matched_scale_residual_add"},
    )
    architecture_ablation_trials: dict[str, dict[str, Any]] = {"current": baseline_trial}
    architecture_ablation_trials[matched_scale_ablation_spec.name] = evaluate_policy_trial(
        export_args,
        device,
        calib_samples,
        matched_scale_ablation_spec,
        known_bad_sample=known_bad_sample,
        batch_samples=batch_samples,
        application_map=application_map,
        output_dir=output_dir,
        thresholds=thresholds,
        weights=weights,
    )
    attach_dataset_views_to_trial(architecture_ablation_trials[matched_scale_ablation_spec.name], hard_case_names)
    add_baseline_comparisons(architecture_ablation_trials, baseline_name="current")
    microblock_trials: dict[str, dict[str, Any]] = {"current": baseline_trial}
    for spec in microblock_specs:
        microblock_trials[spec.name] = evaluate_policy_trial(
            export_args,
            device,
            calib_samples,
            spec,
            known_bad_sample=known_bad_sample,
            batch_samples=batch_samples,
            application_map=application_map,
            output_dir=output_dir,
            thresholds=thresholds,
            weights=weights,
        )
        attach_dataset_views_to_trial(microblock_trials[spec.name], hard_case_names)

    add_baseline_comparisons(microblock_trials, baseline_name="current")
    microblock_ranking = rank_candidates(
        microblock_trials,
        baseline_name="current",
        target_operator_name=operator_under_test,
        target_label=target_label,
    )
    recommendation = make_recommendation(
        microblock_trials,
        microblock_ranking,
        baseline_name="current",
        target_operator_name=operator_under_test,
        target_label=target_label,
    )

    training_branch = {
        "enabled": False,
        "triggered": False,
        "command": None,
    }

    local_payload = {
        "known_bad_image": str(known_bad_image),
        "operator_under_test": operator_under_test,
        "target_label": target_label,
        "baseline_policy": "current",
        "hard_case_names": hard_case_names,
        "activation_policy_catalog": activation_policy_catalog,
        "add_scale_catalog": add_scale_catalog,
        "group_rankings": group_rankings,
        "architecture_ablation": make_json_ready(architecture_ablation_trials),
        "groups": make_json_ready(
            {
                group_name: {
                    "config": {
                        **config,
                        "policy_names": sorted(config["policy_names"]),
                    },
                    "ranking": group_rankings.get(group_name, []),
                    "policies": {
                        policy_name: local_trials[policy_name]
                        for policy_name in config["policy_names"]
                        if policy_name in local_trials
                    },
                }
                for group_name, config in group_configs.items()
            }
        ),
    }
    batch_payload = {
        "operator_under_test": operator_under_test,
        "target_label": target_label,
        "baseline_policy": "current",
        "hard_case_names": hard_case_names,
        "group_winners": {
            group_name: ("current" if spec is None else spec.name)
            for group_name, spec in selected_specs.items()
        },
        "ranking": microblock_ranking,
        "recommendation": recommendation,
        "weights": weights,
        "thresholds": thresholds,
        "policies": make_json_ready(microblock_trials),
    }
    summary_payload = {
        "known_bad_image": str(known_bad_image),
        "eval_dir": str(eval_dir),
        "eval_count": len(batch_samples),
        "hard_case_names": hard_case_names,
        "operator_under_test": operator_under_test,
        "target_label": target_label,
        "baseline_policy": "current",
        "active_integer_add_policy": HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
        "active_conv_bias_scale_source": HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE,
        "active_conv_bias_rounding": HYBRID_FOLLOW_CONV_BIAS_ROUNDING,
        "group_winners": batch_payload["group_winners"],
        "recommendation": recommendation,
        "ranking": microblock_ranking,
        "training_branch": training_branch,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "local_operator_sweep_json": str(output_dir / "local_operator_sweep.json"),
            "batch_score_compare_json": str(output_dir / "batch_score_compare.json"),
        },
    }

    write_json(output_dir / "local_operator_sweep.json", local_payload)
    write_markdown(
        output_dir / "local_operator_sweep.md",
        local_operator_sweep_markdown(
            local_trials,
            group_rankings,
            baseline_name="current",
            known_bad_image=known_bad_sample["image_name"],
        ),
    )
    write_json(output_dir / "batch_score_compare.json", batch_payload)
    write_markdown(
        output_dir / "batch_score_compare.md",
        batch_score_compare_markdown(
            microblock_trials,
            microblock_ranking,
            baseline_name="current",
            recommendation=recommendation,
        ),
    )
    write_json(output_dir / "summary.json", summary_payload)
    write_markdown(output_dir / "summary.md", summary_markdown(summary_payload))

    print(f"Output dir: {output_dir}")
    print(f"Operator under test: {operator_under_test}")
    print(f"Recommendation: {recommendation['action']}")
    if recommendation.get("policy"):
        print(f"Recommended policy: {recommendation['policy']}")


if __name__ == "__main__":
    main()

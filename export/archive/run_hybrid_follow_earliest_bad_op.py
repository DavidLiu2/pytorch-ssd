#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F

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
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_localize_hybrid_follow_stages import (  # noqa: E402
    TAP_LABELS,
    build_onnx_context,
    build_onnx_probe_context,
    run_onnx_probe,
    semantic_tolerance_for_alias,
    semanticize_focus_taps,
)
from compare_hybrid_follow_stages import (  # noqa: E402
    Thresholds,
    compare_runtime_layers,
    compare_stage_pair,
    parse_numeric_artifact,
    to_stage_result,
)
from export_nemo_quant import (  # noqa: E402
    collect_calib_samples,
    collect_integer_add_branch_samples,
    collect_module_output_samples,
    compare_arrays_rich,
    make_json_ready,
    patch_model_to_graph_compat,
    resolve_dotted_module,
    run_hybrid_follow_integer_add_audit,
    run_hybrid_follow_pytorch_probe,
    saturation_stats,
    tensor_stats,
)
from sweep_hybrid_follow_quant_drift import (  # noqa: E402
    ACTIVATION_REGION_SPECS,
    DEFAULT_CKPT,
    DEFAULT_EVAL_DIR,
    DEFAULT_KNOWN_BAD_IMAGE,
    DEFAULT_THRESHOLDS,
    PolicySpec,
    activation_policy_reports,
    anti_collapse_sort_key,
    build_activation_policy_specs,
    build_add_scale_policy_reports,
    build_add_scale_policy_specs,
    build_export_args,
    build_models_for_policy,
    discover_images,
    export_integer_model_onnx,
    load_hybrid_follow_sample,
    resolve_repo_path,
    run_onnx_output,
    sanitize_name,
    summarize_anti_collapse,
)


DEFAULT_APPLICATION_SUMMARY = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "4_application_vs_checkpoint"
    / "application_vs_checkpoint_20260326_exporter_legacy_default"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "export" / "hybrid_follow" / "earliest_bad_op_loop"
FINAL_OUTPUT_SCALE = 32768.0
LOCAL_MEAN_ABS_DIFF_THRESHOLD = 0.01
LOCAL_TAP_PRIORITY = [
    "stage4_1_conv1",
    "stage4_1_conv2",
    "stage4_1_add_pre_requant",
    "stage4_1_add_post_requant",
    "global_pool_post_requant",
    "head_input",
    "model_output",
]
BOUNDARY_SPECS = [
    ("fp_to_fq", "FP -> FQ", "exporter-side"),
    ("fq_to_id", "FQ -> ID", "exporter-side"),
    ("id_to_onnx", "ID -> ONNX", "exporter-side"),
    ("onnx_to_golden", "ONNX -> golden", "exporter-side"),
    ("golden_to_gvsoc", "golden -> GVSOC", "runtime-side"),
]
EXPORT_BOUNDARY_KEYS = {"fp_to_fq", "fq_to_id", "id_to_onnx"}
FULL_EXPORT_BOUNDARY_KEYS = {"fp_to_fq", "fq_to_id", "id_to_onnx", "onnx_to_golden"}
ACTIONABLE_OPERATOR_BY_ALIAS = {
    "stage4_1_conv1": "stage4.1.conv1",
    "stage4_1_conv2": "stage4.1.conv2",
    "stage4_1_add_pre_requant": "stage4.1.add",
    "stage4_1_add_post_requant": "stage4.1.add",
    "global_pool_post_requant": "stage4.1.add",
    "head_input": "stage4.1.add",
    "model_output": "stage4.1.add",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-iteration earliest-bad-op patch loop for hybrid_follow quant drift. "
            "The loop localizes the first materially bad boundary/operator against the fixed "
            "FP/FQ/ID/ONNX/golden/GVSOC stack, evaluates only local patch candidates for that "
            "operator or immediate micro-region, and emits decision reports."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--calib-dir", default=str(PROJECT_DIR / "data" / "coco" / "images" / "val2017"))
    parser.add_argument("--known-bad-image", default=str(DEFAULT_KNOWN_BAD_IMAGE))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--application-summary", default=str(DEFAULT_APPLICATION_SUMMARY))
    parser.add_argument("--layer-manifest", default=None)
    parser.add_argument("--hard-case-list", default=None, help="Optional text/json file with a fixed hard-case subset.")
    parser.add_argument("--hard-case-count", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
    parser.add_argument("--warn-x-abs-diff", type=float, default=DEFAULT_THRESHOLDS["x_abs_diff"])
    parser.add_argument("--warn-size-abs-diff", type=float, default=DEFAULT_THRESHOLDS["size_abs_diff"])
    parser.add_argument("--warn-vis-conf-abs-diff", type=float, default=DEFAULT_THRESHOLDS["vis_conf_abs_diff"])
    parser.add_argument("--material-local-mean-abs-diff", type=float, default=LOCAL_MEAN_ABS_DIFF_THRESHOLD)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--qat-batch-size", type=int, default=8)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--qat-num-workers", type=int, default=2)
    parser.add_argument("--qat-calib-batches", type=int, default=16)
    parser.add_argument("--qat-activation-range-reg-weight", type=float, default=1e-3)
    parser.add_argument("--qat-max-train-batches", type=int, default=32)
    parser.add_argument("--qat-max-val-batches", type=int, default=16)
    parser.add_argument(
        "--skip-focused-qat",
        action="store_true",
        help="Skip the focused stage4+heads QAT fallback even if exporter-side conv1 tuning is exhausted.",
    )
    parser.add_argument(
        "--allow-missing-runtime-compare",
        action="store_true",
        help="Allow the loop to continue if runtime layer comparison cannot be built.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Run the strict earliest-bad-op localization/report stack only and skip local patch/QAT sweeps.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def nested_tensor_payload(value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "stats": tensor_stats(arr),
        "values": make_json_ready(arr.tolist()),
    }


def tensor_shape_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return list(value.detach().cpu().shape)
    return list(np.asarray(value).shape)


def module_bits(module, attr_name: str) -> int | None:
    value = getattr(module, attr_name, None)
    if value is not None and hasattr(value, "get_bits"):
        return int(value.get_bits())
    return None


def conv_like_module(
    module,
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor | None,
) -> torch.Tensor:
    return F.conv2d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )


def quantized_weight_semantic_from_module(module) -> torch.Tensor:
    import nemo.quant.pact as pact_mod

    weight = module.weight.detach()
    bits = module_bits(module, "W_precision")
    if bits is None or bits <= 0:
        return weight.detach().clone()

    if getattr(module, "quant_asymm", False):
        w_alpha = getattr(module, "W_alpha").detach()
        w_beta = getattr(module, "W_beta").detach()
        eps = (w_beta + w_alpha) / ((2.0 ** bits) - 1.0)
        return pact_mod.pact_quantize_asymm_inference(
            weight,
            eps,
            torch.ceil(w_alpha / eps) * eps,
            torch.floor(w_beta / eps) * eps,
            train_loop=getattr(module, "train_loop", False),
            train_loop_oldprec=getattr(module, "train_loop_oldprec", False),
        ).detach()

    w_alpha = getattr(module, "W_alpha").detach()
    eps = (2.0 * w_alpha) / ((2.0 ** bits) - 1.0)
    return pact_mod.pact_quantize_signed_inference(
        weight,
        eps,
        w_alpha,
    ).detach()


def deploy_bias_semantic_from_module(module, eps_out: float | None) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if eps_out in (None, 0.0):
        return bias.detach().reshape(-1).to(dtype=torch.float32)
    stored = getattr(module, "_deploy_bias_integerization_report", None)
    if isinstance(stored, dict) and stored.get("bias_reconstructed_semantic") is not None:
        return torch.as_tensor(
            stored["bias_reconstructed_semantic"],
            dtype=torch.float32,
        ).detach().reshape(-1)
    if getattr(module, "_deploy_bias_integerized", False):
        return (bias.detach().reshape(-1).to(dtype=torch.float32) * float(eps_out)).detach()
    return bias.detach().reshape(-1).to(dtype=torch.float32)


def drift_share(step_mean_abs_diff: float | None, total_mean_abs_diff: float | None) -> float | None:
    if step_mean_abs_diff is None or total_mean_abs_diff is None or total_mean_abs_diff <= 0.0:
        return None
    return float(step_mean_abs_diff) / float(total_mean_abs_diff)


def int32_headroom(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.int64)
    if arr.size == 0:
        return {
            "min_headroom_to_int32": None,
            "max_headroom_to_int32": None,
        }
    i32 = np.iinfo(np.int32)
    return {
        "min_headroom_to_int32": int(np.min(arr - np.int64(i32.min))),
        "max_headroom_to_int32": int(np.max(np.int64(i32.max) - arr)),
    }


def conv_weight_saturation(module, weight_semantic: torch.Tensor, weight_eps: float | None) -> dict[str, Any]:
    bits = module_bits(module, "W_precision")
    quant_asymm = bool(getattr(module, "quant_asymm", False))
    if bits is None:
        return {"available": False}
    if quant_asymm:
        min_value = float(-float(getattr(module, "W_alpha").detach().cpu().reshape(-1)[0]))
        max_value = float(getattr(module, "W_beta").detach().cpu().reshape(-1)[0])
    else:
        alpha = float(getattr(module, "W_alpha").detach().cpu().reshape(-1)[0])
        min_value = -alpha
        max_value = alpha
    report = saturation_stats(
        weight_semantic.detach().cpu().numpy(),
        min_value=min_value,
        max_value=max_value,
        quantum=weight_eps,
    )
    report["bits"] = bits
    report["quant_asymm"] = quant_asymm
    return report


def stage41_conv1_per_channel_support(reference_models: dict[str, Any]) -> dict[str, Any]:
    fq_module = resolve_dotted_module(reference_models["model_fq"], "stage4.1.conv1")
    id_module = resolve_dotted_module(reference_models["model_id"], "stage4.1.conv1")
    context = reference_models.get("context_map", {}).get("stage4.1.conv1") or {}

    out_channels = int(getattr(fq_module, "out_channels", 0))
    fq_w_alpha = getattr(fq_module, "W_alpha", None)
    id_w_alpha = getattr(id_module, "W_alpha", None)
    id_eps_out_static = getattr(id_module, "eps_out_static", None)

    supported = True
    reasons: list[str] = []

    if fq_w_alpha is None or int(fq_w_alpha.numel()) != out_channels:
        supported = False
        reasons.append(
            "PACT_Conv2d exposes scalar W_alpha in the current NeMO path instead of one scale per output channel."
        )
    if id_eps_out_static is None or int(id_eps_out_static.numel()) != out_channels:
        supported = False
        reasons.append(
            "The integerized conv1 export path stores scalar eps_out_static rather than a per-channel output scale."
        )
    if isinstance(context.get("eps_out"), (int, float)):
        supported = False
        reasons.append(
            "The hybrid_follow semantic decode path currently assumes scalar eps_out for stage4.1.conv1."
        )

    return {
        "supported": supported,
        "operator_name": "stage4.1.conv1",
        "observed": {
            "out_channels": out_channels,
            "fq_W_alpha_shape": tensor_shape_list(fq_w_alpha),
            "id_W_alpha_shape": tensor_shape_list(id_w_alpha),
            "id_eps_out_static_shape": tensor_shape_list(id_eps_out_static),
            "context_eps_out": make_json_ready(context.get("eps_out")),
            "context_weight_eps": make_json_ready(context.get("weight_eps")),
        },
        "reason_if_unsupported": reasons,
    }


def build_stage41_conv1_decomposition(
    *,
    models: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    fp_probe = run_hybrid_follow_pytorch_probe(models["model_fp"], sample["float"])
    fq_probe = run_hybrid_follow_pytorch_probe(models["model_fq"], sample["float"])
    id_probe = run_hybrid_follow_pytorch_probe(models["model_id"], sample["staged"])

    conv_fp = resolve_dotted_module(models["model_fp"], "stage4.1.conv1")
    conv_fq = resolve_dotted_module(models["model_fq"], "stage4.1.conv1")
    conv_id = resolve_dotted_module(models["model_id"], "stage4.1.conv1")
    input_ctx = (models.get("context_map", {}).get("stage4.1.conv1_input") or {})
    conv_ctx = (models.get("context_map", {}).get("stage4.1.conv1") or {})

    fp_input = torch.as_tensor(fp_probe["tensors"]["stage4_1_conv1_input"], dtype=torch.float32)
    fq_input = torch.as_tensor(fq_probe["tensors"]["stage4_1_conv1_input"], dtype=torch.float32)
    id_input_raw = torch.as_tensor(id_probe["tensors"]["stage4_1_conv1_input"], dtype=torch.float32)
    fp_output = torch.as_tensor(fp_probe["tensors"]["stage4_1_conv1"], dtype=torch.float32)
    fq_output = torch.as_tensor(fq_probe["tensors"]["stage4_1_conv1"], dtype=torch.float32)
    id_output_raw = torch.as_tensor(id_probe["tensors"]["stage4_1_conv1"], dtype=torch.float32)

    eps_in = conv_ctx.get("eps_in")
    eps_out = conv_ctx.get("eps_out")
    weight_eps = conv_ctx.get("weight_eps")

    fp_weight = conv_fp.weight.detach().to(dtype=torch.float32)
    fp_bias = None
    if getattr(conv_fp, "bias", None) is not None:
        fp_bias = conv_fp.bias.detach().reshape(-1).to(dtype=torch.float32)

    id_weight_raw = conv_id.weight.detach().to(dtype=torch.float32)
    id_weight_semantic = (
        id_weight_raw * float(weight_eps)
        if weight_eps not in (None, 0.0)
        else id_weight_raw.clone()
    )
    id_bias_semantic = deploy_bias_semantic_from_module(conv_id, eps_out)

    id_input_semantic = (
        id_input_raw * float(eps_in)
        if eps_in not in (None, 0.0)
        else id_input_raw.clone()
    )
    id_output_semantic = (
        id_output_raw * float(eps_out)
        if eps_out not in (None, 0.0)
        else id_output_raw.clone()
    )

    activation_only = conv_like_module(conv_fp, id_input_semantic, fp_weight, fp_bias)
    activation_plus_weight = conv_like_module(conv_fp, id_input_semantic, id_weight_semantic, fp_bias)
    activation_weight_bias = conv_like_module(conv_fp, id_input_semantic, id_weight_semantic, id_bias_semantic)
    recomputed_accumulator = conv_like_module(
        conv_id,
        id_input_raw,
        id_weight_raw,
        None if getattr(conv_id, "bias", None) is None else conv_id.bias.detach().to(dtype=torch.float32),
    )

    total_fp_to_id = compare_arrays_rich(fp_output.detach().cpu().numpy(), id_output_semantic.detach().cpu().numpy())
    activation_step = compare_arrays_rich(fp_output.detach().cpu().numpy(), activation_only.detach().cpu().numpy())
    weight_step = compare_arrays_rich(activation_only.detach().cpu().numpy(), activation_plus_weight.detach().cpu().numpy())
    bias_step = compare_arrays_rich(activation_plus_weight.detach().cpu().numpy(), activation_weight_bias.detach().cpu().numpy())
    output_step = compare_arrays_rich(activation_weight_bias.detach().cpu().numpy(), id_output_semantic.detach().cpu().numpy())

    input_clip_bounds = input_ctx.get("semantic_clip_bounds") or {}
    fq_weight_semantic = quantized_weight_semantic_from_module(conv_fq)
    output_quantization_note = (
        "PACT_Conv2d does not apply a post-conv clamp/requant in this path; the final step is the decoded raw accumulator output."
    )

    return {
        "sample": {
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
        },
        "context": {
            "module_class": conv_id.__class__.__name__,
            "eps_in": eps_in,
            "eps_out": eps_out,
            "weight_eps": weight_eps,
            "activation_input_context": make_json_ready(input_ctx),
            "conv_output_context": make_json_ready(conv_ctx),
            "fq_weight_bits": module_bits(conv_fq, "W_precision"),
            "id_weight_bits": module_bits(conv_id, "W_precision"),
            "deploy_bias_integerized": bool(getattr(conv_id, "_deploy_bias_integerized", False)),
        },
        "tensors": {
            "fq_tensor": nested_tensor_payload(fq_output.detach().cpu().numpy()),
            "id_raw_integer_tensor": nested_tensor_payload(id_output_raw.detach().cpu().numpy()),
            "decoded_semantic_tensor": nested_tensor_payload(id_output_semantic.detach().cpu().numpy()),
        },
        "accumulator": {
            "captured_raw_stats": tensor_stats(id_output_raw.detach().cpu().numpy()),
            "recomputed_raw_stats": tensor_stats(recomputed_accumulator.detach().cpu().numpy()),
            "captured_vs_recomputed_raw": compare_arrays_rich(
                id_output_raw.detach().cpu().numpy(),
                recomputed_accumulator.detach().cpu().numpy(),
            ),
            "headroom": int32_headroom(np.rint(id_output_raw.detach().cpu().numpy()).astype(np.int64)),
        },
        "saturation_and_clipping": {
            "activation_input": saturation_stats(
                id_input_semantic.detach().cpu().numpy(),
                min_value=input_clip_bounds.get("min"),
                max_value=input_clip_bounds.get("max"),
                quantum=eps_in,
            ),
            "activation_input_fq_vs_id_semantic": compare_arrays_rich(
                fq_input.detach().cpu().numpy(),
                id_input_semantic.detach().cpu().numpy(),
            ),
            "weight_quantization": conv_weight_saturation(conv_fq, fq_weight_semantic, weight_eps),
            "bias_quantization": (
                None
                if fp_bias is None or id_bias_semantic is None
                else {
                    "fp_bias_stats": tensor_stats(fp_bias.detach().cpu().numpy()),
                    "id_bias_semantic_stats": tensor_stats(id_bias_semantic.detach().cpu().numpy()),
                    "fp_vs_id_bias": compare_arrays_rich(
                        fp_bias.detach().cpu().numpy(),
                        id_bias_semantic.detach().cpu().numpy(),
                    ),
                }
            ),
            "output": {
                "available": False,
                "note": output_quantization_note,
            },
        },
        "drift_breakdown": {
            "observed": {
                "fp_vs_fq": compare_arrays_rich(fp_output.detach().cpu().numpy(), fq_output.detach().cpu().numpy()),
                "fq_vs_id": compare_arrays_rich(fq_output.detach().cpu().numpy(), id_output_semantic.detach().cpu().numpy()),
                "fp_vs_id": total_fp_to_id,
            },
            "component_stages": {
                "fp_reference": tensor_stats(fp_output.detach().cpu().numpy()),
                "after_activation_quantizer": tensor_stats(activation_only.detach().cpu().numpy()),
                "after_weight_quantization": tensor_stats(activation_plus_weight.detach().cpu().numpy()),
                "after_bias_quantization": tensor_stats(activation_weight_bias.detach().cpu().numpy()),
                "final_decoded_id": tensor_stats(id_output_semantic.detach().cpu().numpy()),
            },
            "contributions": {
                "activation_quantizer": {
                    "step_drift": activation_step,
                    "share_of_total_mean_abs_diff": drift_share(
                        activation_step.get("mean_abs_diff"),
                        total_fp_to_id.get("mean_abs_diff"),
                    ),
                },
                "weight_quantization": {
                    "step_drift": weight_step,
                    "share_of_total_mean_abs_diff": drift_share(
                        weight_step.get("mean_abs_diff"),
                        total_fp_to_id.get("mean_abs_diff"),
                    ),
                },
                "bias_quantization": {
                    "step_drift": bias_step,
                    "share_of_total_mean_abs_diff": drift_share(
                        bias_step.get("mean_abs_diff"),
                        total_fp_to_id.get("mean_abs_diff"),
                    ),
                },
                "output_requantization": {
                    "step_drift": output_step,
                    "share_of_total_mean_abs_diff": drift_share(
                        output_step.get("mean_abs_diff"),
                        total_fp_to_id.get("mean_abs_diff"),
                    ),
                    "note": output_quantization_note,
                },
            },
        },
    }


def conv1_decomposition_markdown(report: dict[str, Any]) -> str:
    ctx = report.get("context") or {}
    drifts = ((report.get("drift_breakdown") or {}).get("observed") or {})
    contrib = ((report.get("drift_breakdown") or {}).get("contributions") or {})
    accum = report.get("accumulator") or {}
    clipping = report.get("saturation_and_clipping") or {}
    lines = [
        "# stage4.1.conv1 Decomposition Report",
        "",
        f"- Sample: `{((report.get('sample') or {}).get('image_name'))}`",
        f"- eps_in: `{ctx.get('eps_in')}`",
        f"- eps_out: `{ctx.get('eps_out')}`",
        f"- weight_eps: `{ctx.get('weight_eps')}`",
        f"- deploy_bias_integerized: `{ctx.get('deploy_bias_integerized')}`",
        "",
        "## Observed Drift",
        "",
        "- fp_vs_fq mean_abs_diff=`{}`".format(((drifts.get("fp_vs_fq") or {}).get("mean_abs_diff"))),
        "- fq_vs_id mean_abs_diff=`{}` max_abs_diff=`{}` abs_mean_ratio=`{}` cosine=`{}`".format(
            ((drifts.get("fq_vs_id") or {}).get("mean_abs_diff")),
            ((drifts.get("fq_vs_id") or {}).get("max_abs_diff")),
            ((drifts.get("fq_vs_id") or {}).get("abs_mean_ratio")),
            ((drifts.get("fq_vs_id") or {}).get("cosine_similarity")),
        ),
        "- fp_vs_id mean_abs_diff=`{}`".format(((drifts.get("fp_vs_id") or {}).get("mean_abs_diff"))),
        "",
        "## Accumulator",
        "",
        f"- Captured raw stats: `{make_json_ready(accum.get('captured_raw_stats'))}`",
        f"- Captured vs recomputed raw: `{make_json_ready(accum.get('captured_vs_recomputed_raw'))}`",
        f"- Int32 headroom: `{make_json_ready(accum.get('headroom'))}`",
        "",
        "## Clipping",
        "",
        f"- Activation input: `{make_json_ready(clipping.get('activation_input'))}`",
        f"- Activation fq_vs_id semantic: `{make_json_ready(clipping.get('activation_input_fq_vs_id_semantic'))}`",
        f"- Weight quantization: `{make_json_ready(clipping.get('weight_quantization'))}`",
        f"- Bias quantization: `{make_json_ready(clipping.get('bias_quantization'))}`",
        f"- Output: `{make_json_ready(clipping.get('output'))}`",
        "",
        "## Contribution Breakdown",
        "",
    ]
    for key in ("activation_quantizer", "weight_quantization", "bias_quantization", "output_requantization"):
        row = contrib.get(key) or {}
        lines.append(
            "- {}: step_mean_abs_diff=`{}` share_of_total=`{}` note=`{}`".format(
                key,
                ((row.get("step_drift") or {}).get("mean_abs_diff")),
                row.get("share_of_total_mean_abs_diff"),
                row.get("note"),
            )
        )
    return "\n".join(lines)


def flatten_float_arrays(tensors: list[Any]) -> np.ndarray:
    if not tensors:
        return np.asarray([], dtype=np.float64)
    rows = []
    for tensor in tensors:
        arr = np.asarray(tensor, dtype=np.float64).reshape(-1)
        if arr.size:
            rows.append(arr)
    if not rows:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(rows, axis=0)


def collect_module_io_samples(model, module_name: str, sample_tensors: list[torch.Tensor]) -> dict[str, list[np.ndarray]]:
    module = resolve_dotted_module(model, module_name)
    captures = {
        "input": [],
        "output": [],
    }

    def pre_hook(_module, inputs):
        if inputs:
            captures["input"].append(inputs[0].detach().cpu().numpy())

    def hook(_module, _inputs, output):
        captures["output"].append(output.detach().cpu().numpy())

    pre_handle = module.register_forward_pre_hook(pre_hook)
    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for sample_tensor in sample_tensors:
                _ = model(sample_tensor)
    finally:
        pre_handle.remove()
        handle.remove()
    return captures


def activation_histogram_summary(relu_values: np.ndarray, alpha: float) -> dict[str, Any]:
    if relu_values.size == 0 or alpha <= 0.0:
        return {
            "count": int(relu_values.size),
            "bins": {},
        }
    normalized = relu_values / float(alpha)
    bin_specs = [
        ("zero", normalized <= 0.0),
        ("0_to_25pct", (normalized > 0.0) & (normalized <= 0.25)),
        ("25_to_50pct", (normalized > 0.25) & (normalized <= 0.50)),
        ("50_to_75pct", (normalized > 0.50) & (normalized <= 0.75)),
        ("75_to_90pct", (normalized > 0.75) & (normalized <= 0.90)),
        ("90_to_95pct", (normalized > 0.90) & (normalized <= 0.95)),
        ("95_to_100pct", (normalized > 0.95) & (normalized <= 1.00)),
        ("above_alpha", normalized > 1.00),
    ]
    return {
        "count": int(relu_values.size),
        "bins": {
            name: {
                "fraction": float(np.mean(mask)),
                "count": int(np.sum(mask)),
            }
            for name, mask in bin_specs
        },
    }


def activation_value_percentiles(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {}
    percentiles = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    return {
        f"p{str(p).replace('.', '_')}": float(np.percentile(values, p))
        for p in percentiles
    }


def activation_quantizer_dataset_audit(
    *,
    module,
    captures: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    alpha = float(getattr(module, "alpha").detach().cpu().item())
    precision_bits = int(module.precision.get_bits())
    pre_values = flatten_float_arrays(captures.get("input") or [])
    output_values = flatten_float_arrays(captures.get("output") or [])
    relu_values = np.maximum(pre_values, 0.0)
    quant_step = float(alpha / max((2.0 ** precision_bits) - 1.0, 1.0))
    negative_fraction = float(np.mean(pre_values < 0.0)) if pre_values.size else None
    above_alpha_fraction = float(np.mean(relu_values > alpha)) if relu_values.size else None
    clipped_fraction = (
        float(np.mean((pre_values < 0.0) | (relu_values > alpha)))
        if pre_values.size
        else None
    )
    return {
        "input_stats": tensor_stats(pre_values),
        "relu_semantic_stats": tensor_stats(relu_values),
        "output_stats": tensor_stats(output_values),
        "quant_step": quant_step,
        "negative_fraction": negative_fraction,
        "above_alpha_fraction": above_alpha_fraction,
        "clipped_fraction": clipped_fraction,
        "saturation_histogram": activation_histogram_summary(relu_values, alpha),
        "relu_percentiles": activation_value_percentiles(relu_values),
        "output_percentiles": activation_value_percentiles(output_values),
        "relu_vs_output": (
            None
            if relu_values.size == 0 or output_values.size == 0
            else compare_arrays_rich(relu_values, output_values)
        ),
    }


def build_stage41_conv1_input_activation_audit(
    *,
    models: dict[str, Any],
    calib_samples: list[dict[str, Any]],
    known_bad_sample: dict[str, Any],
    rep16_samples: list[dict[str, Any]],
    hard_case_names: list[str],
) -> dict[str, Any]:
    module_name = "stage4.0.out_relu"
    module = resolve_dotted_module(models["model_fq"], module_name)
    precision = getattr(module, "precision", None)
    alpha = float(getattr(module, "alpha").detach().cpu().item())
    precision_bits = int(precision.get_bits())
    positive_flag = None if precision is None else bool(getattr(precision, "positive", False))

    hard_case_set = set(hard_case_names)
    hard_case_samples = [sample for sample in rep16_samples if sample["image_name"] in hard_case_set]
    dataset_tensors = {
        "real_calibration": [sample["tensor"] for sample in calib_samples],
        "known_bad": [known_bad_sample["float"]],
        "rep16": [sample["float"] for sample in rep16_samples],
        "hard_case": [sample["float"] for sample in hard_case_samples],
    }

    datasets = {}
    fixed_eval_max = alpha
    fixed_eval_p99_9 = alpha
    for dataset_name, tensors in dataset_tensors.items():
        captures = collect_module_io_samples(models["model_fq"], module_name, tensors)
        dataset_report = activation_quantizer_dataset_audit(module=module, captures=captures)
        datasets[dataset_name] = dataset_report
        if dataset_name in {"known_bad", "rep16", "hard_case"}:
            relu_percentiles = dataset_report.get("relu_percentiles") or {}
            fixed_eval_max = max(fixed_eval_max, float(relu_percentiles.get("p100") or alpha))
            fixed_eval_p99_9 = max(fixed_eval_p99_9, float(relu_percentiles.get("p99_9") or alpha))

    current_alpha = alpha
    recommended_alpha = max(current_alpha * 1.25, fixed_eval_p99_9)
    return {
        "module_name": module_name,
        "feeds_operator": "stage4.1.conv1",
        "module_class": module.__class__.__name__,
        "alpha": current_alpha,
        "precision_bits": precision_bits,
        "precision_positive_flag": positive_flag,
        "signed": False,
        "unsigned": True,
        "symmetric": False,
        "asymmetric": True,
        "relu_aware": True,
        "clip_min": 0.0,
        "clip_max": current_alpha,
        "wastes_dynamic_range_on_negative_values": False,
        "behavior_summary": (
            "PACT_Act replaces ReLU here and quantizes/clamps to [0, alpha], so the activation is already "
            "unsigned and ReLU-aware rather than spending code points on negative values."
        ),
        "datasets": make_json_ready(datasets),
        "recommendations": {
            "widened_alpha": float(recommended_alpha),
            "widened_alpha_scale": float(recommended_alpha / max(current_alpha, 1e-12)),
            "higher_precision_bits": 10,
            "bypass_fake_quant_mode": "fake_quant_only_non_deployment",
            "fixed_eval_max_relu_value": float(fixed_eval_max),
            "fixed_eval_p99_9_relu_value": float(fixed_eval_p99_9),
        },
    }


def activation_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# stage4.1.conv1 Input Activation Audit",
        "",
        f"- Module: `{report.get('module_name')}`",
        f"- Feeds operator: `{report.get('feeds_operator')}`",
        f"- Class: `{report.get('module_class')}`",
        f"- alpha: `{report.get('alpha')}`",
        f"- precision_bits: `{report.get('precision_bits')}`",
        f"- unsigned: `{report.get('unsigned')}`",
        f"- symmetric: `{report.get('symmetric')}`",
        f"- relu_aware: `{report.get('relu_aware')}`",
        f"- clip_min/max: `{report.get('clip_min')}` / `{report.get('clip_max')}`",
        f"- wastes_negative_dynamic_range: `{report.get('wastes_dynamic_range_on_negative_values')}`",
        f"- behavior: `{report.get('behavior_summary')}`",
        "",
        "## Dataset Summary",
        "",
    ]
    for dataset_name, dataset in (report.get("datasets") or {}).items():
        lines.extend(
            [
                f"### {dataset_name}",
                "",
                f"- clipped_fraction: `{dataset.get('clipped_fraction')}`",
                f"- negative_fraction: `{dataset.get('negative_fraction')}`",
                f"- above_alpha_fraction: `{dataset.get('above_alpha_fraction')}`",
                f"- relu_percentiles: `{make_json_ready(dataset.get('relu_percentiles'))}`",
                f"- saturation_histogram: `{make_json_ready(dataset.get('saturation_histogram'))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendations",
            "",
            f"- widened_alpha: `{((report.get('recommendations') or {}).get('widened_alpha'))}`",
            f"- widened_alpha_scale: `{((report.get('recommendations') or {}).get('widened_alpha_scale'))}`",
            f"- higher_precision_bits: `{((report.get('recommendations') or {}).get('higher_precision_bits'))}`",
            f"- bypass_fake_quant_mode: `{((report.get('recommendations') or {}).get('bypass_fake_quant_mode'))}`",
        ]
    )
    return "\n".join(lines)


def build_stage41_eps_audit(models: dict[str, Any]) -> dict[str, Any]:
    ctx = models.get("context_map") or {}
    input_ctx = ctx.get("stage4.1.conv1_input") or {}
    conv_ctx = ctx.get("stage4.1.conv1") or {}
    skip_module = resolve_dotted_module(models["model_id"], "stage4.0.out_relu")
    conv_module = resolve_dotted_module(models["model_id"], "stage4.1.conv1")
    explicit_input_eps = None
    for attr_name in ("eps_out", "eps_out_static", "eps_static"):
        if hasattr(skip_module, attr_name):
            value = getattr(skip_module, attr_name)
            if torch.is_tensor(value):
                explicit_input_eps = float(value.detach().cpu().reshape(-1)[0])
                explicit_input_source = attr_name
                break
    else:
        explicit_input_source = "alpha/precision_or_fallback"
        explicit_input_eps = float(input_ctx.get("eps_out")) if input_ctx.get("eps_out") is not None else None

    explicit_conv_eps = None
    for attr_name in ("eps_out", "eps_out_static", "eps_static"):
        if hasattr(conv_module, attr_name):
            value = getattr(conv_module, attr_name)
            if torch.is_tensor(value):
                explicit_conv_eps = float(value.detach().cpu().reshape(-1)[0])
                explicit_conv_source = attr_name
                break
    else:
        explicit_conv_source = "alpha/precision_or_fallback"
        explicit_conv_eps = float(conv_ctx.get("eps_out")) if conv_ctx.get("eps_out") is not None else None

    return {
        "stage4.1.conv1_input": {
            "eps_out": input_ctx.get("eps_out"),
            "eps_out_source": input_ctx.get("eps_out_source"),
            "explicit_module_eps_out": explicit_input_eps,
            "explicit_module_eps_out_source": explicit_input_source,
        },
        "stage4.1.conv1": {
            "eps_in": conv_ctx.get("eps_in"),
            "eps_in_source": conv_ctx.get("eps_in_source"),
            "eps_out": conv_ctx.get("eps_out"),
            "eps_out_source": conv_ctx.get("eps_out_source"),
            "explicit_module_eps_out": explicit_conv_eps,
            "explicit_module_eps_out_source": explicit_conv_source,
        },
        "consistency": {
            "input_eps_matches_conv_eps_in": (
                None
                if input_ctx.get("eps_out") is None or conv_ctx.get("eps_in") is None
                else abs(float(input_ctx["eps_out"]) - float(conv_ctx["eps_in"]))
            ),
            "prefer_explicit_per_module_eps_mapping": True,
            "fallback_in_use": bool(
                str(input_ctx.get("eps_out_source") or "").endswith("_fallback")
                or str(conv_ctx.get("eps_out_source") or "").endswith("_fallback")
            ),
        },
    }


def eps_audit_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# stage4.1.conv1 Eps Audit",
            "",
            f"- stage4.1.conv1_input: `{make_json_ready(report.get('stage4.1.conv1_input'))}`",
            f"- stage4.1.conv1: `{make_json_ready(report.get('stage4.1.conv1'))}`",
            f"- consistency: `{make_json_ready(report.get('consistency'))}`",
        ]
    )


def select_qat_checkpoint(qat_output_dir: Path) -> Path:
    for name in ("hybrid_follow_best_x.pth", "hybrid_follow_best_follow_score.pth"):
        candidate = qat_output_dir / name
        if candidate.is_file():
            return candidate
    epoch_ckpts = sorted(qat_output_dir.glob("hybrid_follow_epoch_*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    raise FileNotFoundError(f"No QAT checkpoint found in {qat_output_dir}")


def ensure_python_module(import_stmt: str, pip_spec: str, log_handle) -> None:
    check_cmd = [sys.executable, "-c", import_stmt]
    check_result = subprocess.run(
        check_cmd,
        cwd=str(PROJECT_DIR.parent),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check_result.returncode == 0:
        return
    print(f"[focused_qat] Installing missing dependency for `{import_stmt}`: {pip_spec}", file=log_handle)
    log_handle.flush()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", pip_spec],
        cwd=str(PROJECT_DIR.parent),
        env=os.environ.copy(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
    )


def extract_activation_alpha_overrides_from_qat_checkpoint(
    ckpt_path: Path,
    module_names: list[str],
) -> dict[str, dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    overrides: dict[str, dict[str, Any]] = {}
    for module_name in module_names:
        alpha_key = f"{module_name}.alpha"
        if alpha_key not in state_dict:
            continue
        alpha_value = state_dict[alpha_key]
        if torch.is_tensor(alpha_value):
            alpha = float(alpha_value.detach().cpu().reshape(-1)[0])
        else:
            alpha = float(alpha_value)
        overrides[module_name] = {
            "alpha": alpha,
            "policy_name": f"qat_learned_alpha::{module_name}",
        }
    return overrides


def run_focused_qat_trial(
    *,
    args,
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    known_bad_sample: dict[str, Any],
    rep16_samples: list[dict[str, Any]],
    thresholds: Thresholds,
    local_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    trial_dir = output_dir / "trials" / "focused_qat_stage4_heads"
    trial_dir.mkdir(parents=True, exist_ok=True)
    qat_output_dir = trial_dir / "trained_ckpt"
    train_log_path = trial_dir / "qat_train.log"
    command = [
        sys.executable,
        str(PROJECT_DIR / "train.py"),
        "--model-type",
        "hybrid_follow",
        "--init-ckpt",
        str(export_args.ckpt),
        "--stage4-heads-only",
        "--quant-aware-finetune",
        "--activation-range-regularization",
        "--epochs",
        str(args.qat_epochs),
        "--batch_size",
        str(args.qat_batch_size),
        "--lr",
        str(args.qat_lr),
        "--num_workers",
        str(args.qat_num_workers),
        "--qat-bits",
        str(args.bits),
        "--qat-calib-batches",
        str(args.qat_calib_batches),
        "--activation-range-reg-weight",
        str(args.qat_activation_range_reg_weight),
        "--max-train-batches",
        str(args.qat_max_train_batches),
        "--max-val-batches",
        str(args.qat_max_val_batches),
        "--output_dir",
        str(qat_output_dir),
    ]

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        ensure_python_module("from pycocotools.coco import COCO", "pycocotools==2.0.7", log_handle)
        subprocess.run(
            command,
            cwd=str(PROJECT_DIR.parent),
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    qat_ckpt = select_qat_checkpoint(qat_output_dir)
    qat_export_args = deepcopy(export_args)
    qat_export_args.ckpt = str(qat_ckpt)

    spec = PolicySpec(
        name="focused_qat_stage4_heads",
        description=(
            "Focused QAT fallback: init from current float checkpoint, freeze earlier layers, "
            "train stage4+heads only, and keep activation-range regularization enabled."
        ),
        operator_name="stage4.1.conv1",
        family="focused_qat",
        search_context={
            "train_command": command,
            "train_log_path": str(train_log_path),
            "qat_checkpoint": str(qat_ckpt),
        },
    )
    trial = evaluate_trial(
        export_args=qat_export_args,
        device=device,
        calib_samples=calib_samples,
        spec=spec,
        known_bad_sample=known_bad_sample,
        rep16_samples=rep16_samples,
        thresholds=thresholds,
        local_threshold=local_threshold,
        output_dir=output_dir,
    )
    trial["qat_train_log"] = str(train_log_path)
    trial["qat_checkpoint"] = str(qat_ckpt)
    trial["qat_command"] = command
    return trial


def run_activation_only_qat_trial(
    *,
    args,
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    known_bad_sample: dict[str, Any],
    rep16_samples: list[dict[str, Any]],
    thresholds: Thresholds,
    local_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    module_names = ["stage4.0.out_relu"]
    trial_dir = output_dir / "trials" / "activation_only_qat_stage4_1_input"
    trial_dir.mkdir(parents=True, exist_ok=True)
    qat_output_dir = trial_dir / "trained_ckpt"
    train_log_path = trial_dir / "qat_train.log"
    command = [
        sys.executable,
        str(PROJECT_DIR / "train.py"),
        "--model-type",
        "hybrid_follow",
        "--init-ckpt",
        str(export_args.ckpt),
        "--quant-aware-finetune",
        "--activation-range-regularization",
        "--epochs",
        str(args.qat_epochs),
        "--batch_size",
        str(args.qat_batch_size),
        "--lr",
        str(args.qat_lr),
        "--num_workers",
        str(args.qat_num_workers),
        "--qat-bits",
        str(args.bits),
        "--qat-calib-batches",
        str(args.qat_calib_batches),
        "--activation-range-reg-weight",
        str(args.qat_activation_range_reg_weight),
        "--max-train-batches",
        str(args.qat_max_train_batches),
        "--max-val-batches",
        str(args.qat_max_val_batches),
        "--qat-train-activation-modules",
        ",".join(module_names),
        "--output_dir",
        str(qat_output_dir),
    ]

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        ensure_python_module("from pycocotools.coco import COCO", "pycocotools==2.0.7", log_handle)
        subprocess.run(
            command,
            cwd=str(PROJECT_DIR.parent),
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    qat_ckpt = select_qat_checkpoint(qat_output_dir)
    learned_activation_overrides = extract_activation_alpha_overrides_from_qat_checkpoint(
        qat_ckpt,
        module_names,
    )
    if not learned_activation_overrides:
        raise RuntimeError(
            "Activation-only QAT did not emit any learned activation alpha overrides for "
            f"{module_names}."
        )
    spec = PolicySpec(
        name="activation_only_qat_stage4_1_input",
        description=(
            "Activation-only QAT: freeze weights everywhere and fine-tune only the stage4.1.conv1 input "
            "activation quantizer parameters."
        ),
        operator_name="stage4.1.conv1",
        family="activation_only_qat",
        activation_overrides=deepcopy(learned_activation_overrides),
        search_context={
            "train_command": command,
            "train_log_path": str(train_log_path),
            "qat_checkpoint": str(qat_ckpt),
            "module_names": module_names,
            "learned_activation_overrides": make_json_ready(learned_activation_overrides),
        },
    )
    trial = evaluate_trial(
        export_args=export_args,
        device=device,
        calib_samples=calib_samples,
        spec=spec,
        known_bad_sample=known_bad_sample,
        rep16_samples=rep16_samples,
        thresholds=thresholds,
        local_threshold=local_threshold,
        output_dir=output_dir,
    )
    trial["qat_train_log"] = str(train_log_path)
    trial["qat_checkpoint"] = str(qat_ckpt)
    trial["qat_command"] = command
    trial["learned_activation_overrides"] = make_json_ready(learned_activation_overrides)
    return trial


def run_stage41_architecture_ablation_trial(
    *,
    args,
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    known_bad_sample: dict[str, Any],
    rep16_samples: list[dict[str, Any]],
    thresholds: Thresholds,
    local_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    trial_dir = output_dir / "trials" / "stage4_1_single_conv_ablation"
    trial_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = trial_dir / "trained_ckpt"
    train_log_path = trial_dir / "ablation_train.log"
    command = [
        sys.executable,
        str(PROJECT_DIR / "train.py"),
        "--model-type",
        "hybrid_follow",
        "--stage4-1-ablation",
        "single_conv",
        "--init-ckpt",
        str(export_args.ckpt),
        "--stage4-heads-only",
        "--quant-aware-finetune",
        "--activation-range-regularization",
        "--epochs",
        str(args.qat_epochs),
        "--batch_size",
        str(args.qat_batch_size),
        "--lr",
        str(args.qat_lr),
        "--num_workers",
        str(args.qat_num_workers),
        "--qat-bits",
        str(args.bits),
        "--qat-calib-batches",
        str(args.qat_calib_batches),
        "--activation-range-reg-weight",
        str(args.qat_activation_range_reg_weight),
        "--max-train-batches",
        str(args.qat_max_train_batches),
        "--max-val-batches",
        str(args.qat_max_val_batches),
        "--output_dir",
        str(train_output_dir),
    ]

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        ensure_python_module("from pycocotools.coco import COCO", "pycocotools==2.0.7", log_handle)
        subprocess.run(
            command,
            cwd=str(PROJECT_DIR.parent),
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    ablation_ckpt = select_qat_checkpoint(train_output_dir)
    ablation_export_args = deepcopy(export_args)
    ablation_export_args.ckpt = str(ablation_ckpt)
    ablation_export_args.stage4_1_ablation = "single_conv"

    spec = PolicySpec(
        name="stage4_1_single_conv_ablation",
        description=(
            "Tiny architecture ablation: simplify only stage4.1 to a single-conv residual block, "
            "then run the same bounded stage4+heads QAT setup before export evaluation."
        ),
        operator_name="stage4.1.conv1",
        family="architecture_ablation",
        search_context={
            "train_command": command,
            "train_log_path": str(train_log_path),
            "ablation_checkpoint": str(ablation_ckpt),
            "stage4_1_ablation": "single_conv",
        },
    )
    trial = evaluate_trial(
        export_args=ablation_export_args,
        device=device,
        calib_samples=calib_samples,
        spec=spec,
        known_bad_sample=known_bad_sample,
        rep16_samples=rep16_samples,
        thresholds=thresholds,
        local_threshold=local_threshold,
        output_dir=output_dir,
    )
    trial["ablation_train_log"] = str(train_log_path)
    trial["ablation_checkpoint"] = str(ablation_ckpt)
    trial["ablation_command"] = command
    trial["stage4_1_ablation"] = "single_conv"
    return trial


def pair_score(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict) or report.get("status") == "skipped":
        return None
    decoded = report.get("decoded_abs_diff") or {}
    return float(
        decoded.get("x_offset", 0.0)
        + decoded.get("size_proxy", 0.0)
        + decoded.get("visibility_confidence", 0.0)
    )


def stage_result_dict(stage_result) -> dict[str, Any]:
    return stage_result.to_dict()


def make_stage_result(
    *,
    key: str,
    label: str,
    stage_tag: str,
    representation: str,
    raw_native: np.ndarray | list[int] | list[float],
    integer_output_scale: float = FINAL_OUTPUT_SCALE,
):
    return to_stage_result(
        key=key,
        label=label,
        status="ok",
        stage_tag=stage_tag,
        representation=representation,
        raw_native=raw_native,
        integer_output_scale=integer_output_scale,
        metadata={},
    )


def boundary_label(boundary_key: str | None) -> str | None:
    for key, label, _side in BOUNDARY_SPECS:
        if key == boundary_key:
            return label
    return None


def boundary_side(boundary_key: str | None) -> str | None:
    for key, _label, side in BOUNDARY_SPECS:
        if key == boundary_key:
            return side
    return None


def boundary_position(boundary_key: str | None) -> int:
    if boundary_key is None:
        return len(BOUNDARY_SPECS)
    for idx, (key, _label, _side) in enumerate(BOUNDARY_SPECS):
        if key == boundary_key:
            return idx
    return len(BOUNDARY_SPECS)


def alias_position(alias: str | None) -> int:
    if alias is None:
        return len(LOCAL_TAP_PRIORITY)
    try:
        return LOCAL_TAP_PRIORITY.index(alias)
    except ValueError:
        return len(LOCAL_TAP_PRIORITY)


def primary_score_from_row(row: dict[str, Any]) -> float | None:
    return (
        row.get("scores", {}).get("fp_to_gvsoc")
        if row.get("scores", {}).get("fp_to_gvsoc") is not None
        else row.get("scores", {}).get("fp_to_onnx")
    )


def recover_summary_dir_path(path_value: str | Path | None, *, summary_root: Path, kind: str) -> Path | None:
    candidate = resolve_repo_path(path_value)
    if candidate is not None and candidate.exists():
        return candidate

    if path_value is None:
        return None

    raw = Path(str(path_value))
    leaf = raw.name
    if not leaf:
        return None

    if kind == "sample_dir":
        rebased = summary_root / "application_validation" / leaf
        if rebased.exists():
            return rebased
    if kind == "stage_drift_dir":
        rebased = summary_root / "stage_drift" / leaf
        if rebased.exists():
            return rebased

    return candidate


def load_application_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_root = summary_path.parent.resolve()
    mapping: dict[str, dict[str, Any]] = {}
    for row in payload.get("results") or []:
        image_name = row.get("image_name")
        if not image_name and row.get("image_path"):
            image_name = Path(str(row["image_path"])).name
        if not image_name:
            continue
        sample_dir = recover_summary_dir_path(
            row.get("application_sample_dir") or row.get("sample_dir"),
            summary_root=summary_root,
            kind="sample_dir",
        )
        stage_drift_dir = recover_summary_dir_path(
            row.get("stage_drift_dir"),
            summary_root=summary_root,
            kind="stage_drift_dir",
        )
        if sample_dir is None:
            continue
        gvsoc_log_path = resolve_repo_path(row.get("gvsoc_log"))
        if gvsoc_log_path is None:
            for candidate_name in ("gvsoc.log", "run_aideck_val.log"):
                candidate = sample_dir / candidate_name
                if candidate.is_file():
                    gvsoc_log_path = candidate
                    break
        mapping[image_name] = {
            **row,
            "image_name": image_name,
            "sample_dir": str(sample_dir),
            "stage_drift_dir": None if stage_drift_dir is None else str(stage_drift_dir),
            "golden_path": str(sample_dir / "output.txt"),
            "gvsoc_json_path": str(sample_dir / "gvsoc_final_tensor.json"),
            "gvsoc_log_path": None if gvsoc_log_path is None else str(gvsoc_log_path),
        }
    return {
        "payload": payload,
        "mapping": mapping,
        "summary_root": str(summary_root),
    }


def read_fixed_runtime_artifacts(
    summary_row: dict[str, Any],
    *,
    runtime_cache_dir: Path,
    layer_manifest_path: Path | None,
    allow_missing_runtime_compare: bool,
) -> dict[str, Any]:
    sample_dir = resolve_repo_path(summary_row.get("sample_dir"))
    if sample_dir is None or not sample_dir.is_dir():
        raise FileNotFoundError(f"Application sample dir not found: {summary_row.get('sample_dir')}")

    golden_path = resolve_repo_path(summary_row.get("golden_path")) or (sample_dir / "output.txt")
    gvsoc_json_path = resolve_repo_path(summary_row.get("gvsoc_json_path")) or (sample_dir / "gvsoc_final_tensor.json")
    gvsoc_log_path = resolve_repo_path(summary_row.get("gvsoc_log_path"))
    if gvsoc_log_path is None:
        for candidate_name in ("gvsoc.log", "run_aideck_val.log"):
            candidate = sample_dir / candidate_name
            if candidate.is_file():
                gvsoc_log_path = candidate
                break
    if gvsoc_log_path is None:
        gvsoc_log_path = sample_dir / "gvsoc.log"
    if not golden_path.is_file():
        raise FileNotFoundError(f"Golden output artifact not found: {golden_path}")
    if not gvsoc_json_path.is_file():
        raise FileNotFoundError(f"GVSOC final tensor artifact not found: {gvsoc_json_path}")

    gvsoc_payload = json.loads(gvsoc_json_path.read_text(encoding="utf-8"))
    runtime_compare = None
    runtime_compare_path = None

    stage_drift_dir = resolve_repo_path(summary_row.get("stage_drift_dir"))
    if stage_drift_dir is not None:
        candidate = stage_drift_dir / "runtime_layer_compare.json"
        if candidate.is_file():
            runtime_compare = json.loads(candidate.read_text(encoding="utf-8"))
            runtime_compare_path = candidate

    if runtime_compare is None and layer_manifest_path is not None and gvsoc_log_path.is_file():
        runtime_output_dir = runtime_cache_dir / sanitize_name(Path(summary_row["image_name"]).stem)
        runtime_compare = compare_runtime_layers(
            gvsoc_log_path=gvsoc_log_path,
            layer_manifest_path=layer_manifest_path,
            output_dir=runtime_output_dir,
        )
        runtime_compare_path = runtime_output_dir / "runtime_layer_compare.json"

    if runtime_compare is None and not allow_missing_runtime_compare:
        raise RuntimeError(
            f"Runtime layer comparison is required for {summary_row['image_name']}. "
            "Provide --layer-manifest or rerun application validation with strict stage drift."
        )

    return {
        "sample_dir": str(sample_dir),
        "golden_path": str(golden_path),
        "gvsoc_json_path": str(gvsoc_json_path),
        "gvsoc_log_path": str(gvsoc_log_path),
        "golden_values": [int(value) for value in parse_numeric_artifact(golden_path)],
        "gvsoc_values": [int(value) for value in gvsoc_payload.get("values", [])],
        "runtime_layer_compare": runtime_compare,
        "runtime_layer_compare_path": None if runtime_compare_path is None else str(runtime_compare_path),
    }


def build_rep16_samples(
    *,
    rep16_dir: Path,
    application_summary: dict[str, Any],
    image_size: tuple[int, int],
    device: torch.device,
    runtime_cache_dir: Path,
    layer_manifest_path: Path | None,
    allow_missing_runtime_compare: bool,
) -> list[dict[str, Any]]:
    image_paths = discover_images(rep16_dir, limit=None)
    if not image_paths:
        raise FileNotFoundError(f"No rep16 images found under {rep16_dir}")

    samples = []
    for image_path in image_paths:
        summary_row = application_summary["mapping"].get(image_path.name)
        if summary_row is None:
            raise KeyError(
                f"Application summary does not contain rep16 image '{image_path.name}'. "
                f"Expected a fixed GVSOC sample directory for {image_path}."
            )
        sample = load_hybrid_follow_sample(image_path, image_size, device)
        sample["application_summary_row"] = make_json_ready(summary_row)
        sample.update(
            read_fixed_runtime_artifacts(
                summary_row,
                runtime_cache_dir=runtime_cache_dir,
                layer_manifest_path=layer_manifest_path,
                allow_missing_runtime_compare=allow_missing_runtime_compare,
            )
        )
        samples.append(sample)
    return samples


def compact_runtime_layer_compare(runtime_compare: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime_compare, dict):
        return None
    first = runtime_compare.get("first_divergent_layer")
    return {
        "layer_count": int(runtime_compare.get("layer_count") or 0),
        "first_divergent_layer": make_json_ready(first),
    }


def fp_focus_tensors(fp_probe: dict[str, Any]) -> dict[str, np.ndarray]:
    conv2_tensor_key = "stage4_1_conv2"
    if conv2_tensor_key not in fp_probe["tensors"] and "stage4_1_add_input0" in fp_probe["tensors"]:
        conv2_tensor_key = "stage4_1_add_input0"
    add_pre = np.asarray(
        fp_probe["tensors"].get("stage4_1_add", fp_probe["tensors"][conv2_tensor_key]),
        dtype=np.float64,
    )
    add_post = np.asarray(
        fp_probe["tensors"].get(
            "stage4_1_out_relu",
            fp_probe["tensors"].get("stage4_1_add", fp_probe["tensors"][conv2_tensor_key]),
        ),
        dtype=np.float64,
    )
    return {
        "stage4_1_conv1": np.asarray(fp_probe["tensors"]["stage4_1_conv1"], dtype=np.float64),
        "stage4_1_conv2": np.asarray(fp_probe["tensors"][conv2_tensor_key], dtype=np.float64),
        "stage4_1_add_pre_requant": add_pre,
        "stage4_1_add_post_requant": add_post,
        "global_pool_post_requant": np.asarray(fp_probe["tensors"]["global_pool_post_requant"], dtype=np.float64),
        "head_input": np.asarray(fp_probe["tensors"]["head_input"], dtype=np.float64),
        "model_output": np.asarray(fp_probe["tensors"]["model_output"], dtype=np.float64).reshape(-1),
    }


def build_export_boundary_tap_records(
    *,
    boundary_key: str,
    left_tensors: dict[str, np.ndarray],
    right_tensors: dict[str, np.ndarray],
    local_threshold: float,
    semantic_tolerances: dict[str, float] | None = None,
    operator_name_by_alias: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    semantic_tolerances = semantic_tolerances or {}
    operator_name_by_alias = operator_name_by_alias or {}
    for alias in LOCAL_TAP_PRIORITY:
        left = np.asarray(left_tensors[alias], dtype=np.float64)
        right = np.asarray(right_tensors[alias], dtype=np.float64)
        drift = compare_arrays_rich(left, right)
        tolerance = semantic_tolerances.get(alias)
        if tolerance is None:
            status = "warn" if float(drift.get("mean_abs_diff") or 0.0) >= local_threshold else "ok"
        else:
            status = "warn" if float(drift.get("max_abs_diff") or 0.0) > float(tolerance) else "ok"
        records.append(
            {
                "boundary_key": boundary_key,
                "alias": alias,
                "label": TAP_LABELS[alias],
                "operator_name": operator_name_by_alias.get(alias, ACTIONABLE_OPERATOR_BY_ALIAS[alias]),
                "status": status,
                "semantic_tolerance": tolerance,
                "left_stats": tensor_stats(left),
                "right_stats": tensor_stats(right),
                "drift": drift,
            }
        )
    return records


def first_material_export_tap(tap_records: list[dict[str, Any]]) -> dict[str, Any]:
    for tap in tap_records:
        if tap.get("status") == "warn":
            return tap
    return max(
        tap_records,
        key=lambda tap: float(((tap.get("drift") or {}).get("mean_abs_diff")) or -1.0),
    )


def build_focus_boundary_reports(
    *,
    models: dict[str, Any],
    onnx_probe_ctx: dict[str, Any],
    sample: dict[str, Any],
    local_threshold: float,
) -> dict[str, Any]:
    fp_probe = run_hybrid_follow_pytorch_probe(models["model_fp"], sample["float"])
    fq_probe = run_hybrid_follow_pytorch_probe(models["model_fq"], sample["float"])
    id_probe = run_hybrid_follow_pytorch_probe(models["model_id"], sample["staged"])
    id_add_audit = run_hybrid_follow_integer_add_audit(models["model_id"], sample["staged"])
    onnx_probe = run_onnx_probe(onnx_probe_ctx, sample["staged"])

    semantic_tensors = semanticize_focus_taps(
        fq_probe,
        id_probe,
        onnx_probe,
        id_add_audit,
        models["context_map"],
        models["head_eps_in"],
    )
    fp_tensors = fp_focus_tensors(fp_probe)
    operator_name_by_alias = {
        alias: ((onnx_probe.get("selected") or {}).get(alias) or {}).get("operator_name")
        or ACTIONABLE_OPERATOR_BY_ALIAS[alias]
        for alias in LOCAL_TAP_PRIORITY
    }
    tolerances = {
        alias: semantic_tolerance_for_alias(alias, id_add_audit, models["context_map"], models["head_eps_in"])
        for alias in LOCAL_TAP_PRIORITY
    }

    fp_to_fq_taps = build_export_boundary_tap_records(
        boundary_key="fp_to_fq",
        left_tensors=fp_tensors,
        right_tensors=semantic_tensors["fq"],
        local_threshold=local_threshold,
        operator_name_by_alias=operator_name_by_alias,
    )
    fq_to_id_taps = build_export_boundary_tap_records(
        boundary_key="fq_to_id",
        left_tensors=semantic_tensors["fq"],
        right_tensors=semantic_tensors["id"],
        local_threshold=local_threshold,
        operator_name_by_alias=operator_name_by_alias,
    )
    id_to_onnx_taps = build_export_boundary_tap_records(
        boundary_key="id_to_onnx",
        left_tensors=semantic_tensors["id"],
        right_tensors=semantic_tensors["onnx"],
        local_threshold=local_threshold,
        semantic_tolerances=tolerances,
        operator_name_by_alias=operator_name_by_alias,
    )

    return {
        "focus_sample": {
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
        },
        "id_add_scale_selection": make_json_ready(
            (id_add_audit.get("reports", {}).get("stage4.1.add") or {}).get("scale_selection")
        ),
        "operator_name_by_alias": operator_name_by_alias,
        "boundaries": {
            "fp_to_fq": {
                "taps": fp_to_fq_taps,
                "first_bad_tap": first_material_export_tap(fp_to_fq_taps),
            },
            "fq_to_id": {
                "taps": fq_to_id_taps,
                "first_bad_tap": first_material_export_tap(fq_to_id_taps),
            },
            "id_to_onnx": {
                "taps": id_to_onnx_taps,
                "first_bad_tap": first_material_export_tap(id_to_onnx_taps),
            },
        },
    }


def first_bad_boundary_for_row(row: dict[str, Any]) -> dict[str, Any]:
    pairwise = row.get("pairwise") or {}
    for key, label, side in BOUNDARY_SPECS:
        report = pairwise.get(key)
        if isinstance(report, dict) and report.get("status") == "warn":
            return {
                "boundary_key": key,
                "boundary_label": label,
                "side": side,
                "score": pair_score(report),
            }
    return {
        "boundary_key": None,
        "boundary_label": None,
        "side": None,
        "score": None,
    }


def summarize_boundary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    first_bad = None
    for key, label, side in BOUNDARY_SPECS:
        warn_rows = [row for row in rows if ((row.get("pairwise") or {}).get(key) or {}).get("status") == "warn"]
        severity = [
            float(pair_score((row.get("pairwise") or {}).get(key)) or 0.0)
            for row in warn_rows
        ]
        payload = {
            "boundary_key": key,
            "boundary_label": label,
            "side": side,
            "warn_count": len(warn_rows),
            "mean_score": float(np.mean(severity)) if severity else None,
            "max_score": float(np.max(severity)) if severity else None,
        }
        summary[key] = payload
        if first_bad is None and warn_rows:
            first_bad = payload
    return {
        "counts": summary,
        "first_bad_boundary": first_bad
        if first_bad is not None
        else {
            "boundary_key": None,
            "boundary_label": None,
            "side": None,
            "warn_count": 0,
            "mean_score": None,
            "max_score": None,
        },
    }


def evaluate_image_fixed_pipeline(
    *,
    models: dict[str, Any],
    onnx_path: Path,
    onnx_session,
    sample: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    with torch.no_grad():
        fp_output = models["model_fp"](sample["float"]).detach().cpu().numpy()
        fq_output = models["model_fq"](sample["float"]).detach().cpu().numpy()
        id_output = models["model_id"](sample["staged"]).detach().cpu().numpy()
    onnx_output = run_onnx_output(onnx_session, sample["staged"])

    stage_results = {
        "fp": make_stage_result(key="fp", label="PyTorch FP", stage_tag="fp", representation="float", raw_native=fp_output),
        "fq": make_stage_result(key="fq", label="NEMO FQ", stage_tag="fq", representation="float", raw_native=fq_output),
        "id": make_stage_result(key="id", label="NEMO ID", stage_tag="id", representation="fixed-point-int32", raw_native=id_output),
        "onnx": make_stage_result(
            key="onnx",
            label=f"Exported ONNX ({onnx_path.name})",
            stage_tag="id",
            representation="fixed-point-int32",
            raw_native=onnx_output,
        ),
        "golden": make_stage_result(
            key="golden",
            label="Golden artifact",
            stage_tag="id",
            representation="fixed-point-int32",
            raw_native=sample["golden_values"],
        ),
        "gvsoc": make_stage_result(
            key="gvsoc",
            label="GVSOC final tensor",
            stage_tag="id",
            representation="fixed-point-int32",
            raw_native=sample["gvsoc_values"],
        ),
    }

    pairwise = {
        "fp_to_fq": compare_stage_pair(stage_results["fp"], stage_results["fq"], thresholds),
        "fq_to_id": compare_stage_pair(stage_results["fq"], stage_results["id"], thresholds),
        "id_to_onnx": compare_stage_pair(stage_results["id"], stage_results["onnx"], thresholds),
        "onnx_to_golden": compare_stage_pair(stage_results["onnx"], stage_results["golden"], thresholds),
        "golden_to_gvsoc": compare_stage_pair(stage_results["golden"], stage_results["gvsoc"], thresholds),
        "fp_to_onnx": compare_stage_pair(stage_results["fp"], stage_results["onnx"], thresholds),
        "fp_to_gvsoc": compare_stage_pair(stage_results["fp"], stage_results["gvsoc"], thresholds),
    }
    first_bad = first_bad_boundary_for_row({"pairwise": pairwise})
    return {
        "image_name": sample["image_name"],
        "image_path": sample["image_path"],
        "sample_dir": sample["sample_dir"],
        "stage_outputs": {name: stage_result_dict(stage_result) for name, stage_result in stage_results.items()},
        "pairwise": pairwise,
        "first_bad_boundary": first_bad,
        "scores": {
            "fp_to_onnx": pair_score(pairwise["fp_to_onnx"]),
            "fp_to_gvsoc": pair_score(pairwise["fp_to_gvsoc"]),
            "fq_to_id": pair_score(pairwise["fq_to_id"]),
            "id_to_onnx": pair_score(pairwise["id_to_onnx"]),
            "onnx_to_golden": pair_score(pairwise["onnx_to_golden"]),
            "golden_to_gvsoc": pair_score(pairwise["golden_to_gvsoc"]),
        },
        "runtime_layer_compare": compact_runtime_layer_compare(sample.get("runtime_layer_compare")),
        "fixed_artifacts": {
            "golden_path": sample["golden_path"],
            "gvsoc_json_path": sample["gvsoc_json_path"],
            "gvsoc_log_path": sample["gvsoc_log_path"],
            "runtime_layer_compare_path": sample.get("runtime_layer_compare_path"),
        },
    }


def onnx_anti_collapse_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fp_x = [float(row["stage_outputs"]["fp"]["decoded"]["x_offset"]) for row in rows]
    onnx_x = [float(row["stage_outputs"]["onnx"]["decoded"]["x_offset"]) for row in rows]
    gvsoc_x = [float(row["stage_outputs"]["gvsoc"]["decoded"]["x_offset"]) for row in rows]
    return {
        "onnx": summarize_anti_collapse(fp_x, onnx_x),
        "gvsoc": summarize_anti_collapse(fp_x, gvsoc_x),
    }


def build_dataset_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_scores = [float(primary_score_from_row(row)) for row in rows if primary_score_from_row(row) is not None]
    fp_to_onnx_scores = [float(row["scores"]["fp_to_onnx"]) for row in rows if row["scores"]["fp_to_onnx"] is not None]
    id_to_onnx_warn_count = sum(1 for row in rows if row["pairwise"]["id_to_onnx"]["status"] == "warn")
    onnx_to_golden_warn_count = sum(1 for row in rows if row["pairwise"]["onnx_to_golden"]["status"] == "warn")
    golden_to_gvsoc_warn_count = sum(1 for row in rows if row["pairwise"]["golden_to_gvsoc"]["status"] == "warn")
    runtime_mismatch_count = sum(
        1
        for row in rows
        if ((row.get("runtime_layer_compare") or {}).get("first_divergent_layer") is not None)
    )
    return {
        "count": len(rows),
        "primary_score_mean": float(np.mean(primary_scores)) if primary_scores else None,
        "primary_score_median": float(median(primary_scores)) if primary_scores else None,
        "primary_score_max": float(np.max(primary_scores)) if primary_scores else None,
        "fp_to_onnx_mean": float(np.mean(fp_to_onnx_scores)) if fp_to_onnx_scores else None,
        "id_to_onnx_warn_count": id_to_onnx_warn_count,
        "onnx_to_golden_warn_count": onnx_to_golden_warn_count,
        "golden_to_gvsoc_warn_count": golden_to_gvsoc_warn_count,
        "runtime_mismatch_count": runtime_mismatch_count,
    }


def load_hard_case_names(
    path: Path | None,
    *,
    baseline_rows: list[dict[str, Any]],
    count: int,
) -> list[str]:
    if path is None:
        ranked = [
            row for row in baseline_rows if primary_score_from_row(row) is not None
        ]
        ranked.sort(key=lambda row: float(primary_score_from_row(row) or 0.0), reverse=True)
        return [row["image_name"] for row in ranked[: max(int(count), 0)]]

    payload = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(payload)
        if isinstance(raw, dict):
            raw = raw.get("images") or raw.get("hard_case_names") or []
        return [str(value) for value in raw]
    return [line.strip() for line in payload.splitlines() if line.strip()]


def select_focus_export_sample(
    *,
    boundary_key: str,
    known_bad_summary: dict[str, Any],
    rep16_rows: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    known_bad_first = known_bad_summary.get("first_bad_boundary") or {}
    if known_bad_first.get("boundary_key") == boundary_key:
        return sample_lookup[known_bad_summary["image_name"]]

    boundary_rows = [
        row for row in rep16_rows if (row.get("first_bad_boundary") or {}).get("boundary_key") == boundary_key
    ]
    if not boundary_rows:
        boundary_rows = [
            row for row in rep16_rows if ((row.get("pairwise") or {}).get(boundary_key) or {}).get("status") == "warn"
        ]
    if not boundary_rows:
        return sample_lookup[known_bad_summary["image_name"]]
    boundary_rows.sort(
        key=lambda row: float(((row.get("scores") or {}).get(boundary_key)) or (primary_score_from_row(row) or 0.0)),
        reverse=True,
    )
    return sample_lookup[boundary_rows[0]["image_name"]]


def select_first_runtime_mismatch(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        first = ((row.get("runtime_layer_compare") or {}).get("first_divergent_layer") or None)
        if first is None:
            continue
        candidates.append(
            {
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "boundary_key": "golden_to_gvsoc",
                "boundary_label": "golden -> GVSOC",
                "side": "runtime-side",
                "operator_name": first.get("layer_name"),
                "operator_label": first.get("layer_name"),
                "layer_index": first.get("index"),
                "local_metrics": make_json_ready(first),
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("layer_index") if item.get("layer_index") is not None else 1_000_000),
            -float((item.get("local_metrics") or {}).get("mean_abs_diff") or 0.0),
        )
    )
    return candidates[0]


def summarize_known_bad_outputs(
    *,
    models: dict[str, Any],
    onnx_session,
    sample: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    with torch.no_grad():
        fp_output = models["model_fp"](sample["float"]).detach().cpu().numpy()
        fq_output = models["model_fq"](sample["float"]).detach().cpu().numpy()
        id_output = models["model_id"](sample["staged"]).detach().cpu().numpy()
    onnx_output = run_onnx_output(onnx_session, sample["staged"])

    stage_results = {
        "fp": make_stage_result(key="fp", label="PyTorch FP", stage_tag="fp", representation="float", raw_native=fp_output),
        "fq": make_stage_result(key="fq", label="NEMO FQ", stage_tag="fq", representation="float", raw_native=fq_output),
        "id": make_stage_result(key="id", label="NEMO ID", stage_tag="id", representation="fixed-point-int32", raw_native=id_output),
        "onnx": make_stage_result(key="onnx", label="Exported ONNX", stage_tag="id", representation="fixed-point-int32", raw_native=onnx_output),
    }
    pairwise = {
        "fp_to_fq": compare_stage_pair(stage_results["fp"], stage_results["fq"], thresholds),
        "fq_to_id": compare_stage_pair(stage_results["fq"], stage_results["id"], thresholds),
        "id_to_onnx": compare_stage_pair(stage_results["id"], stage_results["onnx"], thresholds),
    }
    first_bad = first_bad_boundary_for_row({"pairwise": pairwise})
    return {
        "image_name": sample["image_name"],
        "image_path": sample["image_path"],
        "stage_outputs": {name: stage_result_dict(stage) for name, stage in stage_results.items()},
        "pairwise": pairwise,
        "first_bad_boundary": first_bad,
    }


def operator_key_from_focus_tap(boundary_key: str, focus_boundary_report: dict[str, Any] | None) -> dict[str, Any]:
    if boundary_key not in EXPORT_BOUNDARY_KEYS or not isinstance(focus_boundary_report, dict):
        return {
            "boundary_key": boundary_key,
            "boundary_label": boundary_label(boundary_key),
            "side": boundary_side(boundary_key),
            "operator_name": None,
            "operator_label": None,
            "actionable_operator_name": None,
            "actionable_reason": None,
            "alias": None,
            "local_metrics": None,
        }

    first_bad = (focus_boundary_report.get("boundaries") or {}).get(boundary_key, {}).get("first_bad_tap")
    if first_bad is None:
        return {
            "boundary_key": boundary_key,
            "boundary_label": boundary_label(boundary_key),
            "side": boundary_side(boundary_key),
            "operator_name": None,
            "operator_label": None,
            "actionable_operator_name": None,
            "actionable_reason": None,
            "alias": None,
            "local_metrics": None,
        }

    alias = first_bad.get("alias")
    actionable = first_bad.get("operator_name") or ACTIONABLE_OPERATOR_BY_ALIAS.get(alias)
    actionable_reason = None
    if (
        actionable is not None
        and first_bad.get("operator_name") is not None
        and actionable != first_bad.get("operator_name")
    ):
        actionable_reason = (
            f"{first_bad.get('label')} is downstream, so the loop patches the immediate "
            f"micro-region at {actionable} instead of widening to a whole-stage change."
        )
    return {
        "boundary_key": boundary_key,
        "boundary_label": boundary_label(boundary_key),
        "side": boundary_side(boundary_key),
        "operator_name": first_bad.get("operator_name"),
        "operator_label": first_bad.get("label"),
        "actionable_operator_name": actionable,
        "actionable_reason": actionable_reason,
        "alias": alias,
        "local_metrics": make_json_ready(first_bad),
    }


def anti_collapse_regressed(baseline: dict[str, Any], trial: dict[str, Any]) -> bool:
    if not baseline:
        return False
    if trial.get("sign_flip_rate") is not None and baseline.get("sign_flip_rate") is not None:
        if float(trial["sign_flip_rate"]) > float(baseline["sign_flip_rate"]) + 1e-6:
            return True
    if trial.get("collapsed_fraction") is not None and baseline.get("collapsed_fraction") is not None:
        if float(trial["collapsed_fraction"]) > float(baseline["collapsed_fraction"]) + 1e-6:
            return True
    if trial.get("correlation") is not None and baseline.get("correlation") is not None:
        if float(trial["correlation"]) + 1e-6 < float(baseline["correlation"]):
            return True
    if trial.get("left_right_ordering_agreement") is not None and baseline.get("left_right_ordering_agreement") is not None:
        if float(trial["left_right_ordering_agreement"]) + 1e-6 < float(baseline["left_right_ordering_agreement"]):
            return True
    if trial.get("slope") is not None and baseline.get("slope") is not None:
        if abs(float(trial["slope"]) - 1.0) > abs(float(baseline["slope"]) - 1.0) + 1e-6:
            return True
    return False


def looks_like_zero_collapse(baseline: dict[str, Any], trial: dict[str, Any], *, score_improved: bool) -> bool:
    if not score_improved:
        return False
    if trial.get("collapsed_fraction") is not None and baseline.get("collapsed_fraction") is not None:
        if float(trial["collapsed_fraction"]) > float(baseline["collapsed_fraction"]) + 0.10:
            return True
    if trial.get("correlation") is not None and baseline.get("correlation") is not None:
        if float(trial["correlation"]) + 0.05 < float(baseline["correlation"]):
            return True
    if trial.get("slope") is not None and baseline.get("slope") is not None:
        if abs(float(trial["slope"])) + 0.10 < abs(float(baseline["slope"])):
            return True
    if trial.get("left_right_ordering_agreement") is not None and baseline.get("left_right_ordering_agreement") is not None:
        if float(trial["left_right_ordering_agreement"]) + 0.05 < float(baseline["left_right_ordering_agreement"]):
            return True
    return False


def compare_earliest_progress(baseline: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    base_bad = baseline.get("earliest_bad") or {}
    trial_bad = trial.get("earliest_bad") or {}
    base_boundary = base_bad.get("boundary_key")
    trial_boundary = trial_bad.get("boundary_key")
    base_boundary_pos = boundary_position(base_boundary)
    trial_boundary_pos = boundary_position(trial_boundary)
    boundary_moved_later = trial_boundary_pos > base_boundary_pos
    same_boundary = base_boundary == trial_boundary
    op_moved_later = False
    if same_boundary:
        base_alias = base_bad.get("alias")
        trial_alias = trial_bad.get("alias")
        op_moved_later = alias_position(trial_alias) > alias_position(base_alias)
    return {
        "baseline_boundary_key": base_boundary,
        "trial_boundary_key": trial_boundary,
        "boundary_moved_later": boundary_moved_later,
        "same_boundary": same_boundary,
        "operator_moved_later": op_moved_later,
        "moved_later_or_disappeared": boundary_moved_later or (same_boundary and op_moved_later),
    }


def build_candidate_specs_for_target(
    *,
    actionable_operator_name: str | None,
    target_alias: str | None,
    reference_models: dict[str, Any],
    calib_samples: list[dict[str, Any]],
    activation_audit: dict[str, Any] | None = None,
) -> tuple[list[PolicySpec], dict[str, Any]]:
    baseline_spec = PolicySpec(
        name="current",
        description="Current exporter/runtime-fixed baseline",
        operator_name=str(actionable_operator_name or "none"),
        family="baseline",
    )
    if actionable_operator_name is None:
        return [baseline_spec], {"catalog": {}}

    specs: list[PolicySpec] = [baseline_spec]
    catalog: dict[str, Any] = {}

    if actionable_operator_name == "stage4.1.conv1":
        activation_module = "stage4.0.out_relu"
        audit = activation_audit or {}
        recs = audit.get("recommendations") or {}
        current_alpha = float(audit.get("alpha") or 0.0)
        widened_alpha = float(recs.get("widened_alpha") or current_alpha or 1e-12)
        widened_scale = float(recs.get("widened_alpha_scale") or 1.0)
        higher_precision_bits = int(recs.get("higher_precision_bits") or 10)
        calib_audit = ((audit.get("datasets") or {}).get("real_calibration") or {})
        rep16_audit = ((audit.get("datasets") or {}).get("rep16") or {})
        catalog["stage4_1_conv1_input_activation"] = {
            "note": (
                "Only activation-side stage4.1.conv1 input probes are allowed in this loop: "
                "bypass fake quant diagnostically, widen alpha once, or raise activation precision."
            ),
            "module_name": activation_module,
            "current_alpha": current_alpha,
            "current_precision_bits": audit.get("precision_bits"),
            "unsigned": audit.get("unsigned"),
            "relu_aware": audit.get("relu_aware"),
            "wastes_negative_dynamic_range": audit.get("wastes_dynamic_range_on_negative_values"),
            "real_calibration_clipped_fraction": calib_audit.get("clipped_fraction"),
            "rep16_clipped_fraction": rep16_audit.get("clipped_fraction"),
            "recommended_widened_alpha": widened_alpha,
            "recommended_widened_alpha_scale": widened_scale,
            "recommended_higher_precision_bits": higher_precision_bits,
        }
        specs.append(
            PolicySpec(
                name="stage4_1_input_bypass_fake_quant_diagnostic",
                description=(
                    "Diagnostic only: bypass fake quantization at the stage4.1.conv1 input activation "
                    "while keeping the rest of the model quantized."
                ),
                operator_name="stage4.1.conv1",
                family="activation_diagnostic",
                group_name="stage4_1_input_activation",
                target_label="stage4.1.conv1",
                patched_regions=["conv1"],
                diagnostic_only=True,
                activation_overrides={
                    activation_module: {
                        "policy_name": "bypass_fake_quant",
                        "bypass_fake_quant": True,
                    }
                },
                search_context={
                    "method": "activation_input_diagnostic",
                    "module_name": activation_module,
                    "mode": "bypass_fake_quant_only",
                    "expected_boundary_shift": "fp_to_fq_later_if_activation_quantizer_is_primary_blocker",
                    "activation_audit_summary": make_json_ready(catalog["stage4_1_conv1_input_activation"]),
                },
            )
        )
        specs.append(
            PolicySpec(
                name="stage4_1_input_widened_alpha",
                description=(
                    f"Widen only the stage4.1.conv1 input activation alpha to {widened_alpha:.8f} "
                    "using the fixed real-calibration audit."
                ),
                operator_name="stage4.1.conv1",
                family="activation",
                group_name="stage4_1_input_activation",
                target_label="stage4.1.conv1",
                patched_regions=["conv1"],
                activation_overrides={
                    activation_module: {
                        "alpha": widened_alpha,
                        "policy_name": "activation_audit_widened_alpha",
                        "audit_recommended": True,
                    }
                },
                search_context={
                    "method": "activation_alpha_override",
                    "module_name": activation_module,
                    "current_alpha": current_alpha,
                    "recommended_alpha": widened_alpha,
                    "recommended_alpha_scale": widened_scale,
                    "activation_audit_summary": make_json_ready(catalog["stage4_1_conv1_input_activation"]),
                },
            )
        )
        specs.append(
            PolicySpec(
                name="stage4_1_input_higher_precision",
                description=(
                    f"Raise only the stage4.1.conv1 input activation precision to {higher_precision_bits} bits."
                ),
                operator_name="stage4.1.conv1",
                family="activation_precision",
                group_name="stage4_1_input_activation",
                target_label="stage4.1.conv1",
                patched_regions=["conv1"],
                activation_overrides={
                    activation_module: {
                        "precision_bits": higher_precision_bits,
                        "policy_name": f"activation_precision_{higher_precision_bits}b",
                    }
                },
                search_context={
                    "method": "activation_precision_override",
                    "module_name": activation_module,
                    "baseline_bits": audit.get("precision_bits"),
                    "candidate_bits": higher_precision_bits,
                    "activation_audit_summary": make_json_ready(catalog["stage4_1_conv1_input_activation"]),
                },
            )
        )
        return specs, {"catalog": catalog}

    if actionable_operator_name == "stage4.1.conv2":
        region_key = "conv1" if actionable_operator_name.endswith("conv1") else "conv2"
        region_spec = ACTIVATION_REGION_SPECS[region_key]
        module_name = str(region_spec["activation_module"])
        module = resolve_dotted_module(reference_models["model_fq"], module_name)
        activation_samples = collect_module_output_samples(
            reference_models["model_fq"],
            calib_samples,
            [module_name],
            statistics_act=True,
        )
        policy_reports = activation_policy_reports(activation_samples[module_name], module)
        specs.extend(build_activation_policy_specs(region_key, region_spec, policy_reports))
        catalog[region_key] = make_json_ready(policy_reports)
        return specs, {"catalog": catalog}

    if actionable_operator_name == "stage4.1.add":
        region_key = "add_activation"
        region_spec = ACTIVATION_REGION_SPECS[region_key]
        module_name = str(region_spec["activation_module"])
        module = resolve_dotted_module(reference_models["model_fq"], module_name)
        activation_samples = collect_module_output_samples(
            reference_models["model_fq"],
            calib_samples,
            [module_name],
            statistics_act=True,
        )
        activation_reports = activation_policy_reports(activation_samples[module_name], module)
        activation_specs = build_activation_policy_specs(region_key, region_spec, activation_reports)
        specs.extend(activation_specs)
        catalog["add_activation"] = make_json_ready(activation_reports)

        add_branch_samples = collect_integer_add_branch_samples(
            reference_models["model_fq"],
            calib_samples,
            "stage4.1.add",
        )
        add_eps_in_list = [
            float(value)
            for value in ((reference_models["context_map"].get("stage4.1.add") or {}).get("eps_in_list") or [])
        ]
        add_reports = build_add_scale_policy_reports(add_branch_samples, add_eps_in_list)
        add_scale_specs = build_add_scale_policy_specs(add_reports)
        specs.extend(add_scale_specs)
        catalog["add_scale"] = make_json_ready(add_reports)

        matched_scale_spec = PolicySpec(
            name="matched_scale_branch_alignment",
            description="Use matched-scale residual add alignment around stage4.1.add only.",
            operator_name="stage4.1.add",
            family="architecture_ablation",
            group_name="architecture_ablation",
            target_label="stage4.1.add post-requant",
            patched_regions=["add"],
            integer_add_operator_overrides={"stage4.1.add": "max_branch"},
            search_context={"variant": "matched_scale_branch_alignment"},
        )
        specs.append(matched_scale_spec)

        if target_alias in {"stage4_1_add_post_requant", "global_pool_post_requant", "head_input", "model_output"}:
            for add_scale_spec in add_scale_specs:
                for activation_spec in activation_specs:
                    specs.append(
                        PolicySpec(
                            name=f"microregion_{add_scale_spec.name}__{activation_spec.name}",
                            description=(
                                "Patch stage4.1.add requant plus its immediate consumer using "
                                f"{add_scale_spec.name} and {activation_spec.name}."
                            ),
                            operator_name="stage4.1.add",
                            family="micro_region",
                            group_name="add_post_micro_region",
                            target_label="stage4.1.add post-requant",
                            patched_regions=["add"],
                            activation_overrides=deepcopy(activation_spec.activation_overrides),
                            integer_add_operator_overrides=deepcopy(add_scale_spec.integer_add_operator_overrides),
                            search_context={
                                "add_scale": add_scale_spec.name,
                                "add_activation": activation_spec.name,
                            },
                        )
                    )

        return specs, {"catalog": catalog}

    return specs, {"catalog": catalog}


def evaluate_trial(
    *,
    export_args,
    device: torch.device,
    calib_samples: list[dict[str, Any]],
    spec: PolicySpec,
    known_bad_sample: dict[str, Any],
    rep16_samples: list[dict[str, Any]],
    thresholds: Thresholds,
    local_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    trial_dir = output_dir / "trials" / sanitize_name(spec.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial = {
        "policy": spec.name,
        "description": spec.description,
        "operator_name": spec.operator_name,
        "family": spec.family,
        "diagnostic_only": bool(getattr(spec, "diagnostic_only", False)),
        "patched_regions": spec.patched_regions,
        "search_context": make_json_ready(spec.search_context),
        "trial_dir": str(trial_dir),
    }

    try:
        models = build_models_for_policy(export_args, device, calib_samples, spec)
        onnx_path = export_integer_model_onnx(
            models["model_id"],
            trial_dir / "model_id.onnx",
            input_channels=export_args.input_channels,
            height=export_args.height,
            width=export_args.width,
        )
        onnx_session = build_onnx_context(onnx_path, export_args.height, export_args.width)
        onnx_probe_ctx = build_onnx_probe_context(onnx_session, trial_dir / "probe_cache")

        known_bad_summary = summarize_known_bad_outputs(
            models=models,
            onnx_session=onnx_session["session"],
            sample=known_bad_sample,
            thresholds=thresholds,
        )

        rep16_rows = [
            evaluate_image_fixed_pipeline(
                models=models,
                onnx_path=onnx_path,
                onnx_session=onnx_session["session"],
                sample=sample,
                thresholds=thresholds,
            )
            for sample in rep16_samples
        ]
        boundary_summary = summarize_boundary_counts(rep16_rows)
        earliest_boundary = boundary_summary["first_bad_boundary"]
        sample_lookup = {sample["image_name"]: sample for sample in rep16_samples}
        sample_lookup[known_bad_sample["image_name"]] = known_bad_sample

        earliest_bad = None
        focus_report = None
        if earliest_boundary["boundary_key"] in EXPORT_BOUNDARY_KEYS:
            focus_sample = select_focus_export_sample(
                boundary_key=earliest_boundary["boundary_key"],
                known_bad_summary=known_bad_summary,
                rep16_rows=rep16_rows,
                sample_lookup=sample_lookup,
            )
            focus_report = build_focus_boundary_reports(
                models=models,
                onnx_probe_ctx=onnx_probe_ctx,
                sample=focus_sample,
                local_threshold=local_threshold,
            )
            earliest_bad = operator_key_from_focus_tap(earliest_boundary["boundary_key"], focus_report)
        elif earliest_boundary["boundary_key"] == "onnx_to_golden":
            warn_rows = [
                row for row in rep16_rows if row["pairwise"]["onnx_to_golden"]["status"] == "warn"
            ]
            warn_rows.sort(
                key=lambda row: float(row["scores"]["onnx_to_golden"] or 0.0),
                reverse=True,
            )
            chosen = warn_rows[0] if warn_rows else None
            earliest_bad = {
                "boundary_key": "onnx_to_golden",
                "boundary_label": boundary_label("onnx_to_golden"),
                "side": boundary_side("onnx_to_golden"),
                "operator_name": "deployment output",
                "operator_label": "final deployment output",
                "actionable_operator_name": None,
                "actionable_reason": "The first material mismatch appears only at the final ONNX -> golden boundary.",
                "alias": "model_output",
                "local_metrics": None if chosen is None else make_json_ready(chosen["pairwise"]["onnx_to_golden"]),
                "focus_sample": None if chosen is None else {"image_name": chosen["image_name"], "image_path": chosen["image_path"]},
            }
        elif earliest_boundary["boundary_key"] == "golden_to_gvsoc":
            earliest_bad = select_first_runtime_mismatch(rep16_rows)
        else:
            earliest_bad = {
                "boundary_key": None,
                "boundary_label": None,
                "side": None,
                "operator_name": None,
                "operator_label": None,
                "actionable_operator_name": None,
                "actionable_reason": None,
                "alias": None,
                "local_metrics": None,
            }

        rep16_anti = onnx_anti_collapse_for_rows(rep16_rows)
        trial.update(
            {
                "onnx_path": str(onnx_path),
                "activation_override_report": make_json_ready(models.get("activation_override_report")),
                "conv_bias_report": make_json_ready(models.get("conv_bias_report")),
                "model_patch_report": make_json_ready(models.get("model_patch_report")),
                "known_bad": make_json_ready(known_bad_summary),
                "rep16_rows": make_json_ready(rep16_rows),
                "boundary_summary": make_json_ready(boundary_summary),
                "earliest_bad": make_json_ready(earliest_bad),
                "focus_report": make_json_ready(focus_report),
                "anti_collapse": make_json_ready(rep16_anti),
                "rep16_aggregate": make_json_ready(build_dataset_aggregate(rep16_rows)),
            }
        )
        return trial
    except Exception as exc:
        trial["error"] = f"{type(exc).__name__}: {exc}"
        return trial


def attach_dataset_views(trial: dict[str, Any], hard_case_names: list[str]) -> None:
    rep16_rows = list(trial.get("rep16_rows") or [])
    hard_case_set = set(hard_case_names)
    hard_case_rows = [row for row in rep16_rows if row["image_name"] in hard_case_set]
    datasets = {
        "rep16": {
            "aggregate": build_dataset_aggregate(rep16_rows),
            "boundary_summary": summarize_boundary_counts(rep16_rows),
            "anti_collapse": onnx_anti_collapse_for_rows(rep16_rows),
            "rows": rep16_rows,
        },
        "hard_case": {
            "aggregate": build_dataset_aggregate(hard_case_rows),
            "boundary_summary": summarize_boundary_counts(hard_case_rows),
            "anti_collapse": onnx_anti_collapse_for_rows(hard_case_rows) if hard_case_rows else {"onnx": {}, "gvsoc": {}},
            "rows": hard_case_rows,
        },
    }
    trial["datasets"] = make_json_ready(datasets)
    trial["rep16_aggregate"] = make_json_ready(datasets["rep16"]["aggregate"])


def add_vs_baseline_metrics(trials: dict[str, dict[str, Any]], baseline_name: str) -> None:
    baseline = trials.get(baseline_name) or {}
    baseline_datasets = baseline.get("datasets") or {}
    for trial in trials.values():
        datasets = trial.get("datasets") or {}
        for dataset_name, dataset in datasets.items():
            baseline_rows = {
                row["image_name"]: row for row in ((baseline_datasets.get(dataset_name) or {}).get("rows") or [])
            }
            wins = 0
            losses = 0
            ties = 0
            deltas = []
            for row in dataset.get("rows") or []:
                baseline_row = baseline_rows.get(row["image_name"])
                if baseline_row is None:
                    continue
                base_score = primary_score_from_row(baseline_row)
                trial_score = primary_score_from_row(row)
                if base_score is None or trial_score is None:
                    continue
                delta = float(trial_score) - float(base_score)
                deltas.append(delta)
                if delta < -1e-12:
                    wins += 1
                elif delta > 1e-12:
                    losses += 1
                else:
                    ties += 1
            dataset["vs_baseline"] = {
                "baseline_policy": baseline_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_primary_score_delta": float(np.mean(deltas)) if deltas else None,
                "median_primary_score_delta": float(median(deltas)) if deltas else None,
            }


def target_local_metric_for_trial(
    trial: dict[str, Any],
    *,
    target_boundary_key: str | None,
    target_alias: str | None,
) -> float | None:
    if target_boundary_key not in EXPORT_BOUNDARY_KEYS or target_alias is None:
        return None
    focus_report = trial.get("focus_report") or {}
    boundary = (focus_report.get("boundaries") or {}).get(target_boundary_key) or {}
    for tap in boundary.get("taps") or []:
        if tap.get("alias") == target_alias:
            return float(((tap.get("drift") or {}).get("mean_abs_diff")) or 0.0)
    return None


def ranking_rows(
    trials: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    target_boundary_key: str | None,
    target_alias: str | None,
) -> list[dict[str, Any]]:
    baseline = trials[baseline_name]
    rows = []
    for policy_name, trial in trials.items():
        if trial.get("error"):
            continue
        rep16 = (trial.get("datasets") or {}).get("rep16") or {}
        hard_case = (trial.get("datasets") or {}).get("hard_case") or {}
        rep16_agg = rep16.get("aggregate") or {}
        hard_agg = hard_case.get("aggregate") or {}
        rep16_vs = rep16.get("vs_baseline") or {}
        earliest_progress = compare_earliest_progress(baseline, trial)
        onnx_anti = (rep16.get("anti_collapse") or {}).get("onnx") or {}
        rows.append(
            {
                "policy": policy_name,
                "family": trial.get("family"),
                "description": trial.get("description"),
                "first_bad_boundary": (trial.get("earliest_bad") or {}).get("boundary_label"),
                "first_bad_operator": (trial.get("earliest_bad") or {}).get("operator_label"),
                "target_local_mean_abs_diff": target_local_metric_for_trial(
                    trial,
                    target_boundary_key=target_boundary_key,
                    target_alias=target_alias,
                ),
                "rep16_primary_score_mean": rep16_agg.get("primary_score_mean"),
                "hard_case_primary_score_mean": hard_agg.get("primary_score_mean"),
                "wins": int(rep16_vs.get("wins") or 0),
                "losses": int(rep16_vs.get("losses") or 0),
                "onnx_anti_collapse": make_json_ready(onnx_anti),
                "progress": earliest_progress,
                "id_to_onnx_warn_count": int(rep16_agg.get("id_to_onnx_warn_count") or 0),
                "onnx_to_golden_warn_count": int(rep16_agg.get("onnx_to_golden_warn_count") or 0),
                "runtime_mismatch_count": int(rep16_agg.get("runtime_mismatch_count") or 0),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["progress"]["moved_later_or_disappeared"] else 1,
            boundary_position(
                None
                if (trials[row["policy"]].get("earliest_bad") or {}).get("boundary_key") is None
                else (trials[row["policy"]].get("earliest_bad") or {}).get("boundary_key")
            ) * -1,
            float(row["target_local_mean_abs_diff"] if row["target_local_mean_abs_diff"] is not None else 1e9),
            anti_collapse_sort_key(row.get("onnx_anti_collapse")),
            float(row["rep16_primary_score_mean"] if row["rep16_primary_score_mean"] is not None else 1e9),
            row["id_to_onnx_warn_count"] + row["onnx_to_golden_warn_count"] + row["runtime_mismatch_count"],
        )
    )
    return rows


def choose_patch(
    trials: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    ranking: list[dict[str, Any]],
    target_boundary_key: str | None,
    target_alias: str | None,
) -> dict[str, Any]:
    baseline = trials[baseline_name]
    baseline_rep16 = ((baseline.get("datasets") or {}).get("rep16") or {})
    baseline_hard = ((baseline.get("datasets") or {}).get("hard_case") or {})
    baseline_rep16_agg = baseline_rep16.get("aggregate") or {}
    baseline_hard_agg = baseline_hard.get("aggregate") or {}
    baseline_anti = (baseline_rep16.get("anti_collapse") or {}).get("onnx") or {}
    baseline_target_metric = target_local_metric_for_trial(
        baseline,
        target_boundary_key=target_boundary_key,
        target_alias=target_alias,
    )

    if not ranking:
        return {
            "action": "keep baseline",
            "reason": "No candidate trials completed successfully.",
            "accepted_policy": None,
            "earliest_bad_moved_later": False,
        }

    acceptable_rows = []
    for row in ranking:
        if row["policy"] == baseline_name:
            continue
        trial = trials[row["policy"]]
        if trial.get("diagnostic_only"):
            continue
        rep16 = (trial.get("datasets") or {}).get("rep16") or {}
        hard_case = (trial.get("datasets") or {}).get("hard_case") or {}
        rep16_agg = rep16.get("aggregate") or {}
        hard_agg = hard_case.get("aggregate") or {}
        rep16_vs = rep16.get("vs_baseline") or {}
        hard_vs = hard_case.get("vs_baseline") or {}
        trial_anti = (rep16.get("anti_collapse") or {}).get("onnx") or {}

        score_improved = (
            rep16_agg.get("primary_score_mean") is not None
            and baseline_rep16_agg.get("primary_score_mean") is not None
            and float(rep16_agg["primary_score_mean"]) < float(baseline_rep16_agg["primary_score_mean"]) - 1e-12
        )
        hard_ok = True
        if baseline_hard_agg.get("primary_score_mean") is not None and hard_agg.get("primary_score_mean") is not None:
            hard_ok = float(hard_agg["primary_score_mean"]) <= float(baseline_hard_agg["primary_score_mean"]) + 1e-12

        local_metric = row["target_local_mean_abs_diff"]
        local_improved = (
            baseline_target_metric is None
            or local_metric is None
            or float(local_metric) < float(baseline_target_metric) - 1e-12
        )
        rep16_non_worse = (
            rep16_agg.get("primary_score_mean") is None
            or baseline_rep16_agg.get("primary_score_mean") is None
            or float(rep16_agg["primary_score_mean"]) <= float(baseline_rep16_agg["primary_score_mean"]) + 1e-12
        )
        non_losing = int(rep16_vs.get("wins") or 0) >= int(rep16_vs.get("losses") or 0)
        non_losing_hard = int(hard_vs.get("wins") or 0) >= int(hard_vs.get("losses") or 0)
        anti_ok = not anti_collapse_regressed(baseline_anti, trial_anti)
        collapse_guard = not looks_like_zero_collapse(baseline_anti, trial_anti, score_improved=score_improved)
        warn_ok = (
            int(rep16_agg.get("id_to_onnx_warn_count") or 0) <= int(baseline_rep16_agg.get("id_to_onnx_warn_count") or 0)
            and int(rep16_agg.get("onnx_to_golden_warn_count") or 0) <= int(baseline_rep16_agg.get("onnx_to_golden_warn_count") or 0)
            and int(rep16_agg.get("runtime_mismatch_count") or 0) <= int(baseline_rep16_agg.get("runtime_mismatch_count") or 0)
        )
        if (
            row["progress"]["moved_later_or_disappeared"]
            and local_improved
            and rep16_non_worse
            and non_losing
            and hard_ok
            and non_losing_hard
            and anti_ok
            and collapse_guard
            and warn_ok
        ):
            acceptable_rows.append(row)

    if acceptable_rows:
        winner = acceptable_rows[0]
        return {
            "action": "accept patch",
            "reason": (
                f"{winner['policy']} moved the earliest bad boundary/operator later on the fixed test set, "
                "improved the targeted local drift, held or improved the batch score, kept x anti-collapse "
                "acceptable, and introduced no new deploy/runtime warnings."
            ),
            "accepted_policy": winner["policy"],
            "earliest_bad_moved_later": True,
        }

    diagnostic_progress = None
    for row in ranking:
        if row["policy"] == baseline_name:
            continue
        trial = trials[row["policy"]]
        if trial.get("diagnostic_only") and row["progress"]["moved_later_or_disappeared"]:
            diagnostic_progress = row
            break

    return {
        "action": "keep baseline",
        "reason": (
            "No local candidate met the acceptance bar. The loop rejected score gains that "
            "looked like x-collapse and rejected patches that failed to move the earliest bad op later."
            if diagnostic_progress is None
            else (
                "A diagnostic-only activation bypass moved the earliest bad point later, but no deployable local "
                "patch met the acceptance criteria."
            )
        ),
        "accepted_policy": None,
        "earliest_bad_moved_later": False,
    }


def earliest_bad_report_markdown(
    *,
    baseline: dict[str, Any],
    hard_case_names: list[str],
) -> str:
    earliest = baseline.get("earliest_bad") or {}
    rep16 = (baseline.get("datasets") or {}).get("rep16") or {}
    hard_case = (baseline.get("datasets") or {}).get("hard_case") or {}
    known_bad = baseline.get("known_bad") or {}
    onnx_anti = (rep16.get("anti_collapse") or {}).get("onnx") or {}
    gvsoc_anti = (rep16.get("anti_collapse") or {}).get("gvsoc") or {}
    lines = [
        "# Earliest Bad Op Report",
        "",
        f"- First bad boundary: `{earliest.get('boundary_label') or 'none'}`",
        f"- First bad operator: `{earliest.get('operator_label') or 'none'}`",
        f"- Actionable operator/micro-region: `{earliest.get('actionable_operator_name') or earliest.get('operator_name') or 'none'}`",
        f"- Side: `{earliest.get('side') or 'none'}`",
        f"- Known bad sample: `{known_bad.get('image_name')}`",
        f"- Fixed hard-case subset: `{hard_case_names}`",
    ]
    if earliest.get("actionable_reason"):
        lines.append(f"- Actionable-note: `{earliest['actionable_reason']}`")
    if baseline.get("focus_report"):
        focus_sample = (baseline["focus_report"].get("focus_sample") or {})
        lines.append(f"- Focus sample: `{focus_sample.get('image_name')}`")
    local_metrics = earliest.get("local_metrics") or {}
    if local_metrics:
        drift = local_metrics.get("drift") or local_metrics
        lines.extend(
            [
                "",
                "## Local Metrics",
                "",
                "- mean_abs_diff=`{}` max_abs_diff=`{}` cosine=`{}` abs_mean_ratio=`{}`".format(
                    drift.get("mean_abs_diff"),
                    drift.get("max_abs_diff"),
                    drift.get("cosine_similarity"),
                    drift.get("abs_mean_ratio"),
                ),
            ]
        )

    conv1_decomposition = baseline.get("conv1_decomposition") or {}
    if conv1_decomposition:
        obs = ((conv1_decomposition.get("drift_breakdown") or {}).get("observed") or {})
        contrib = ((conv1_decomposition.get("drift_breakdown") or {}).get("contributions") or {})
        lines.extend(
            [
                "",
                "## stage4.1.conv1 Decomposition",
                "",
                f"- eps_in=`{((conv1_decomposition.get('context') or {}).get('eps_in'))}` eps_out=`{((conv1_decomposition.get('context') or {}).get('eps_out'))}` weight_eps=`{((conv1_decomposition.get('context') or {}).get('weight_eps'))}`",
                "- observed fq_vs_id mean_abs_diff=`{}` abs_mean_ratio=`{}`".format(
                    ((obs.get("fq_vs_id") or {}).get("mean_abs_diff")),
                    ((obs.get("fq_vs_id") or {}).get("abs_mean_ratio")),
                ),
                "- contribution activation/weight/bias/output=`{}` / `{}` / `{}` / `{}`".format(
                    (((contrib.get("activation_quantizer") or {}).get("step_drift") or {}).get("mean_abs_diff")),
                    (((contrib.get("weight_quantization") or {}).get("step_drift") or {}).get("mean_abs_diff")),
                    (((contrib.get("bias_quantization") or {}).get("step_drift") or {}).get("mean_abs_diff")),
                    (((contrib.get("output_requantization") or {}).get("step_drift") or {}).get("mean_abs_diff")),
                ),
            ]
        )

    activation_audit = baseline.get("activation_quantizer_audit") or {}
    if activation_audit:
        calib = ((activation_audit.get("datasets") or {}).get("real_calibration") or {})
        rep16_audit = ((activation_audit.get("datasets") or {}).get("rep16") or {})
        hard_audit = ((activation_audit.get("datasets") or {}).get("hard_case") or {})
        recs = activation_audit.get("recommendations") or {}
        lines.extend(
            [
                "",
                "## stage4.1.conv1 Input Activation Audit",
                "",
                "- module=`{}` alpha=`{}` bits=`{}` unsigned=`{}` relu_aware=`{}` wastes_negative_dynamic_range=`{}`".format(
                    activation_audit.get("module_name"),
                    activation_audit.get("alpha"),
                    activation_audit.get("precision_bits"),
                    activation_audit.get("unsigned"),
                    activation_audit.get("relu_aware"),
                    activation_audit.get("wastes_dynamic_range_on_negative_values"),
                ),
                "- real_calibration clipped_fraction=`{}` above_alpha_fraction=`{}`".format(
                    calib.get("clipped_fraction"),
                    calib.get("above_alpha_fraction"),
                ),
                "- rep16 clipped_fraction=`{}` above_alpha_fraction=`{}`".format(
                    rep16_audit.get("clipped_fraction"),
                    rep16_audit.get("above_alpha_fraction"),
                ),
                "- hard_case clipped_fraction=`{}` above_alpha_fraction=`{}`".format(
                    hard_audit.get("clipped_fraction"),
                    hard_audit.get("above_alpha_fraction"),
                ),
                "- recommended widened_alpha=`{}` widened_alpha_scale=`{}` higher_precision_bits=`{}`".format(
                    recs.get("widened_alpha"),
                    recs.get("widened_alpha_scale"),
                    recs.get("higher_precision_bits"),
                ),
            ]
        )

    eps_audit = baseline.get("eps_audit") or {}
    if eps_audit:
        lines.extend(
            [
                "",
                "## Eps Audit",
                "",
                f"- stage4.1.conv1_input: `{make_json_ready(eps_audit.get('stage4.1.conv1_input'))}`",
                f"- stage4.1.conv1: `{make_json_ready(eps_audit.get('stage4.1.conv1'))}`",
                f"- consistency: `{make_json_ready(eps_audit.get('consistency'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Batch Summary",
            "",
            f"- rep16 first bad boundary: `{((rep16.get('boundary_summary') or {}).get('first_bad_boundary') or {}).get('boundary_label') or 'none'}`",
            f"- rep16 primary score mean: `{(rep16.get('aggregate') or {}).get('primary_score_mean')}`",
            f"- hard-case primary score mean: `{(hard_case.get('aggregate') or {}).get('primary_score_mean')}`",
            "",
            "## Anti-Collapse (x vs fp)",
            "",
            "- onnx: sign_flip_rate=`{}` corr=`{}` slope=`{}` collapsed_fraction=`{}` left_right_ordering_agreement=`{}`".format(
                onnx_anti.get("sign_flip_rate"),
                onnx_anti.get("correlation"),
                onnx_anti.get("slope"),
                onnx_anti.get("collapsed_fraction"),
                onnx_anti.get("left_right_ordering_agreement"),
            ),
            "- gvsoc: sign_flip_rate=`{}` corr=`{}` slope=`{}` collapsed_fraction=`{}` left_right_ordering_agreement=`{}`".format(
                gvsoc_anti.get("sign_flip_rate"),
                gvsoc_anti.get("correlation"),
                gvsoc_anti.get("slope"),
                gvsoc_anti.get("collapsed_fraction"),
                gvsoc_anti.get("left_right_ordering_agreement"),
            ),
        ]
    )

    rep16_rows = rep16.get("rows") or []
    runtime_first = select_first_runtime_mismatch(rep16_rows)
    if runtime_first is not None:
        lines.extend(
            [
                "",
                "## Runtime Mismatch",
                "",
                f"- First deployed mismatch: `{runtime_first.get('operator_label')}` on `{runtime_first.get('image_name')}`",
                f"- Runtime local metrics: `{runtime_first.get('local_metrics')}`",
            ]
        )
    return "\n".join(lines)


def candidates_report_markdown(
    *,
    ranking: list[dict[str, Any]],
    trials: dict[str, dict[str, Any]],
    baseline_name: str,
    candidate_catalog: dict[str, Any] | None = None,
) -> str:
    candidate_catalog = candidate_catalog or {}
    per_channel = (candidate_catalog.get("catalog") or {}).get("conv1_per_channel") or {}
    lines = [
        "# Local Patch Candidates",
        "",
        f"- Baseline policy: `{baseline_name}`",
    ]
    if per_channel:
        lines.append(f"- stage4.1.conv1 per-channel supported safely: `{per_channel.get('supported')}`")
        if per_channel.get("reason_if_unsupported"):
            lines.append(f"- stage4.1.conv1 per-channel note: `{per_channel.get('reason_if_unsupported')}`")
    lines.extend(
        [
            "",
            "| Policy | Family | First bad boundary | First bad operator | Target local diff | rep16 mean | hard-case mean | Wins | Losses |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ranking:
        trial = trials[row["policy"]]
        rep16_vs = (((trial.get("datasets") or {}).get("rep16") or {}).get("vs_baseline") or {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["policy"],
                row.get("family"),
                row.get("first_bad_boundary") or "none",
                row.get("first_bad_operator") or "none",
                "n/a" if row.get("target_local_mean_abs_diff") is None else f"{float(row['target_local_mean_abs_diff']):.6f}",
                "n/a" if row.get("rep16_primary_score_mean") is None else f"{float(row['rep16_primary_score_mean']):.6f}",
                "n/a" if row.get("hard_case_primary_score_mean") is None else f"{float(row['hard_case_primary_score_mean']):.6f}",
                int(rep16_vs.get("wins") or 0),
                int(rep16_vs.get("losses") or 0),
            )
        )

    for row in ranking:
        trial = trials[row["policy"]]
        rep16 = (trial.get("datasets") or {}).get("rep16") or {}
        onnx_anti = (rep16.get("anti_collapse") or {}).get("onnx") or {}
        lines.extend(
            [
                "",
                f"## {row['policy']}",
                "",
                f"- Description: `{trial.get('description')}`",
                f"- Diagnostic only: `{trial.get('diagnostic_only')}`",
                f"- Search context: `{trial.get('search_context')}`",
                f"- Model patch report: `{trial.get('model_patch_report')}`",
                f"- First bad boundary: `{row.get('first_bad_boundary') or 'none'}`",
                f"- First bad operator: `{row.get('first_bad_operator') or 'none'}`",
                f"- Target local diff: `{row.get('target_local_mean_abs_diff')}`",
                f"- rep16 mean: `{row.get('rep16_primary_score_mean')}`",
                f"- hard-case mean: `{row.get('hard_case_primary_score_mean')}`",
                f"- Wins/Losses: `{((rep16.get('vs_baseline') or {}).get('wins'))}/{((rep16.get('vs_baseline') or {}).get('losses'))}`",
                "- ONNX anti-collapse: sign_flip_rate=`{}` corr=`{}` slope=`{}` collapsed_fraction=`{}` left_right_ordering_agreement=`{}`".format(
                    onnx_anti.get("sign_flip_rate"),
                    onnx_anti.get("correlation"),
                    onnx_anti.get("slope"),
                    onnx_anti.get("collapsed_fraction"),
                    onnx_anti.get("left_right_ordering_agreement"),
                ),
            ]
        )
    return "\n".join(lines)


def patch_decision_markdown(
    *,
    baseline: dict[str, Any],
    decision: dict[str, Any],
    chosen_trial: dict[str, Any] | None,
) -> str:
    earliest = baseline.get("earliest_bad") or {}
    lines = [
        "# Patch Decision",
        "",
        f"- Decision: `{decision.get('action')}`",
        f"- Reason: `{decision.get('reason')}`",
        f"- Baseline earliest bad boundary: `{earliest.get('boundary_label') or 'none'}`",
        f"- Baseline earliest bad operator: `{earliest.get('operator_label') or 'none'}`",
        f"- Earliest bad op moved later: `{decision.get('earliest_bad_moved_later')}`",
    ]
    if chosen_trial is not None:
        chosen = chosen_trial.get("earliest_bad") or {}
        lines.extend(
            [
                f"- Accepted policy: `{chosen_trial.get('policy')}`",
                f"- Accepted earliest bad boundary: `{chosen.get('boundary_label') or 'none'}`",
                f"- Accepted earliest bad operator: `{chosen.get('operator_label') or 'none'}`",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_repo_path(args.ckpt)
    calib_dir = resolve_repo_path(args.calib_dir)
    known_bad_image = resolve_repo_path(args.known_bad_image)
    rep16_dir = resolve_repo_path(args.rep16_dir)
    application_summary_path = resolve_repo_path(args.application_summary)
    layer_manifest_path = resolve_repo_path(args.layer_manifest) if args.layer_manifest else None
    hard_case_list_path = resolve_repo_path(args.hard_case_list) if args.hard_case_list else None
    output_dir = resolve_repo_path(args.output_dir)

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if calib_dir is None or not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {args.calib_dir}")
    if known_bad_image is None or not known_bad_image.is_file():
        raise FileNotFoundError(f"Known bad image not found: {args.known_bad_image}")
    if rep16_dir is None or not rep16_dir.is_dir():
        raise FileNotFoundError(f"rep16 directory not found: {args.rep16_dir}")
    if application_summary_path is None or not application_summary_path.is_file():
        raise FileNotFoundError(f"Application summary not found: {args.application_summary}")
    if layer_manifest_path is not None and not layer_manifest_path.is_file():
        raise FileNotFoundError(f"Layer manifest not found: {args.layer_manifest}")
    if output_dir is None:
        raise RuntimeError("Could not resolve output directory.")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    device = torch.device("cpu")
    patch_model_to_graph_compat()
    image_size = (args.height, args.width)
    application_summary = load_application_summary(application_summary_path)
    rep16_samples = build_rep16_samples(
        rep16_dir=rep16_dir,
        application_summary=application_summary,
        image_size=image_size,
        device=device,
        runtime_cache_dir=output_dir / "runtime_compare_cache",
        layer_manifest_path=layer_manifest_path,
        allow_missing_runtime_compare=args.allow_missing_runtime_compare,
    )
    known_bad_sample = load_hybrid_follow_sample(known_bad_image, image_size, device)
    sample_lookup = {sample["image_name"]: sample for sample in rep16_samples}
    sample_lookup[known_bad_sample["image_name"]] = known_bad_sample
    export_args = build_export_args(args, ckpt_path, calib_dir)
    calib_samples = collect_calib_samples(export_args, image_size, device)
    thresholds = Thresholds(
        x_abs_diff=args.warn_x_abs_diff,
        size_abs_diff=args.warn_size_abs_diff,
        vis_conf_abs_diff=args.warn_vis_conf_abs_diff,
    )

    baseline_spec = PolicySpec(
        name="current",
        description="Current exporter/runtime-fixed baseline",
        operator_name="auto",
        family="baseline",
    )
    baseline_trial = evaluate_trial(
        export_args=export_args,
        device=device,
        calib_samples=calib_samples,
        spec=baseline_spec,
        known_bad_sample=known_bad_sample,
        rep16_samples=rep16_samples,
        thresholds=thresholds,
        local_threshold=args.material_local_mean_abs_diff,
        output_dir=output_dir,
    )
    if baseline_trial.get("error"):
        raise RuntimeError(f"Baseline trial failed: {baseline_trial['error']}")

    hard_case_names = load_hard_case_names(
        hard_case_list_path,
        baseline_rows=baseline_trial.get("rep16_rows") or [],
        count=args.hard_case_count,
    )
    attach_dataset_views(baseline_trial, hard_case_names)

    earliest = baseline_trial.get("earliest_bad") or {}
    actionable_operator_name = earliest.get("actionable_operator_name")
    target_alias = earliest.get("alias")
    target_boundary_key = earliest.get("boundary_key")
    candidate_catalog = {"catalog": {}}
    trials: dict[str, dict[str, Any]] = {"current": baseline_trial}
    per_channel_support = None
    if args.report_only:
        candidate_catalog["catalog"]["note"] = (
            "Report-only mode: skipped local patch candidates and focused QAT/PTQ sweeps."
        )
    else:
        reference_models = build_models_for_policy(export_args, device, calib_samples, baseline_spec)
        candidate_specs, candidate_catalog = build_candidate_specs_for_target(
            actionable_operator_name=actionable_operator_name,
            target_alias=target_alias,
            reference_models=reference_models,
            calib_samples=calib_samples,
            activation_audit=None,
        )
        candidate_catalog.setdefault("catalog", {})

        focus_sample_meta = (baseline_trial.get("focus_report") or {}).get("focus_sample") or {}
        if actionable_operator_name == "stage4.1.conv1":
            activation_audit = build_stage41_conv1_input_activation_audit(
                models=reference_models,
                calib_samples=calib_samples,
                known_bad_sample=known_bad_sample,
                rep16_samples=rep16_samples,
                hard_case_names=hard_case_names,
            )
            eps_audit = build_stage41_eps_audit(reference_models)
            baseline_trial["activation_quantizer_audit"] = make_json_ready(activation_audit)
            baseline_trial["eps_audit"] = make_json_ready(eps_audit)
            candidate_specs, candidate_catalog = build_candidate_specs_for_target(
                actionable_operator_name=actionable_operator_name,
                target_alias=target_alias,
                reference_models=reference_models,
                calib_samples=calib_samples,
                activation_audit=activation_audit,
            )
            candidate_catalog.setdefault("catalog", {})
            candidate_catalog["catalog"]["stage4_1_conv1_input_activation_audit"] = make_json_ready(activation_audit)
            candidate_catalog["catalog"]["stage4_1_conv1_eps_audit"] = make_json_ready(eps_audit)
            focus_sample = sample_lookup.get(focus_sample_meta.get("image_name"), known_bad_sample)
            baseline_trial["conv1_decomposition"] = make_json_ready(
                build_stage41_conv1_decomposition(
                    models=reference_models,
                    sample=focus_sample,
                )
            )

        if earliest.get("side") == "exporter-side" and actionable_operator_name == "stage4.1.conv1":
            per_channel_support = stage41_conv1_per_channel_support(reference_models)
            candidate_catalog["catalog"]["conv1_per_channel"] = make_json_ready(per_channel_support)

        if earliest.get("side") == "exporter-side" and actionable_operator_name is not None and len(candidate_specs) > 1:
            for spec in candidate_specs[1:]:
                trial = evaluate_trial(
                    export_args=export_args,
                    device=device,
                    calib_samples=calib_samples,
                    spec=spec,
                    known_bad_sample=known_bad_sample,
                    rep16_samples=rep16_samples,
                    thresholds=thresholds,
                    local_threshold=args.material_local_mean_abs_diff,
                    output_dir=output_dir,
                )
                if not trial.get("error"):
                    attach_dataset_views(trial, hard_case_names)
                trials[spec.name] = trial
        else:
            candidate_catalog["catalog"]["note"] = (
                "No exporter-side local patch loop was run because the earliest bad boundary is "
                "runtime-side or no actionable exporter micro-region was identified."
            )

        if (
            earliest.get("side") == "exporter-side"
            and actionable_operator_name == "stage4.1.conv1"
            and not args.skip_focused_qat
        ):
            qat_trial = run_focused_qat_trial(
                args=args,
                export_args=export_args,
                device=device,
                calib_samples=calib_samples,
                known_bad_sample=known_bad_sample,
                rep16_samples=rep16_samples,
                thresholds=thresholds,
                local_threshold=args.material_local_mean_abs_diff,
                output_dir=output_dir,
            )
            if not qat_trial.get("error"):
                attach_dataset_views(qat_trial, hard_case_names)
            trials[qat_trial["policy"]] = qat_trial
            candidate_catalog["catalog"]["focused_qat"] = {
                "ran": True,
                "policy": qat_trial.get("policy"),
                "checkpoint": qat_trial.get("qat_checkpoint"),
                "train_log_path": qat_trial.get("qat_train_log"),
                "error": qat_trial.get("error"),
            }

            activation_only_qat_trial = run_activation_only_qat_trial(
                args=args,
                export_args=export_args,
                device=device,
                calib_samples=calib_samples,
                known_bad_sample=known_bad_sample,
                rep16_samples=rep16_samples,
                thresholds=thresholds,
                local_threshold=args.material_local_mean_abs_diff,
                output_dir=output_dir,
            )
            if not activation_only_qat_trial.get("error"):
                attach_dataset_views(activation_only_qat_trial, hard_case_names)
            trials[activation_only_qat_trial["policy"]] = activation_only_qat_trial
            candidate_catalog["catalog"]["activation_only_qat"] = {
                "ran": True,
                "policy": activation_only_qat_trial.get("policy"),
                "checkpoint": activation_only_qat_trial.get("qat_checkpoint"),
                "train_log_path": activation_only_qat_trial.get("qat_train_log"),
                "error": activation_only_qat_trial.get("error"),
                "learned_activation_overrides": activation_only_qat_trial.get("learned_activation_overrides"),
            }
        elif actionable_operator_name == "stage4.1.conv1":
            candidate_catalog["catalog"]["focused_qat"] = {
                "ran": False,
                "reason": "--skip-focused-qat was set.",
            }
            candidate_catalog["catalog"]["activation_only_qat"] = {
                "ran": False,
                "reason": "--skip-focused-qat was set.",
            }

    add_vs_baseline_metrics(trials, baseline_name="current")
    ranking = ranking_rows(
        trials,
        baseline_name="current",
        target_boundary_key=target_boundary_key,
        target_alias=target_alias,
    )
    chosen_trial = None
    if args.report_only:
        decision = {
            "action": "report_only",
            "reason": "Strict localization/report run completed without evaluating local patch or QAT sweeps.",
            "accepted_policy": None,
            "earliest_bad_moved_later": False,
        }
    else:
        decision = choose_patch(
            trials,
            baseline_name="current",
            ranking=ranking,
            target_boundary_key=target_boundary_key,
            target_alias=target_alias,
        )
        if decision.get("accepted_policy"):
            chosen_trial = trials.get(decision["accepted_policy"])

    summary_payload = {
        "args": vars(args),
        "baseline_policy": "current",
        "hard_case_names": hard_case_names,
        "candidate_catalog": make_json_ready(candidate_catalog),
        "baseline": make_json_ready(baseline_trial),
        "trials": make_json_ready(trials),
        "ranking": make_json_ready(ranking),
        "decision": make_json_ready(decision),
    }
    write_json(output_dir / "summary.json", summary_payload)
    if baseline_trial.get("conv1_decomposition"):
        write_json(
            output_dir / "stage4_1_conv1_decomposition_report.json",
            baseline_trial["conv1_decomposition"],
        )
        write_markdown(
            output_dir / "stage4_1_conv1_decomposition_report.md",
            conv1_decomposition_markdown(baseline_trial["conv1_decomposition"]),
        )
    if baseline_trial.get("activation_quantizer_audit"):
        write_json(
            output_dir / "stage4_1_conv1_input_activation_audit.json",
            baseline_trial["activation_quantizer_audit"],
        )
        write_markdown(
            output_dir / "stage4_1_conv1_input_activation_audit.md",
            activation_audit_markdown(baseline_trial["activation_quantizer_audit"]),
        )
    if baseline_trial.get("eps_audit"):
        write_json(
            output_dir / "stage4_1_conv1_eps_audit.json",
            baseline_trial["eps_audit"],
        )
        write_markdown(
            output_dir / "stage4_1_conv1_eps_audit.md",
            eps_audit_markdown(baseline_trial["eps_audit"]),
        )
    write_markdown(
        output_dir / "earliest_bad_op_report.md",
        earliest_bad_report_markdown(
            baseline=baseline_trial,
            hard_case_names=hard_case_names,
        ),
    )
    write_markdown(
        output_dir / "local_patch_candidates.md",
        candidates_report_markdown(
            ranking=ranking,
            trials=trials,
            baseline_name="current",
            candidate_catalog=candidate_catalog,
        ),
    )
    write_markdown(
        output_dir / "patch_decision.md",
        patch_decision_markdown(
            baseline=baseline_trial,
            decision=decision,
            chosen_trial=chosen_trial,
        ),
    )


if __name__ == "__main__":
    main()

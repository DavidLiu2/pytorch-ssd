#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import nemo
import numpy as np
import onnx
import onnxruntime as ort
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_hybrid_follow_stages import (  # noqa: E402
    Thresholds,
    compare_stage_pair,
    parse_numeric_artifact,
    preprocess_image_once,
    resolve_repo_path,
    to_stage_result,
)
from export_nemo_quant import (  # noqa: E402
    HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
    apply_hybrid_follow_export_preset_config,
    build_hybrid_follow_onnx_probe_model,
    collect_calib_samples,
    compare_arrays,
    derive_hybrid_follow_export_preset_config,
    integerize_deploy_conv_biases,
    normalize_hybrid_follow_export_preset,
    normalize_integer_requant_tensors,
    patch_integer_add_scale_selection,
    patch_model_to_graph_compat,
    prepare_model_fp,
    repair_hybrid_follow_fused_quant_graph,
    resolve_hybrid_follow_head_input_eps,
    run_activation_calibration,
    run_hybrid_follow_integer_add_audit,
    run_hybrid_follow_pytorch_probe,
    stage4_1_path_quant_context,
    tensor_stats,
)
from hybrid_follow_image_artifacts import _convert_input_for_model, _get_model_input_meta  # noqa: E402

DEFAULT_RUN_VAL_RESULTS_DIR = (
    PROJECT_DIR / "logs" / "hybrid_follow_val" / "application_vs_checkpoint_20260326_exporter_legacy_default"
)
DEFAULT_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_follow_score.pth"
DEFAULT_ONNX = PROJECT_DIR / "export" / "hybrid_follow" / "hybrid_follow_dory.onnx"
DEFAULT_CALIB_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"

TAP_ALIAS_ORDER = [
    "stage4_1_conv1",
    "stage4_1_conv2",
    "stage4_1_add_pre_requant",
    "stage4_1_add_post_requant",
    "global_pool_post_requant",
    "head_input",
    "model_output",
]

TAP_LABELS = {
    "stage4_1_conv1": "stage4.1.conv1",
    "stage4_1_conv2": "stage4.1.conv2",
    "stage4_1_add_pre_requant": "stage4.1.add pre-requant",
    "stage4_1_add_post_requant": "stage4.1.add post-requant",
    "global_pool_post_requant": "global pool",
    "head_input": "flatten/head input",
    "model_output": "final head output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a representative-batch stage-localization pass for hybrid_follow, "
            "using existing run_val application artifacts plus fresh FP/FQ/QD/ID/ONNX evaluation."
        )
    )
    parser.add_argument(
        "--run-val-results-dir",
        default=str(DEFAULT_RUN_VAL_RESULTS_DIR),
        help="Existing run_val results directory that contains summary.json and application_validation/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-val-results-dir>/stage_localization.",
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument(
        "--hybrid-follow-export-preset",
        default="baseline",
        help="Named hybrid_follow export preset to mirror during FQ/QD/ID localization.",
    )
    parser.add_argument("--warn-x-abs-diff", type=float, default=0.05)
    parser.add_argument("--warn-size-abs-diff", type=float, default=0.05)
    parser.add_argument("--warn-vis-conf-abs-diff", type=float, default=0.10)
    parser.add_argument("--focus-per-ranking", type=int, default=3)
    parser.add_argument("--tap-limit", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def pair_score(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict) or report.get("status") == "skipped":
        return None
    decoded = report.get("decoded_abs_diff") or {}
    return float(decoded.get("x_offset", 0.0) + decoded.get("size_proxy", 0.0) + decoded.get("visibility_confidence", 0.0))


def first_material_transition(pairwise: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [
        ("fp_to_fq", "FP -> FQ"),
        ("fq_to_id", "FQ -> ID"),
        ("id_to_onnx", "ID -> ONNX"),
        ("onnx_to_application", "ONNX -> application"),
    ]
    for key, label in ordered:
        report = pairwise.get(key)
        if isinstance(report, dict) and report.get("status") == "warn":
            return {
                "status": "warn",
                "transition_key": key,
                "transition_label": label,
                "report": report,
            }
    return {
        "status": "ok",
        "transition_key": None,
        "transition_label": None,
        "report": None,
    }


def sanitize_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name).strip("._")
    return cleaned or "item"


def build_export_args(args: argparse.Namespace, ckpt_path: Path, calib_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="hybrid_follow",
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        height=args.height,
        width=args.width,
        input_channels=args.input_channels,
        ckpt=str(ckpt_path),
        calib_dir=str(calib_dir),
        calib_tensor=None,
        calib_batches=args.calib_batches,
        calib_seed=args.calib_seed,
        mean=None,
        std=None,
        bits=args.bits,
        eps_in=args.eps_in,
        disable_conv_bn_fusion=False,
        disable_hybrid_follow_head_collapse=False,
        hybrid_follow_export_preset=normalize_hybrid_follow_export_preset(args.hybrid_follow_export_preset),
    )


def build_stage_models(args: argparse.Namespace, ckpt_path: Path, calib_dir: Path) -> dict[str, Any]:
    patch_integer_add_scale_selection()
    patch_model_to_graph_compat()

    device = torch.device("cpu")
    export_args = build_export_args(args, ckpt_path, calib_dir)
    image_size = (args.height, args.width)
    calib_samples = collect_calib_samples(export_args, image_size, device)

    model_fp = prepare_model_fp(export_args, device)

    dummy_input = torch.randn(1, args.input_channels, args.height, args.width, device=device)
    
    def build_quant_base():
        model_quant = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
        model_quant.to(device).eval()
        repair_hybrid_follow_fused_quant_graph(model_quant)
        model_quant.change_precision(bits=args.bits, scale_weights=True, scale_activations=True)
        run_activation_calibration(model_quant, calib_samples)
        return model_quant

    model_fq = build_quant_base()
    preset_config = derive_hybrid_follow_export_preset_config(
        model_fq,
        calib_samples,
        export_args.hybrid_follow_export_preset,
    )
    preset_report = apply_hybrid_follow_export_preset_config(model_fq, preset_config)
    patch_integer_add_scale_selection(
        HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
        preset_report.get("integer_add_operator_overrides"),
    )

    model_qd = build_quant_base()
    apply_hybrid_follow_export_preset_config(model_qd, preset_config)
    try:
        model_qd.reset_alpha_weights()
    except Exception:
        pass
    repair_hybrid_follow_fused_quant_graph(model_qd)
    model_qd.qd_stage(eps_in=args.eps_in)
    repair_hybrid_follow_fused_quant_graph(model_qd)

    model_id = build_quant_base()
    apply_hybrid_follow_export_preset_config(model_id, preset_config)
    try:
        model_id.reset_alpha_weights()
    except Exception:
        pass
    repair_hybrid_follow_fused_quant_graph(model_id)
    model_id.qd_stage(eps_in=args.eps_in)
    repair_hybrid_follow_fused_quant_graph(model_id)
    model_id.id_stage()
    normalize_integer_requant_tensors(model_id)
    integerize_deploy_conv_biases(model_id)
    model_id.eval()

    return {
        "device": device,
        "export_args": export_args,
        "model_fp": model_fp,
        "model_fq": model_fq,
        "model_qd": model_qd,
        "model_id": model_id,
        "id_quant_context": stage4_1_path_quant_context(model_id),
        "head_eps_in": resolve_hybrid_follow_head_input_eps(model_id),
        "preset_report": preset_report,
    }


def run_model_output(model, input_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(input_tensor)
    return np.asarray(output.detach().cpu().numpy())


def make_stage_result(
    *,
    key: str,
    label: str,
    stage_tag: str,
    representation: str,
    raw_native: np.ndarray | list[int] | list[float],
    integer_output_scale: float = 32768.0,
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


def load_run_val_summary(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run_val summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def collect_batch_items(summary_payload: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    results = summary_payload.get("results") or []
    if limit is not None:
        results = results[:limit]
    return results


def read_sample_artifacts(sample_dir: Path) -> dict[str, Any]:
    golden_path = sample_dir / "output.txt"
    gvsoc_path = sample_dir / "gvsoc_final_tensor.json"
    golden = parse_numeric_artifact(golden_path)
    gvsoc_payload = json.loads(gvsoc_path.read_text(encoding="utf-8"))
    gvsoc = [int(value) for value in gvsoc_payload.get("values", [])]
    return {
        "golden_path": golden_path,
        "gvsoc_path": gvsoc_path,
        "golden": golden,
        "gvsoc": gvsoc,
    }


def build_session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def build_onnx_context(onnx_path: Path, height: int, width: int) -> dict[str, Any]:
    model = onnx.load(str(onnx_path))
    input_name, elem_type, input_shape = _get_model_input_meta(model, height, width)
    session = ort.InferenceSession(str(onnx_path), sess_options=build_session_options(), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if len(output_names) != 1:
        raise RuntimeError(f"Expected one ONNX output, found {len(output_names)}: {output_names}")
    return {
        "onnx_path": onnx_path,
        "model": model,
        "session": session,
        "input_name": input_name,
        "elem_type": elem_type,
        "input_shape": input_shape,
        "output_name": output_names[0],
    }


def onnx_feed_from_staged_input(onnx_ctx: dict[str, Any], x_staged: torch.Tensor):
    feed_values = x_staged.detach().cpu().numpy().reshape(-1)
    return _convert_input_for_model(feed_values, onnx_ctx["input_shape"], onnx_ctx["elem_type"])


def run_onnx_output(onnx_ctx: dict[str, Any], x_staged: torch.Tensor) -> np.ndarray:
    x_feed = onnx_feed_from_staged_input(onnx_ctx, x_staged)
    return np.asarray(
        onnx_ctx["session"].run(
            [onnx_ctx["output_name"]],
            {onnx_ctx["input_name"]: x_feed},
        )[0]
    )


def _find_node_by_input(model: onnx.ModelProto, input_name: str):
    for index, node in enumerate(model.graph.node):
        if input_name in node.input:
            return index, node
    return None, None


def _find_last_op_before(model: onnx.ModelProto, *, op_type: str, start: int, end: int):
    for index in range(end, start - 1, -1):
        node = model.graph.node[index]
        if node.op_type == op_type:
            return index, node
    return None, None


def _find_first_op_after(model: onnx.ModelProto, *, op_type: str, start: int, end: int):
    for index in range(start, end + 1):
        node = model.graph.node[index]
        if node.op_type == op_type:
            return index, node
    return None, None


def resolve_hybrid_follow_onnx_taps(model: onnx.ModelProto, final_output_name: str) -> dict[str, dict[str, Any]]:
    output_to_index = {}
    for index, node in enumerate(model.graph.node):
        for output_name in node.output:
            output_to_index[output_name] = index

    conv1_index, conv1_node = _find_node_by_input(model, "stage4.1.conv1.weight")
    conv2_index, conv2_node = _find_node_by_input(model, "stage4.1.conv2.weight")
    head_index, head_node = _find_node_by_input(model, "head.weight")
    if conv1_node is None or conv2_node is None or head_node is None:
        raise RuntimeError("Could not resolve stage4.1/head nodes in the exported ONNX graph.")

    flatten_input = head_node.input[0]
    flatten_index = output_to_index.get(flatten_input)
    if flatten_index is None:
        raise RuntimeError("Could not resolve Flatten node feeding the head.")
    flatten_node = model.graph.node[flatten_index]

    pool_post_input = flatten_node.input[0]
    pool_post_index = output_to_index.get(pool_post_input)
    if pool_post_index is None:
        raise RuntimeError("Could not resolve post-pool node feeding Flatten.")
    pool_post_node = model.graph.node[pool_post_index]

    add_pre_index, add_pre_node = _find_last_op_before(
        model,
        op_type="Add",
        start=conv2_index,
        end=pool_post_index,
    )
    if add_pre_node is None:
        raise RuntimeError("Could not resolve stage4.1.add pre-requant node.")

    add_post_index, add_post_node = _find_first_op_after(
        model,
        op_type="Floor",
        start=add_pre_index,
        end=pool_post_index,
    )
    if add_post_node is None:
        raise RuntimeError("Could not resolve stage4.1.add post-requant node.")

    relu1_index, relu1_node = _find_last_op_before(
        model,
        op_type="Clip",
        start=conv1_index,
        end=conv2_index,
    )
    if relu1_node is None:
        raise RuntimeError("Could not resolve stage4.1.relu1 clip node.")

    conv1_semantic_output = relu1_node.input[0]
    conv1_semantic_index = output_to_index.get(conv1_semantic_output)
    if conv1_semantic_index is None:
        raise RuntimeError("Could not resolve the node feeding stage4.1.relu1.")
    conv1_semantic_node = model.graph.node[conv1_semantic_index]

    return {
        "stage4_1_conv1": {
            "node_name": conv1_semantic_node.name,
            "node_index": conv1_semantic_index,
            "op_type": conv1_semantic_node.op_type,
            "output_name": conv1_semantic_node.output[0],
            "resolution": "producer of the stage4.1.relu1 Clip input (post-conv1 requant, pre-activation)",
        },
        "stage4_1_conv2": {
            "node_name": conv2_node.name,
            "node_index": conv2_index,
            "op_type": conv2_node.op_type,
            "output_name": conv2_node.output[0],
            "resolution": "node input contains stage4.1.conv2.weight",
        },
        "stage4_1_add_pre_requant": {
            "node_name": add_pre_node.name,
            "node_index": add_pre_index,
            "op_type": add_pre_node.op_type,
            "output_name": add_pre_node.output[0],
            "resolution": "last Add between stage4.1.conv2 and flatten input",
        },
        "stage4_1_add_post_requant": {
            "node_name": add_post_node.name,
            "node_index": add_post_index,
            "op_type": add_post_node.op_type,
            "output_name": add_post_node.output[0],
            "resolution": "first Floor after stage4.1.add and before flatten input",
        },
        "global_pool_post_requant": {
            "node_name": pool_post_node.name,
            "node_index": pool_post_index,
            "op_type": pool_post_node.op_type,
            "output_name": pool_post_node.output[0],
            "resolution": "producer of Flatten input",
        },
        "head_input": {
            "node_name": flatten_node.name,
            "node_index": flatten_index,
            "op_type": flatten_node.op_type,
            "output_name": flatten_node.output[0],
            "resolution": "node feeding head.weight",
        },
        "model_output": {
            "node_name": head_node.name,
            "node_index": head_index,
            "op_type": head_node.op_type,
            "output_name": final_output_name,
            "resolution": "graph output fed by head.weight",
        },
    }


def build_onnx_probe_context(onnx_ctx: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_hybrid_follow_onnx_taps(onnx_ctx["model"], onnx_ctx["output_name"])
    probe_path = output_dir / f"{onnx_ctx['onnx_path'].stem}_stage_localization_probe.onnx"
    requested_outputs = [
        selected[alias]["output_name"]
        for alias in TAP_ALIAS_ORDER
        if alias != "model_output"
    ]
    build_hybrid_follow_onnx_probe_model(onnx_ctx["onnx_path"], probe_path, requested_outputs)
    session = ort.InferenceSession(str(probe_path), sess_options=build_session_options(), providers=["CPUExecutionProvider"])
    return {
        "probe_path": probe_path,
        "session": session,
        "selected": selected,
        "input_name": onnx_ctx["input_name"],
        "elem_type": onnx_ctx["elem_type"],
        "input_shape": onnx_ctx["input_shape"],
    }


def run_onnx_probe(probe_ctx: dict[str, Any], x_staged: torch.Tensor) -> dict[str, Any]:
    x_feed = _convert_input_for_model(
        x_staged.detach().cpu().numpy().reshape(-1),
        probe_ctx["input_shape"],
        probe_ctx["elem_type"],
    )
    output_names = [output.name for output in probe_ctx["session"].get_outputs()]
    output_values = probe_ctx["session"].run(output_names, {probe_ctx["input_name"]: x_feed})
    output_map = dict(zip(output_names, output_values))

    captures = {}
    for alias, info in probe_ctx["selected"].items():
        captures[alias] = np.asarray(output_map[info["output_name"]])
    return {
        "tensors": captures,
        "stats": {alias: tensor_stats(value) for alias, value in captures.items()},
        "selected": probe_ctx["selected"],
        "probe_path": str(probe_ctx["probe_path"]),
    }


def stage_result_dict(stage_result) -> dict[str, Any]:
    return stage_result.to_dict()


def semantic_tensor_for_id_probe(
    alias: str,
    id_probe: dict[str, Any],
    id_add_audit: dict[str, Any],
    id_quant_context: dict[str, Any],
    head_eps_in: float | None,
) -> np.ndarray:
    if alias == "stage4_1_conv1":
        eps = float(id_quant_context["stage4.1.conv1"]["eps_out"])
        return np.asarray(id_probe["tensors"]["stage4_1_conv1"], dtype=np.float64) * eps
    if alias == "stage4_1_conv2":
        eps = float(id_quant_context["stage4.1.conv2"]["eps_out"])
        return np.asarray(id_probe["tensors"]["stage4_1_conv2"], dtype=np.float64) * eps
    if alias == "stage4_1_add_pre_requant":
        return np.asarray(id_add_audit["tensors"]["stage4.1.add"]["pre_requant_semantic"], dtype=np.float64)
    if alias == "stage4_1_add_post_requant":
        return np.asarray(id_add_audit["tensors"]["stage4.1.add"]["post_requant_semantic"], dtype=np.float64)
    if alias == "global_pool_post_requant":
        if head_eps_in is None:
            raise RuntimeError("head_eps_in is unavailable for ID global-pool decoding.")
        return np.asarray(id_probe["tensors"]["global_pool_post_requant"], dtype=np.float64) * float(head_eps_in)
    if alias == "head_input":
        if head_eps_in is None:
            raise RuntimeError("head_eps_in is unavailable for ID head-input decoding.")
        return np.asarray(id_probe["tensors"]["head_input"], dtype=np.float64) * float(head_eps_in)
    if alias == "model_output":
        return np.asarray(id_probe["tensors"]["model_output"], dtype=np.float64).reshape(-1) / 32768.0
    raise KeyError(f"Unsupported ID tap alias: {alias}")


def semantic_tensor_for_onnx_probe(
    alias: str,
    onnx_probe: dict[str, Any],
    id_add_audit: dict[str, Any],
    id_quant_context: dict[str, Any],
    head_eps_in: float | None,
) -> np.ndarray:
    raw = np.asarray(onnx_probe["tensors"][alias], dtype=np.float64)
    if alias == "stage4_1_conv1":
        eps = float(id_quant_context["stage4.1.conv1"]["eps_out"])
        return raw * eps
    if alias == "stage4_1_conv2":
        eps = float(id_quant_context["stage4.1.conv2"]["eps_out"])
        return raw * eps
    if alias == "stage4_1_add_pre_requant":
        scale = id_add_audit["reports"]["stage4.1.add"]["scale_selection"]
        return raw * float(scale["eps_out"]) / float(scale["D"])
    if alias == "stage4_1_add_post_requant":
        scale = id_add_audit["reports"]["stage4.1.add"]["scale_selection"]
        return raw * float(scale["eps_out"])
    if alias == "global_pool_post_requant":
        if head_eps_in is None:
            raise RuntimeError("head_eps_in is unavailable for ONNX global-pool decoding.")
        return raw * float(head_eps_in)
    if alias == "head_input":
        if head_eps_in is None:
            raise RuntimeError("head_eps_in is unavailable for ONNX head-input decoding.")
        return raw * float(head_eps_in)
    if alias == "model_output":
        return raw.reshape(-1) / 32768.0
    raise KeyError(f"Unsupported ONNX tap alias: {alias}")


def semantic_tolerance_for_alias(
    alias: str,
    id_add_audit: dict[str, Any],
    id_quant_context: dict[str, Any],
    head_eps_in: float | None,
) -> float:
    if alias == "stage4_1_conv1":
        return max(float(id_quant_context["stage4.1.conv1"]["eps_out"]) * 0.5, 1e-6)
    if alias == "stage4_1_conv2":
        return max(float(id_quant_context["stage4.1.conv2"]["eps_out"]) * 0.5, 1e-6)
    if alias == "stage4_1_add_pre_requant":
        scale = id_add_audit["reports"]["stage4.1.add"]["scale_selection"]
        return max(float(scale["eps_out"]) / float(scale["D"]) * 0.5, 1e-6)
    if alias == "stage4_1_add_post_requant":
        scale = id_add_audit["reports"]["stage4.1.add"]["scale_selection"]
        return max(float(scale["eps_out"]) * 0.5, 1e-6)
    if alias in {"global_pool_post_requant", "head_input"}:
        if head_eps_in is None:
            return 1e-6
        return max(float(head_eps_in) * 0.5, 1e-6)
    if alias == "model_output":
        return 0.5 / 32768.0
    return 1e-6


def semanticize_focus_taps(
    fq_probe: dict[str, Any],
    id_probe: dict[str, Any],
    onnx_probe: dict[str, Any],
    id_add_audit: dict[str, Any],
    id_quant_context: dict[str, Any],
    head_eps_in: float | None,
) -> dict[str, dict[str, np.ndarray]]:
    fq = {
        "stage4_1_conv1": np.asarray(fq_probe["tensors"]["stage4_1_conv1"], dtype=np.float64),
        "stage4_1_conv2": np.asarray(fq_probe["tensors"]["stage4_1_conv2"], dtype=np.float64),
        "stage4_1_add_pre_requant": np.asarray(fq_probe["tensors"]["stage4_1_add"], dtype=np.float64),
        "stage4_1_add_post_requant": np.asarray(fq_probe["tensors"]["stage4_1_add"], dtype=np.float64),
        "global_pool_post_requant": np.asarray(fq_probe["tensors"]["global_pool_post_requant"], dtype=np.float64),
        "head_input": np.asarray(fq_probe["tensors"]["head_input"], dtype=np.float64),
        "model_output": np.asarray(fq_probe["tensors"]["model_output"], dtype=np.float64).reshape(-1),
    }
    id_semantic = {
        alias: semantic_tensor_for_id_probe(alias, id_probe, id_add_audit, id_quant_context, head_eps_in)
        for alias in TAP_ALIAS_ORDER
    }
    onnx_semantic = {
        alias: semantic_tensor_for_onnx_probe(alias, onnx_probe, id_add_audit, id_quant_context, head_eps_in)
        for alias in TAP_ALIAS_ORDER
    }
    return {
        "fq": fq,
        "id": id_semantic,
        "onnx": onnx_semantic,
    }


def save_focus_tensor_dumps(focus_dir: Path, tensors_by_stage: dict[str, dict[str, np.ndarray]]) -> None:
    dump_dir = focus_dir / "tensor_dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, tensors in tensors_by_stage.items():
        for alias, tensor in tensors.items():
            np.save(dump_dir / f"{stage_name}__{alias}.npy", np.asarray(tensor))


def build_focus_tap_report(
    image_record: dict[str, Any],
    fq_probe: dict[str, Any],
    id_probe: dict[str, Any],
    onnx_probe: dict[str, Any],
    id_add_audit: dict[str, Any],
    id_quant_context: dict[str, Any],
    head_eps_in: float | None,
) -> dict[str, Any]:
    tensors = semanticize_focus_taps(
        fq_probe,
        id_probe,
        onnx_probe,
        id_add_audit,
        id_quant_context,
        head_eps_in,
    )
    points = []
    first_divergent = None
    for alias in TAP_ALIAS_ORDER:
        fq_tensor = tensors["fq"][alias]
        id_tensor = tensors["id"][alias]
        onnx_tensor = tensors["onnx"][alias]
        tolerance = semantic_tolerance_for_alias(alias, id_add_audit, id_quant_context, head_eps_in)
        id_vs_onnx = compare_arrays(id_tensor, onnx_tensor)
        point = {
            "alias": alias,
            "label": TAP_LABELS[alias],
            "onnx_node": onnx_probe["selected"][alias],
            "semantic_tolerance": tolerance,
            "fq_stats": tensor_stats(fq_tensor),
            "id_stats": tensor_stats(id_tensor),
            "onnx_stats": tensor_stats(onnx_tensor),
            "fq_vs_id": compare_arrays(fq_tensor, id_tensor),
            "id_vs_onnx": id_vs_onnx,
            "fq_vs_onnx": compare_arrays(fq_tensor, onnx_tensor),
            "id_vs_onnx_exceeds_tolerance": bool(float(id_vs_onnx["max_abs_diff"]) > tolerance),
        }
        if alias == "model_output":
            point["id_raw_vs_onnx_raw"] = compare_arrays(
                np.asarray(id_probe["tensors"]["model_output"], dtype=np.float64),
                np.asarray(onnx_probe["tensors"]["model_output"], dtype=np.float64),
            )
        points.append(point)
        if first_divergent is None and point["id_vs_onnx_exceeds_tolerance"]:
            first_divergent = {
                "alias": alias,
                "label": TAP_LABELS[alias],
                "onnx_node": onnx_probe["selected"][alias],
                "id_vs_onnx": id_vs_onnx,
                "semantic_tolerance": tolerance,
            }

    return {
        "image": {
            "index": image_record["index"],
            "image_name": image_record["image_name"],
            "image_path": image_record["image_path"],
        },
        "points": points,
        "first_divergent_onnx_tap": first_divergent,
        "stage4_1_add_scale_selection": id_add_audit["reports"]["stage4.1.add"]["scale_selection"],
        "stage4_1_quant_context": id_quant_context,
    }


def focus_report_markdown(focus_report: dict[str, Any]) -> str:
    image = focus_report["image"]
    lines = [
        "# Representative Batch Focus Report",
        "",
        f"- Image: `{image['image_name']}`",
        f"- Source: `{image['image_path']}`",
    ]
    first_divergent = focus_report.get("first_divergent_onnx_tap")
    if first_divergent is None:
        lines.append("- ID -> ONNX tap fidelity: no semantic tap exceeded its per-stage tolerance.")
    else:
        lines.append(
            "- First divergent ONNX tap: `{}` at node `{}` (`max_abs_diff={:.6f}`, tolerance `{:.6f}`).".format(
                first_divergent["label"],
                first_divergent["onnx_node"]["node_name"],
                float(first_divergent["id_vs_onnx"]["max_abs_diff"]),
                float(first_divergent["semantic_tolerance"]),
            )
        )

    lines.extend(
        [
            "",
            "## Tap Comparisons",
            "",
            "| Tap | FQ->ID mean abs | ID->ONNX max abs | ID->ONNX tol |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for point in focus_report["points"]:
        lines.append(
            "| `{}` | `{:.6f}` | `{:.6f}` | `{:.6f}` |".format(
                point["label"],
                float(point["fq_vs_id"]["mean_abs_diff"]),
                float(point["id_vs_onnx"]["max_abs_diff"]),
                float(point["semantic_tolerance"]),
            )
        )
    return "\n".join(lines)


def select_focus_records(
    rank_a: list[dict[str, Any]],
    rank_b: list[dict[str, Any]],
    *,
    per_ranking: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    candidates = list(rank_a[:per_ranking]) + list(rank_b[:per_ranking]) + list(rank_a[per_ranking:]) + list(rank_b[per_ranking:])
    for candidate in candidates:
        image_key = candidate["image_name"]
        if image_key in seen:
            continue
        seen.add(image_key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def summary_markdown(payload: dict[str, Any]) -> str:
    export_fidelity = payload["export_fidelity"]
    transition_counts = payload["transition_counts"]
    lines = [
        "# Representative Batch Stage Localization",
        "",
        f"- Batch size: `{payload['count']}`",
        f"- Export preset: `{payload['hybrid_follow_export_preset']}`",
        f"- Residual-add policy default: `{payload['integer_add_scale_policy']}`",
        f"- Conv-bias integerization fix: `enabled`",
        f"- run_val source: `{payload['run_val_results_dir']}`",
        "",
        "## First Material Transition Counts",
        "",
    ]
    for key, value in transition_counts.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Export Fidelity",
            "",
            f"- ID -> ONNX final raw exact matches: `{export_fidelity['id_to_onnx_final_raw_exact_matches']}/{payload['count']}`",
            f"- ID -> ONNX threshold warnings: `{export_fidelity['id_to_onnx_threshold_warn_count']}`",
            f"- ONNX -> golden final raw exact matches: `{export_fidelity['onnx_to_golden_final_raw_exact_matches']}/{payload['count']}`",
            "",
            "## Worst PyTorch/NEMO -> ONNX Drift",
            "",
            "| Image | FP->ONNX score | First failing transition | ID->ONNX score |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for item in payload["rankings"]["fp_to_onnx"][:5]:
        lines.append(
            "| `{}` | `{:.6f}` | `{}` | `{:.6f}` |".format(
                item["image_name"],
                float(item["scores"]["fp_to_onnx"]),
                item["first_material_transition"]["transition_label"] or "none",
                float(item["scores"]["id_to_onnx"]),
            )
        )

    lines.extend(
        [
            "",
            "## Worst ONNX -> Application Drift",
            "",
            "| Image | ONNX->application score | First failing transition |",
            "| --- | ---: | --- |",
        ]
    )
    for item in payload["rankings"]["onnx_to_application"][:5]:
        lines.append(
            "| `{}` | `{:.6f}` | `{}` |".format(
                item["image_name"],
                float(item["scores"]["onnx_to_application"]),
                item["first_material_transition"]["transition_label"] or "none",
            )
        )

    lines.extend(
        [
            "",
            "## Focus Images",
            "",
        ]
    )
    for item in payload["focus_images"]:
        lines.append(f"- `{item['image_name']}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_val_results_dir = resolve_repo_path(args.run_val_results_dir)
    ckpt_path = resolve_repo_path(args.ckpt)
    onnx_path = resolve_repo_path(args.onnx)
    calib_dir = resolve_repo_path(args.calib_dir)

    if run_val_results_dir is None or not run_val_results_dir.is_dir():
        raise FileNotFoundError(f"run_val results directory not found: {args.run_val_results_dir}")
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if onnx_path is None or not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if calib_dir is None or not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {args.calib_dir}")

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else (run_val_results_dir / "stage_localization")
    output_dir = Path(output_dir)
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    summary_payload = load_run_val_summary(run_val_results_dir)
    batch_items = collect_batch_items(summary_payload, args.limit)
    thresholds = Thresholds(args.warn_x_abs_diff, args.warn_size_abs_diff, args.warn_vis_conf_abs_diff)

    models = build_stage_models(args, ckpt_path, calib_dir)
    onnx_ctx = build_onnx_context(onnx_path, args.height, args.width)

    per_image_records = []
    for row in batch_items:
        image_path = resolve_repo_path(row["image_path"])
        sample_dir = resolve_repo_path(row["application_sample_dir"])
        if image_path is None or sample_dir is None:
            raise FileNotFoundError(f"Missing image or sample dir for row: {row}")

        x_float, x_uint8 = preprocess_image_once(image_path, args.height, args.width)
        x_staged = x_uint8.to(dtype=torch.float32)

        fp_output = run_model_output(models["model_fp"], x_float)
        fq_output = run_model_output(models["model_fq"], x_float)
        qd_output = run_model_output(models["model_qd"], x_staged)
        id_output = run_model_output(models["model_id"], x_staged)
        onnx_output = run_onnx_output(onnx_ctx, x_staged)

        sample_artifacts = read_sample_artifacts(sample_dir)

        stage_results = {
            "fp": make_stage_result(key="fp", label="PyTorch FP", stage_tag="fp", representation="float", raw_native=fp_output),
            "fq": make_stage_result(key="fq", label="NEMO FQ", stage_tag="fq", representation="float", raw_native=fq_output),
            "qd": make_stage_result(key="qd", label="NEMO QD", stage_tag="qd", representation="fixed-point-int32", raw_native=qd_output),
            "id": make_stage_result(key="id", label="NEMO ID", stage_tag="id", representation="fixed-point-int32", raw_native=id_output),
            "onnx": make_stage_result(key="onnx", label="Exported ONNX", stage_tag="id", representation="fixed-point-int32", raw_native=onnx_output),
            "golden": make_stage_result(key="golden", label="Golden artifact", stage_tag="id", representation="fixed-point-int32", raw_native=sample_artifacts["golden"]),
            "application": make_stage_result(key="application", label="Application/GVSOC", stage_tag="id", representation="fixed-point-int32", raw_native=sample_artifacts["gvsoc"]),
        }

        pairwise = {
            "fp_to_fq": compare_stage_pair(stage_results["fp"], stage_results["fq"], thresholds),
            "fq_to_id": compare_stage_pair(stage_results["fq"], stage_results["id"], thresholds),
            "id_to_onnx": compare_stage_pair(stage_results["id"], stage_results["onnx"], thresholds),
            "onnx_to_application": compare_stage_pair(stage_results["onnx"], stage_results["application"], thresholds),
            "fp_to_onnx": compare_stage_pair(stage_results["fp"], stage_results["onnx"], thresholds),
            "onnx_to_golden": compare_stage_pair(stage_results["onnx"], stage_results["golden"], thresholds),
        }
        first_transition = first_material_transition(pairwise)

        id_raw = np.rint(np.asarray(id_output).reshape(-1)).astype(np.int64)
        onnx_raw = np.rint(np.asarray(onnx_output).reshape(-1)).astype(np.int64)
        golden_raw = np.asarray(sample_artifacts["golden"], dtype=np.int64).reshape(-1)

        record = {
            "index": int(row["index"]),
            "image_name": row["image_name"],
            "image_path": str(image_path),
            "application_sample_dir": str(sample_dir),
            "final_outputs": {
                name: stage_result_dict(stage_result)
                for name, stage_result in stage_results.items()
            },
            "pairwise": pairwise,
            "first_material_transition": first_transition,
            "scores": {
                "fp_to_onnx": pair_score(pairwise["fp_to_onnx"]),
                "onnx_to_application": pair_score(pairwise["onnx_to_application"]),
                "id_to_onnx": pair_score(pairwise["id_to_onnx"]),
            },
            "export_fidelity": {
                "id_to_onnx_final_raw_exact_match": bool(np.array_equal(id_raw, onnx_raw)),
                "onnx_to_golden_final_raw_exact_match": bool(np.array_equal(onnx_raw, golden_raw)),
                "id_to_onnx_raw_compare": compare_arrays(id_raw, onnx_raw),
            },
        }
        per_image_records.append(record)

    fp_to_onnx_ranking = sorted(
        per_image_records,
        key=lambda item: float(item["scores"]["fp_to_onnx"] if item["scores"]["fp_to_onnx"] is not None else -1.0),
        reverse=True,
    )
    onnx_to_application_ranking = sorted(
        per_image_records,
        key=lambda item: float(item["scores"]["onnx_to_application"] if item["scores"]["onnx_to_application"] is not None else -1.0),
        reverse=True,
    )

    focus_candidates = select_focus_records(
        fp_to_onnx_ranking,
        onnx_to_application_ranking,
        per_ranking=args.focus_per_ranking,
        limit=args.tap_limit,
    )

    onnx_probe_ctx = build_onnx_probe_context(onnx_ctx, output_dir / "probe_cache")
    focus_reports = []
    for item in focus_candidates:
        image_path = Path(item["image_path"])
        x_float, x_uint8 = preprocess_image_once(image_path, args.height, args.width)
        x_staged = x_uint8.to(dtype=torch.float32)

        fq_probe = run_hybrid_follow_pytorch_probe(models["model_fq"], x_float)
        id_probe = run_hybrid_follow_pytorch_probe(models["model_id"], x_staged)
        id_add_audit = run_hybrid_follow_integer_add_audit(models["model_id"], x_staged)
        onnx_probe = run_onnx_probe(onnx_probe_ctx, x_staged)

        focus_report = build_focus_tap_report(
            item,
            fq_probe,
            id_probe,
            onnx_probe,
            id_add_audit,
            models["id_quant_context"],
            models["head_eps_in"],
        )
        focus_reports.append(focus_report)

        focus_dir = output_dir / "focus" / f"{int(item['index']):04d}_{sanitize_name(Path(item['image_name']).stem)}"
        focus_dir.mkdir(parents=True, exist_ok=True)
        save_focus_tensor_dumps(
            focus_dir,
            semanticize_focus_taps(
                fq_probe,
                id_probe,
                onnx_probe,
                id_add_audit,
                models["id_quant_context"],
                models["head_eps_in"],
            ),
        )
        write_json(focus_dir / "tap_report.json", focus_report)
        write_markdown(focus_dir / "tap_report.md", focus_report_markdown(focus_report))

        for record in per_image_records:
            if record["image_name"] == item["image_name"]:
                record["export_fidelity"]["first_divergent_onnx_tap"] = focus_report["first_divergent_onnx_tap"]
                record["focus_report_dir"] = str(focus_dir)
                break

    transition_counts = {
        "none": 0,
        "FP -> FQ": 0,
        "FQ -> ID": 0,
        "ID -> ONNX": 0,
        "ONNX -> application": 0,
    }
    for record in per_image_records:
        label = record["first_material_transition"]["transition_label"]
        if label is None:
            transition_counts["none"] += 1
        else:
            transition_counts[label] += 1

    export_fidelity = {
        "id_to_onnx_final_raw_exact_matches": sum(
            1 for record in per_image_records if record["export_fidelity"]["id_to_onnx_final_raw_exact_match"]
        ),
        "onnx_to_golden_final_raw_exact_matches": sum(
            1 for record in per_image_records if record["export_fidelity"]["onnx_to_golden_final_raw_exact_match"]
        ),
        "id_to_onnx_threshold_warn_count": sum(
            1 for record in per_image_records if record["pairwise"]["id_to_onnx"]["status"] == "warn"
        ),
    }

    result_payload = {
        "run_val_results_dir": str(run_val_results_dir),
        "output_dir": str(output_dir),
        "count": len(per_image_records),
        "hybrid_follow_export_preset": models["export_args"].hybrid_follow_export_preset,
        "integer_add_scale_policy": HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
        "thresholds": asdict(thresholds),
        "export_fidelity": export_fidelity,
        "transition_counts": transition_counts,
        "rankings": {
            "fp_to_onnx": fp_to_onnx_ranking,
            "onnx_to_application": onnx_to_application_ranking,
        },
        "focus_images": [
            {
                "index": report["image"]["index"],
                "image_name": report["image"]["image_name"],
                "focus_report_dir": str(
                    output_dir / "focus" / f"{int(report['image']['index']):04d}_{sanitize_name(Path(report['image']['image_name']).stem)}"
                ),
            }
            for report in focus_reports
        ],
        "images": per_image_records,
        "onnx_probe": {
            "probe_model_path": str(onnx_probe_ctx["probe_path"]),
            "selected": onnx_probe_ctx["selected"],
        },
    }

    write_json(output_dir / "summary.json", result_payload)
    write_markdown(output_dir / "summary.md", summary_markdown(result_payload))
    write_json(output_dir / "ranked_fp_to_onnx.json", fp_to_onnx_ranking)
    write_json(output_dir / "ranked_onnx_to_application.json", onnx_to_application_ranking)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DORY_DIR = REPO_DIR / "dory"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(DORY_DIR) not in sys.path:
    sys.path.insert(0, str(DORY_DIR))

from hybrid_follow_image_artifacts import preprocess_image_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DORY-graph semantic inference for quant-native follow models using "
            "the parsed DORY graph instead of ONNXRuntime on the cleaned ONNX."
        )
    )
    parser.add_argument("--onnx", required=True, help="Cleaned DORY ONNX path.")
    parser.add_argument("--config", required=True, help="DORY config JSON path used for network_generate.")
    parser.add_argument(
        "--input-bundle",
        required=True,
        help="JSON bundle containing pre-staged uint8 follow inputs.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path for raw int32 predictions.")
    parser.add_argument("--frontend", default="NEMO")
    parser.add_argument("--target", default="PULP.GAP8")
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def normalize_staged_input(raw_value: Any) -> np.ndarray:
    arr = np.asarray(raw_value)
    if arr.size == 0:
        raise RuntimeError("Staged input is empty.")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return np.clip(arr, 0, 255).astype(np.uint8, copy=False).reshape(-1)


def trunc_int(arr: Any, dtype: Any) -> np.ndarray:
    return np.trunc(np.asarray(arr)).astype(dtype)


def infer_integer_dtype(
    flat: np.ndarray,
    *,
    item_count: int,
    signed: bool,
    default_dtype: Any,
) -> np.dtype:
    default = np.dtype(default_dtype)
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if flat.size == item_count:
        return default
    if flat.size % item_count != 0:
        raise RuntimeError(
            f"Packed constant size {flat.size} is not divisible by expected item_count {item_count}"
        )
    bytes_per_item = flat.size // item_count
    if bytes_per_item not in (1, 2, 4, 8):
        raise RuntimeError(f"Unsupported packed integer width: {bytes_per_item} bytes")
    return np.dtype(f"<{'i' if signed else 'u'}{bytes_per_item}")


def decode_integer_constant(
    raw_value: Any,
    *,
    item_count: int,
    signed: bool = True,
    default_dtype: Any = np.int32,
) -> np.ndarray:
    arr = np.asarray(raw_value)
    if arr.ndim == 0:
        return arr.astype(default_dtype, copy=False).reshape(1)

    flat = arr.reshape(-1)
    dtype = infer_integer_dtype(
        flat,
        item_count=item_count,
        signed=signed,
        default_dtype=default_dtype,
    )
    if flat.size == item_count:
        return flat.astype(dtype, copy=False)

    expected_bytes = int(item_count * dtype.itemsize)
    if flat.size != expected_bytes:
        raise RuntimeError(
            f"Packed constant byte stream size mismatch: expected {expected_bytes}, got {flat.size}"
        )
    return flat.astype(np.uint8, copy=False).view(dtype)


def resolve_weight_constant_name(node: Any) -> str | None:
    excluded = {"k", "l", "outshift", "outmul", "outadd", "inmul1", "inmul2", "inshift1", "inshift2", "inadd1", "inadd2"}
    for name in getattr(node, "constant_names", []):
        if name in excluded or "bias" in name:
            continue
        return str(name)
    return None


def resolve_bias_constant_name(node: Any) -> str | None:
    for name in getattr(node, "constant_names", []):
        if "bias" in str(name):
            return str(name)
    return None


def scalar_constant(node: Any, name: str, default: int = 0) -> int:
    if not hasattr(node, name):
        return int(default)
    value = getattr(node, name)
    if isinstance(value, dict):
        value = value.get("value")
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    return int(np.trunc(arr.reshape(-1)[0]))


def clip_to_output_range(values: np.ndarray, node: Any) -> np.ndarray:
    bits = int(getattr(node, "output_activation_bits", 8) or 8)
    signed = str(getattr(node, "output_activation_type", "uint")) == "int"
    if signed:
        low = -(1 << (bits - 1))
        high = (1 << (bits - 1)) - 1
    else:
        low = 0
        high = (1 << bits) - 1
    return np.clip(values, low, high)


def cast_activation(values: np.ndarray, node: Any) -> np.ndarray:
    signed = str(getattr(node, "output_activation_type", "uint")) == "int"
    bits = int(getattr(node, "output_activation_bits", 8) or 8)
    if bits <= 8:
        dtype = np.int8 if signed else np.uint8
    elif bits <= 16:
        dtype = np.int16 if signed else np.uint16
    else:
        dtype = np.int32 if signed else np.uint32
    return clip_to_output_range(values, node).astype(dtype)


def conv2d_hwc(inp: np.ndarray, weights: np.ndarray, stride: list[int], pads: list[int], groups: int) -> np.ndarray:
    if groups != 1:
        raise NotImplementedError(f"DORY semantic simulator currently supports group=1 only, got {groups}")
    top, left, bottom, right = (int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3]))
    stride_y, stride_x = int(stride[0]), int(stride[1])
    kernel_h, kernel_w = int(weights.shape[1]), int(weights.shape[2])
    padded = np.pad(inp.astype(np.int32), ((top, bottom), (left, right), (0, 0)), mode="constant")
    windows = sliding_window_view(padded, (kernel_h, kernel_w), axis=(0, 1))
    windows = windows[::stride_y, ::stride_x, :, :, :]
    windows = np.transpose(windows, (0, 1, 3, 4, 2)).astype(np.int32)
    out = np.tensordot(
        windows,
        weights.astype(np.int32),
        axes=([2, 3, 4], [1, 2, 3]),
    )
    return out.astype(np.int64)


def avg_pool_hwc(
    inp: np.ndarray,
    kernel: list[int],
    stride: list[int],
    pads: list[int],
    *,
    outmul: int,
    outshift: int,
    outadd: int,
    requant: bool,
    node: Any,
) -> np.ndarray:
    top, left, bottom, right = (int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3]))
    stride_y, stride_x = int(stride[0]), int(stride[1])
    kernel_h, kernel_w = int(kernel[0]), int(kernel[1])
    padded = np.pad(inp.astype(np.int64), ((top, bottom), (left, right), (0, 0)), mode="constant")
    windows = sliding_window_view(padded, (kernel_h, kernel_w), axis=(0, 1))
    windows = windows[::stride_y, ::stride_x, :, :, :]
    sums = windows.sum(axis=(3, 4), dtype=np.int64)
    kernel_size_total = int(kernel_h * kernel_w)
    if requant:
        values = (((sums * int(outmul)) // kernel_size_total) + int(outadd)) >> int(outshift)
    else:
        values = sums // kernel_size_total
    return cast_activation(values, node)


def linear_out_32_hwc(inp: np.ndarray, weights_flat: np.ndarray, bias: np.ndarray, out_channels: int) -> np.ndarray:
    flat = inp.reshape(-1).astype(np.int32)
    weights = weights_flat.reshape(int(out_channels), flat.size).astype(np.int32)
    bias_arr = bias.astype(np.int32, copy=False)
    return (weights @ flat + bias_arr).astype(np.int32)


def simulate_dory_graph_trace(graph: list[Any], staged_input_flat: list[int]) -> list[np.ndarray]:
    if not graph:
        raise RuntimeError("DORY graph is empty.")
    first = graph[0]
    input_channels = int(getattr(first, "input_channels"))
    input_h, input_w = [int(v) for v in getattr(first, "input_dimensions")]
    x = np.asarray(staged_input_flat, dtype=np.uint8)
    expected_count = int(input_channels * input_h * input_w)
    if x.size != expected_count:
        raise RuntimeError(f"Staged input size mismatch: expected {expected_count}, got {x.size}")
    x = x.reshape(input_channels, input_h, input_w).transpose(1, 2, 0)
    traces: list[np.ndarray] = []

    for node in graph:
        node_name = str(getattr(node, "name", ""))
        weight_name = resolve_weight_constant_name(node)
        bias_name = resolve_bias_constant_name(node)

        if "Convolution" in node_name:
            if weight_name is None:
                raise RuntimeError(f"Missing weights for node {node_name}")
            output_channels = int(getattr(node, "output_channels"))
            input_channels = int(getattr(node, "input_channels"))
            kernel_h, kernel_w = [int(value) for value in getattr(node, "kernel_shape")]
            weight_count = int(output_channels * input_channels * kernel_h * kernel_w)
            weights = decode_integer_constant(
                getattr(node, weight_name)["value"],
                item_count=weight_count,
                signed=True,
                default_dtype=np.int8,
            ).astype(np.int8, copy=False)
            weights = weights.reshape(output_channels, kernel_h, kernel_w, input_channels)
            values = conv2d_hwc(x, weights, list(getattr(node, "strides")), list(getattr(node, "pads")), int(getattr(node, "group")))
            if bias_name is not None:
                bias = decode_integer_constant(
                    getattr(node, bias_name)["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int32, copy=False).reshape(1, 1, -1)
                values = values + bias
            if node_name.startswith("BNRelu"):
                k = decode_integer_constant(
                    getattr(node, "k")["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int64, copy=False).reshape(1, 1, -1)
                lam = decode_integer_constant(
                    getattr(node, "l")["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int64, copy=False).reshape(1, 1, -1)
                outshift = scalar_constant(node, "outshift")
                x = cast_activation(((values * k) + lam) >> outshift, node)
            elif node_name.startswith("Relu"):
                outmul = scalar_constant(node, "outmul", default=1)
                outshift = scalar_constant(node, "outshift")
                x = cast_activation((values * outmul) >> outshift, node)
            else:
                x = values.astype(np.int32)
            traces.append(np.asarray(x).copy())
            continue

        if "Pooling" in node_name:
            has_requant = hasattr(node, "outmul") or hasattr(node, "outadd") or hasattr(node, "outshift")
            x = avg_pool_hwc(
                x,
                list(getattr(node, "kernel_shape")),
                list(getattr(node, "strides")),
                list(getattr(node, "pads")),
                outmul=scalar_constant(node, "outmul", default=1),
                outshift=scalar_constant(node, "outshift", default=0),
                outadd=scalar_constant(node, "outadd", default=0),
                requant=has_requant,
                node=node,
            )
            traces.append(np.asarray(x).copy())
            continue

        if "FullyConnected" in node_name or str(getattr(node, "op_type", "")) == "Gemm":
            if weight_name is None:
                raise RuntimeError(f"Missing weights for node {node_name}")
            output_channels = int(getattr(node, "output_channels"))
            input_count = int(np.asarray(x).reshape(-1).size)
            weight_count = int(output_channels * input_count)
            weights = decode_integer_constant(
                getattr(node, weight_name)["value"],
                item_count=weight_count,
                signed=True,
                default_dtype=np.int8,
            ).astype(np.int8, copy=False)
            if bias_name is not None:
                bias = decode_integer_constant(
                    getattr(node, bias_name)["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int32, copy=False)
            else:
                bias = np.zeros((output_channels,), dtype=np.int32)
            values = linear_out_32_hwc(x, weights, bias, output_channels)
            if node_name.startswith("BNRelu"):
                k = decode_integer_constant(
                    getattr(node, "k")["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int64, copy=False)
                lam = decode_integer_constant(
                    getattr(node, "l")["value"],
                    item_count=output_channels,
                    signed=True,
                    default_dtype=np.int32,
                ).astype(np.int64, copy=False)
                outshift = scalar_constant(node, "outshift")
                x = cast_activation(((values.astype(np.int64) * k) + lam) >> outshift, node)
            elif node_name.startswith("Relu"):
                outmul = scalar_constant(node, "outmul", default=1)
                outshift = scalar_constant(node, "outshift")
                x = cast_activation(values.astype(np.int64) * outmul >> outshift, node)
            else:
                x = values.astype(np.int32)
            traces.append(np.asarray(x).copy())
            continue

        raise NotImplementedError(f"Unsupported DORY node for semantic simulation: {node_name}")

    return traces


def simulate_dory_graph(graph: list[Any], staged_input_flat: list[int]) -> np.ndarray:
    traces = simulate_dory_graph_trace(graph, staged_input_flat)
    if not traces:
        raise RuntimeError("DORY graph produced no outputs.")
    return np.asarray(traces[-1]).reshape(-1).astype(np.int32)


def build_dory_graph(onnx_path: Path, config: dict[str, Any], *, frontend: str, target: str, prefix: str) -> list[Any]:
    frontend_parser = import_module(f"dory.Frontend_frameworks.{frontend}.Parser").onnx_manager
    hw_parser_cls = import_module(f"dory.Hardware_targets.{target}.HW_Parser").onnx_manager
    dory_graph = frontend_parser(str(onnx_path), config, prefix).full_graph_parsing()
    hw_parser = hw_parser_cls(dory_graph, config, str(Path(config["__config_dir__"])), config.get("n_inputs", 1))
    return list(hw_parser.full_graph_parsing())


def infer_graph_input_hw(graph: list[Any]) -> tuple[int, int]:
    if not graph:
        raise RuntimeError("DORY graph is empty.")
    first = graph[0]
    input_dimensions = [int(value) for value in getattr(first, "input_dimensions")]
    if len(input_dimensions) != 2:
        raise RuntimeError(f"Expected 2D input_dimensions, got {input_dimensions}")
    return int(input_dimensions[0]), int(input_dimensions[1])


def resolve_sample_staged_input(
    *,
    graph: list[Any],
    bundle: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[np.ndarray, str]:
    staging_mode = str(sample.get("staging_mode") or bundle.get("staging_mode") or "").strip().lower()
    model_type = str(sample.get("model_type") or bundle.get("model_type") or "hybrid_follow")
    image_path_raw = sample.get("image_path")

    if staging_mode in {"runtime_preprocess_uint8", "preprocess_image_uint8", "app_runtime_uint8"}:
        if not image_path_raw:
            raise RuntimeError(
                "Bundle requested runtime_preprocess_uint8 staging but sample did not include image_path."
            )
        input_h, input_w = infer_graph_input_hw(graph)
        staged_input = preprocess_image_uint8(
            image_path=Path(str(image_path_raw)).expanduser().resolve(),
            height=int(input_h),
            width=int(input_w),
            model_type=model_type,
        )
        return normalize_staged_input(staged_input), "runtime_preprocess_uint8"

    raw_staged_input = sample.get("staged_input")
    if raw_staged_input is None:
        if image_path_raw:
            input_h, input_w = infer_graph_input_hw(graph)
            staged_input = preprocess_image_uint8(
                image_path=Path(str(image_path_raw)).expanduser().resolve(),
                height=int(input_h),
                width=int(input_w),
                model_type=model_type,
            )
            return normalize_staged_input(staged_input), "runtime_preprocess_uint8_fallback"
        raise RuntimeError("Sample did not include staged_input or image_path.")
    return normalize_staged_input(raw_staged_input), "bundle_staged_input"


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    input_bundle_path = Path(args.input_bundle).expanduser().resolve()
    output_json_path = Path(args.output_json).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not input_bundle_path.is_file():
        raise FileNotFoundError(f"Input bundle not found: {input_bundle_path}")

    config = read_json(config_path)
    config["__config_dir__"] = str(config_path.parent)
    dory_graph = build_dory_graph(
        onnx_path,
        config,
        frontend=str(args.frontend),
        target=str(args.target),
        prefix=str(args.prefix),
    )

    bundle = read_json(input_bundle_path)
    samples = list(bundle.get("samples") or [])
    if not samples:
        raise RuntimeError(f"No samples found in input bundle: {input_bundle_path}")

    rows = []
    for sample in samples:
        staged_input, staged_input_source = resolve_sample_staged_input(
            graph=dory_graph,
            bundle=bundle,
            sample=sample,
        )
        raw_output = simulate_dory_graph(dory_graph, staged_input.tolist())
        rows.append(
            {
                "image_name": sample.get("image_name"),
                "image_path": sample.get("image_path"),
                "staged_input_source": staged_input_source,
                "raw_output": [int(value) for value in raw_output.tolist()],
                "raw_output_shape": [int(raw_output.size)],
            }
        )

    payload = {
        "simulation_source": "dory_hw_graph_python",
        "onnx_path": str(onnx_path),
        "config_path": str(config_path),
        "bundle_staging_mode": str(bundle.get("staging_mode") or ""),
        "bundle_model_type": str(bundle.get("model_type") or ""),
        "frontend": str(args.frontend),
        "target": str(args.target),
        "prefix": str(args.prefix),
        "graph": [
            {
                "index": index,
                "name": str(getattr(node, "name", "")),
                "op_type": str(getattr(node, "op_type", "")),
                "output_index": str(getattr(node, "output_index", "")),
            }
            for index, node in enumerate(dory_graph)
        ],
        "samples": rows,
    }
    write_json(output_json_path, payload)


if __name__ == "__main__":
    main()

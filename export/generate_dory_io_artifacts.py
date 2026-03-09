#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "tensor"


def _write_values_txt(path: Path, values: np.ndarray, header: str, as_int: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header},\n")
        if as_int:
            for v in values:
                f.write(f"{int(v)},\n")
        else:
            for v in values:
                f.write(f"{float(v):.8f},\n")


def _onnx_type_to_numpy(elem_type: int):
    if elem_type == TensorProto.FLOAT:
        return np.float32
    if elem_type == TensorProto.FLOAT16:
        return np.float16
    if elem_type == TensorProto.DOUBLE:
        return np.float64
    if elem_type == TensorProto.UINT8:
        return np.uint8
    if elem_type == TensorProto.INT8:
        return np.int8
    if elem_type == TensorProto.UINT16:
        return np.uint16
    if elem_type == TensorProto.INT16:
        return np.int16
    if elem_type == TensorProto.UINT32:
        return np.uint32
    if elem_type == TensorProto.INT32:
        return np.int32
    if elem_type == TensorProto.UINT64:
        return np.uint64
    if elem_type == TensorProto.INT64:
        return np.int64
    if elem_type == TensorProto.BOOL:
        return np.bool_
    return np.float32


def _value_info_shape(value_info) -> List[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(1)
    return dims


def _get_model_input_meta(
    model: onnx.ModelProto,
    fallback_height: int,
    fallback_width: int,
) -> Tuple[str, int, List[int]]:
    initializer_names = {init.name for init in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in initializer_names]
    if not real_inputs:
        raise RuntimeError("No non-initializer ONNX input tensors found.")
    model_input = real_inputs[0]
    input_name = model_input.name
    elem_type = model_input.type.tensor_type.elem_type
    shape = _value_info_shape(model_input)
    if not shape:
        shape = [1, 3, fallback_height, fallback_width]
    return input_name, elem_type, shape


def _convert_input_for_model(x_uint8: np.ndarray, input_shape: Sequence[int], elem_type: int) -> np.ndarray:
    x = x_uint8.reshape(input_shape)
    if elem_type == TensorProto.INT8:
        return (x.astype(np.int16) - 128).astype(np.int8)
    if elem_type == TensorProto.BOOL:
        return x.astype(np.uint8) > 127
    dtype = _onnx_type_to_numpy(elem_type)
    return x.astype(dtype)


def _collect_tensor_meta(model: onnx.ModelProto) -> Dict[str, Tuple[int, List[object]]]:
    meta: Dict[str, Tuple[int, List[object]]] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == 0:
            continue
        shape: List[object] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        meta[value.name] = (tensor_type.elem_type, shape if shape else None)
    return meta


def _add_intermediate_outputs(model: onnx.ModelProto, tensor_names: Sequence[str]) -> None:
    inferred = shape_inference.infer_shapes(model)
    metadata = _collect_tensor_meta(inferred)
    existing = {out.name for out in model.graph.output}

    for name in tensor_names:
        if name in existing:
            continue
        if name in metadata:
            elem_type, shape = metadata[name]
        else:
            # Fallback when shape inference does not emit metadata for a tensor.
            elem_type, shape = TensorProto.FLOAT, None
        model.graph.output.append(helper.make_tensor_value_info(name, elem_type, shape))
        existing.add(name)


def _collect_layer_output_tensors(
    onnx_path: str,
    conf: dict,
    conf_dir: str,
    frontend: str,
    target: str,
    prefix: str,
) -> List[str]:
    frontend_parser = import_module(f"dory.Frontend_frameworks.{frontend}.Parser").onnx_manager
    dory_graph = frontend_parser(onnx_path, conf, prefix).full_graph_parsing()

    hw_parser_cls = import_module(f"dory.Hardware_targets.{target}.HW_Parser").onnx_manager
    hw_parser = hw_parser_cls(dory_graph, conf, conf_dir, conf.get("n_inputs", 1))

    # Mirror network_generate flow up to checksum preparation, but stop before
    # tiling because some DORY tilers terminate the process with os._exit(0)
    # on infeasible constraints.
    hw_parser.mapping_to_HW_nodes()
    hw_parser.update_branches_graph()
    hw_parser.update_dimensions_graph()
    hw_parser.adjust_data_layout()
    hw_parser.add_tensors_memory_occupation_and_MACs()

    output_names = []
    for node in hw_parser.DORY_Graph:
        out_name = str(getattr(node, "output_index", "")).strip()
        if not out_name:
            raise RuntimeError("Found a DORY node with empty output_index while preparing golden activations.")
        output_names.append(out_name)

    if not output_names:
        raise RuntimeError("No DORY layer outputs found while preparing golden activations.")

    return output_names


def _export_weight_txt_files(model: onnx.ModelProto, weights_dir: Path) -> List[str]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, init in enumerate(model.graph.initializer):
        arr = numpy_helper.to_array(init)
        if arr.size == 0:
            continue
        flat = arr.reshape(-1)
        safe_name = _sanitize_filename(init.name)[:96]
        out_path = weights_dir / f"weight_{idx:03d}_{safe_name}.txt"
        header = f"{init.name} (shape {list(arr.shape)}, dtype {arr.dtype})"
        is_int_like = np.issubdtype(flat.dtype, np.integer) or np.issubdtype(flat.dtype, np.bool_)
        _write_values_txt(out_path, flat, header=header, as_int=is_int_like)
        written.append(str(out_path.resolve()))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DORY/NEMO txt artifacts for deployment checks.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX consumed by DORY.")
    parser.add_argument("--config", required=True, help="Path to DORY JSON config.")
    parser.add_argument("--frontend", default="NEMO", help="DORY frontend name (default: NEMO).")
    parser.add_argument("--target", default="PULP.GAP8", help="DORY target name (default: PULP.GAP8).")
    parser.add_argument("--prefix", default="", help="Optional generated symbol prefix.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed for input generation.")
    parser.add_argument("--weights-dir", default=None, help="Directory for layer weight .txt files.")
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest output path.")
    parser.add_argument("--fallback-height", type=int, default=160)
    parser.add_argument("--fallback-width", type=int, default=160)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        conf = json.load(f)

    io_dir = onnx_path.parent
    weights_dir = (
        Path(args.weights_dir).expanduser().resolve()
        if args.weights_dir
        else (io_dir / "weights_txt").resolve()
    )
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else (io_dir / "nemo_dory_artifacts.json").resolve()
    )

    model = onnx.load(str(onnx_path))
    input_name, input_elem_type, input_shape = _get_model_input_meta(
        model, args.fallback_height, args.fallback_width
    )

    rng = np.random.default_rng(args.seed)
    x_uint8 = rng.integers(0, 256, size=int(np.prod(input_shape)), dtype=np.uint8)
    input_path = io_dir / "input.txt"
    _write_values_txt(input_path, x_uint8, f"input (shape {list(input_shape)})", as_int=True)
    x_feed = _convert_input_for_model(x_uint8, input_shape, input_elem_type)
    print(f"[artifact_gen] input.txt prepared: {input_path}")
    print(f"[artifact_gen] collecting DORY layer outputs from frontend={args.frontend}, target={args.target}")

    layer_tensors = _collect_layer_output_tensors(
        onnx_path=str(onnx_path),
        conf=conf,
        conf_dir=str(config_path.parent),
        frontend=args.frontend,
        target=args.target,
        prefix=args.prefix,
    )
    print(f"[artifact_gen] DORY layer outputs discovered: {len(layer_tensors)}")

    primary_output_name = model.graph.output[0].name if model.graph.output else layer_tensors[-1]
    request_names = []
    seen = set()
    for name in list(layer_tensors) + [primary_output_name]:
        if name not in seen:
            request_names.append(name)
            seen.add(name)

    runtime_model = onnx.load(str(onnx_path))
    _add_intermediate_outputs(runtime_model, request_names)
    print(f"[artifact_gen] runtime outputs requested: {len(request_names)}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to generate DORY golden activations. "
            "Install it in doryenv and re-run."
        ) from exc

    temp_fd, temp_model_path = tempfile.mkstemp(
        prefix="dory_intermediate_", suffix=".onnx", dir=str(io_dir)
    )
    os.close(temp_fd)
    onnx.save(runtime_model, temp_model_path)

    output_map: Dict[str, np.ndarray] = {}
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            temp_model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        session_input_name = sess.get_inputs()[0].name
        results = sess.run(request_names, {session_input_name: x_feed})
        output_map = dict(zip(request_names, results))
        print("[artifact_gen] onnxruntime inference completed")
    finally:
        try:
            os.remove(temp_model_path)
        except OSError:
            pass

    golden_files = []
    for idx, tensor_name in enumerate(layer_tensors):
        if tensor_name not in output_map:
            raise RuntimeError(f"Missing runtime tensor '{tensor_name}' while writing out_layer files.")
        arr = np.asarray(output_map[tensor_name])
        flat = arr.reshape(-1)
        if np.issubdtype(flat.dtype, np.floating):
            flat = np.rint(flat).astype(np.int64)
        else:
            flat = flat.astype(np.int64)
        out_path = io_dir / f"out_layer{idx}.txt"
        _write_values_txt(
            out_path,
            flat,
            f"{tensor_name} (shape {list(arr.shape)})",
            as_int=True,
        )
        golden_files.append(str(out_path.resolve()))

    primary_arr = np.asarray(output_map[primary_output_name]).reshape(-1)
    if np.issubdtype(primary_arr.dtype, np.floating):
        primary_arr = np.rint(primary_arr).astype(np.int64)
    else:
        primary_arr = primary_arr.astype(np.int64)
    output_path = io_dir / "output.txt"
    _write_values_txt(
        output_path,
        primary_arr,
        f"{primary_output_name} (shape {list(np.asarray(output_map[primary_output_name]).shape)})",
        as_int=True,
    )

    weight_files = _export_weight_txt_files(model, weights_dir)

    manifest = {
        "quantized_model": str(onnx_path),
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "golden_activations": golden_files,
        "weight_files": weight_files,
        "num_layers": len(golden_files),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"[artifact_gen] input.txt: {input_path}")
    print(f"[artifact_gen] output.txt: {output_path}")
    print(f"[artifact_gen] golden activations: {len(golden_files)} files")
    print(f"[artifact_gen] weight txt files: {len(weight_files)} files in {weights_dir}")
    print(f"[artifact_gen] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

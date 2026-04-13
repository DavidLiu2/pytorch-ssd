#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.transforms import get_val_transforms  # noqa: E402


PREPROCESS_DESCRIPTION = (
    "Center-crop to square, resize to 128x128, convert to grayscale, "
    "then torchvision ToTensor() with no extra normalization."
)


@dataclass
class StageArtifacts:
    image_path: str
    output_dir: str
    input_txt: str
    input_hex: str
    output_txt: str
    metadata_json: str
    output_name: str
    input_shape: list[int]
    output_shape: list[int]
    expected_tensor: list[int]
    preprocess: str
    app_input_hex: str | None = None
    expected_source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _write_values_txt(path: Path, values: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header},\n")
        for value in values.reshape(-1):
            f.write(f"{int(value)},\n")


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


def _value_info_shape(value_info) -> list[int]:
    dims: list[int] = []
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
) -> tuple[str, int, list[int]]:
    initializer_names = {init.name for init in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in initializer_names]
    if not real_inputs:
        raise RuntimeError("No non-initializer ONNX input tensors found.")
    model_input = real_inputs[0]
    input_name = model_input.name
    elem_type = model_input.type.tensor_type.elem_type
    shape = _value_info_shape(model_input)
    if not shape:
        shape = [1, 1, fallback_height, fallback_width]
    return input_name, elem_type, shape


def _convert_input_for_model(
    x_uint8: np.ndarray,
    input_shape: Sequence[int],
    elem_type: int,
) -> np.ndarray:
    x = x_uint8.reshape(input_shape)
    if elem_type == TensorProto.INT8:
        return (x.astype(np.int16) - 128).astype(np.int8)
    if elem_type == TensorProto.BOOL:
        return x.astype(np.uint8) > 127
    dtype = _onnx_type_to_numpy(elem_type)
    return x.astype(dtype)


def preprocess_image_uint8(
    image_path: Path,
    height: int,
    width: int,
    *,
    model_type: str = "hybrid_follow",
) -> np.ndarray:
    transform = get_val_transforms(
        model_type=model_type,
        input_channels=1,
        image_size=(height, width),
    )
    with Image.open(image_path) as image:
        tensor, _ = transform(image, {})

    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"Expected {model_type} preprocessing to yield [1,H,W], got {tuple(tensor.shape)}"
        )

    staged = torch.round(torch.clamp(tensor, 0.0, 1.0) * 255.0).to(torch.uint8)
    return staged.cpu().numpy().reshape(-1)


def run_onnx_final_output(
    onnx_path: Path,
    input_values: np.ndarray,
    fallback_height: int,
    fallback_width: int,
) -> tuple[str, list[int], np.ndarray, list[int]]:
    model = onnx.load(str(onnx_path))
    _, elem_type, input_shape = _get_model_input_meta(model, fallback_height, fallback_width)
    expected_input_count = int(np.prod(input_shape))
    if input_values.size != expected_input_count:
        raise RuntimeError(
            f"Staged input size mismatch for {onnx_path}: "
            f"expected {expected_input_count}, got {input_values.size}"
        )

    x_feed = _convert_input_for_model(input_values, input_shape, elem_type)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    output_name = session.get_outputs()[0].name
    session_input_name = session.get_inputs()[0].name
    output_value = np.asarray(session.run([output_name], {session_input_name: x_feed})[0])
    flat_output = output_value.reshape(-1)
    if np.issubdtype(flat_output.dtype, np.floating):
        flat_output = np.rint(flat_output).astype(np.int64)
    else:
        flat_output = flat_output.astype(np.int64)
    return output_name, input_shape, flat_output, list(output_value.shape)


def stage_image_artifacts(
    image_path: Path,
    onnx_path: Path,
    output_dir: Path,
    app_dir: Path | None = None,
    fallback_height: int = 128,
    fallback_width: int = 128,
    model_type: str = "hybrid_follow",
    expected_output: Sequence[int] | np.ndarray | None = None,
    expected_output_name: str | None = None,
    expected_source: str | None = None,
) -> StageArtifacts:
    image_path = image_path.expanduser().resolve()
    onnx_path = onnx_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    app_dir = app_dir.expanduser().resolve() if app_dir else None

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"DORY ONNX not found: {onnx_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    _, _, input_shape = _get_model_input_meta(model, fallback_height, fallback_width)
    if len(input_shape) != 4 or input_shape[1] != 1:
        raise RuntimeError(
            f"Expected {model_type} ONNX input shape [1,1,H,W], got {input_shape}"
        )
    height = int(input_shape[2])
    width = int(input_shape[3])

    staged_input = preprocess_image_uint8(
        image_path=image_path,
        height=height,
        width=width,
        model_type=model_type,
    )
    if expected_output is None:
        output_name, input_shape, golden_output, output_shape = run_onnx_final_output(
            onnx_path=onnx_path,
            input_values=staged_input,
            fallback_height=height,
            fallback_width=width,
        )
        active_expected_source = expected_source or f"onnxruntime({onnx_path.name})"
    else:
        input_shape = list(input_shape)
        golden_output = np.asarray(expected_output, dtype=np.int64).reshape(-1)
        output_shape = [int(golden_output.size)]
        output_name = str(expected_output_name or "final")
        active_expected_source = expected_source or "external_expected_output"

    input_txt_path = output_dir / "input.txt"
    input_hex_path = output_dir / "inputs.hex"
    output_txt_path = output_dir / "output.txt"
    metadata_path = output_dir / "metadata.json"

    _write_values_txt(
        input_txt_path,
        staged_input,
        header=f"input (shape {input_shape}) from {image_path.name}",
    )
    staged_input.astype(np.uint8).tofile(input_hex_path)
    _write_values_txt(
        output_txt_path,
        golden_output,
        header=f"{output_name} (shape {output_shape}) from {image_path.name}",
    )

    app_input_hex = None
    if app_dir is not None:
        app_hex_dir = app_dir / "hex"
        if not app_hex_dir.is_dir():
            raise FileNotFoundError(f"Application hex directory not found: {app_hex_dir}")
        app_input_hex_path = app_hex_dir / "inputs.hex"
        shutil.copyfile(input_hex_path, app_input_hex_path)
        app_input_hex = str(app_input_hex_path)

    artifacts = StageArtifacts(
        image_path=str(image_path),
        output_dir=str(output_dir),
        input_txt=str(input_txt_path),
        input_hex=str(input_hex_path),
        output_txt=str(output_txt_path),
        metadata_json=str(metadata_path),
        output_name=output_name,
        input_shape=list(input_shape),
        output_shape=output_shape,
        expected_tensor=golden_output.astype(np.int64).tolist(),
        preprocess=PREPROCESS_DESCRIPTION,
        app_input_hex=app_input_hex,
        expected_source=active_expected_source,
    )

    metadata = artifacts.to_dict()
    metadata["onnx_path"] = str(onnx_path)
    metadata["model_type"] = str(model_type)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a real image into hybrid_follow input/output artifacts."
    )
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument(
        "--onnx",
        default="export/hybrid_follow/hybrid_follow_dory.onnx",
        help="Path to the DORY ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where input.txt, inputs.hex, output.txt, and metadata.json are written.",
    )
    parser.add_argument(
        "--app-dir",
        default=None,
        help="Optional generated app dir. If set, copy inputs.hex into <app-dir>/hex/inputs.hex too.",
    )
    parser.add_argument("--height", type=int, default=128, help="Expected input height.")
    parser.add_argument("--width", type=int, default=128, help="Expected input width.")
    parser.add_argument(
        "--model-type",
        default="hybrid_follow",
        help="Model type passed to the validation preprocessing transform.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = stage_image_artifacts(
        image_path=Path(args.image),
        onnx_path=Path(args.onnx),
        output_dir=Path(args.output_dir),
        app_dir=Path(args.app_dir) if args.app_dir else None,
        fallback_height=args.height,
        fallback_width=args.width,
        model_type=str(args.model_type),
    )
    print(json.dumps(artifacts.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

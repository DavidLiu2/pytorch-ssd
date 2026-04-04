#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

from export_nemo_quant import (  # noqa: E402
    HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
    ExportRequest,
    apply_hybrid_follow_export_preset_config,
    build_model,
    collect_calib_samples,
    integer_add_scale_selection_scope,
    integerize_deploy_conv_biases,
    iter_calib_batches,
    load_checkpoint,
    load_checkpoint_payload,
    maybe_convert_hybrid_follow_to_export_head,
    maybe_fuse_hybrid_follow_for_export,
    normalize_hybrid_follow_export_preset,
    normalize_integer_requant_tensors,
    patch_model_to_graph_compat,
    repair_hybrid_follow_fused_quant_graph,
    derive_hybrid_follow_export_preset_config,
    hybrid_follow_model_kwargs_from_checkpoint_payload,
)
from hybrid_follow_image_artifacts import (  # noqa: E402
    PREPROCESS_DESCRIPTION,
    _convert_input_for_model,
    _get_model_input_meta,
)
from utils.transforms import get_val_transforms  # noqa: E402


DEFAULT_IMAGE = PROJECT_DIR / "training" / "hybrid_follow" / "eval_epoch_015" / "top_fn" / "01_p0.0114_000000132408.jpg"
DEFAULT_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_follow_score.pth"
DEFAULT_ONNX = PROJECT_DIR / "export" / "hybrid_follow" / "hybrid_follow_dory.onnx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "export" / "hybrid_follow" / "stage_drift_report"
DEFAULT_CALIB_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"

STAGE_ORDER = ["fp", "fq", "id", "onnx", "golden", "gvsoc"]
PAIRWISE_SPECS = [
    ("fp", "fq", "fp_to_fq"),
    ("fq", "id", "fq_to_id"),
    ("id", "onnx", "id_to_onnx"),
    ("onnx", "golden", "onnx_to_golden"),
    ("golden", "gvsoc", "golden_to_gvsoc"),
    ("fp", "onnx", "fp_to_onnx"),
    ("fp", "gvsoc", "fp_to_gvsoc"),
    ("pytorch", "nemo", "pytorch_vs_quantized"),
    ("nemo", "onnx", "quantized_vs_onnx"),
    ("pytorch", "onnx", "pytorch_vs_onnx"),
    ("onnx", "golden", "onnx_vs_golden"),
    ("golden", "gvsoc", "golden_vs_gvsoc"),
    ("pytorch", "gvsoc", "pytorch_vs_gvsoc"),
]


def build_hybrid_follow_model_for_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
):
    payload = load_checkpoint_payload(str(ckpt_path), device)
    kwargs = hybrid_follow_model_kwargs_from_checkpoint_payload(payload)
    return build_model(
        "hybrid_follow",
        args.num_classes,
        args.width_mult,
        (args.height, args.width),
        args.input_channels,
        stage4_variant=kwargs.get("stage4_variant"),
        stage4_1_ablation=kwargs.get("stage4_variant"),
        stage_channels=kwargs.get("stage_channels"),
    )


@dataclass
class Thresholds:
    x_abs_diff: float
    size_abs_diff: float
    vis_conf_abs_diff: float


@dataclass
class StageResult:
    key: str
    label: str
    status: str
    source: str | None = None
    stage_tag: str | None = None
    representation: str | None = None
    input_mode: str | None = None
    raw_native: list[int | float] | None = None
    raw_semantic: list[float] | None = None
    decoded: dict[str, float] | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.raw_native is not None:
            payload["raw_native"] = [_to_python_number(value) for value in self.raw_native]
        if self.raw_semantic is not None:
            payload["raw_semantic"] = [float(value) for value in self.raw_semantic]
        if self.decoded is not None:
            payload["decoded"] = {key: float(value) for key, value in self.decoded.items()}
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hybrid_follow outputs across PyTorch, optional in-memory NEMO, "
            "exported ONNX, golden artifacts, and optional GVSOC output."
        )
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Checkpoint path.")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Exported ONNX path.")
    parser.add_argument("--golden", default=None, help="Required golden output artifact path.")
    parser.add_argument("--gvsoc-json", default=None, help="Required gvsoc_final_tensor.json path.")
    parser.add_argument("--gvsoc-log", default=None, help="Required GVSOC run log path with LAYER_BYTES markers.")
    parser.add_argument("--layer-manifest", default=None, help="Required combined GAP8/DORY layer manifest JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Artifact output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument(
        "--nemo-stage",
        choices=["auto", "skip", "fq", "qd", "id"],
        default="auto",
        help="Legacy compatibility selector for the `nemo` alias. Explicit `fq` and `id` are always collected.",
    )
    parser.add_argument(
        "--hybrid-follow-export-preset",
        default="baseline",
        choices=["baseline", "microblock_add_only"],
        help="Hybrid-follow preset mirrored for explicit FQ/ID stage collection.",
    )
    parser.add_argument(
        "--nemo-calib-dir",
        default=str(DEFAULT_CALIB_DIR) if DEFAULT_CALIB_DIR.is_dir() else None,
        help="Calibration image directory for the optional NEMO stage.",
    )
    parser.add_argument("--nemo-calib-tensor", default=None, help="Optional calibration tensor path.")
    parser.add_argument("--nemo-calib-batches", type=int, default=8)
    parser.add_argument("--nemo-calib-seed", type=int, default=0)
    parser.add_argument("--nemo-bits", type=int, default=8)
    parser.add_argument("--nemo-eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument(
        "--onnx-stage",
        choices=["auto", "fp", "fq", "qd", "id"],
        default="auto",
        help="Interpretation hint for the provided ONNX export.",
    )
    parser.add_argument(
        "--integer-output-scale",
        type=float,
        default=32768.0,
        help="Fixed-point scale for deployment outputs.",
    )
    parser.add_argument("--warn-x-abs-diff", type=float, default=0.05)
    parser.add_argument("--warn-size-abs-diff", type=float, default=0.05)
    parser.add_argument("--warn-vis-conf-abs-diff", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
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


def _to_python_number(value: Any) -> int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_values_txt(path: Path, values: list[int], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header},\n")
        for value in values:
            f.write(f"{int(value)},\n")


def parse_numeric_artifact(path: Path) -> list[int]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "values" in payload:
            payload = payload["values"]
        return [int(value) for value in payload]

    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    return values


LAYER_BEGIN_RE = re.compile(
    r"^LAYER_BYTES_BEGIN\s+(\d+)\s+(\S+)\s+bytes=(\d+)\s+sum_mod32=(\d+)\s+hash32=(\d+)$"
)
LAYER_LINE_RE = re.compile(r"^LAYER_BYTES\s+(\d+)\s+(\S+)\s+offset=(\d+)(.*)$")
LAYER_END_RE = re.compile(r"^LAYER_BYTES_END\s+(\d+)\s+(\S+)$")


def parse_gvsoc_layer_bytes(log_path: Path) -> dict[int, dict[str, Any]]:
    layers: dict[int, dict[str, Any]] = {}
    current_index: int | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        begin_match = LAYER_BEGIN_RE.match(line)
        if begin_match:
            current_index = int(begin_match.group(1))
            layers[current_index] = {
                "index": current_index,
                "layer_name": begin_match.group(2),
                "byte_count": int(begin_match.group(3)),
                "sum_mod32": int(begin_match.group(4)),
                "hash32": int(begin_match.group(5)),
                "raw_bytes": [],
            }
            continue

        end_match = LAYER_END_RE.match(line)
        if end_match:
            current_index = None
            continue

        line_match = LAYER_LINE_RE.match(line)
        if line_match:
            index = int(line_match.group(1))
            payload = line_match.group(4).strip()
            if index not in layers:
                raise RuntimeError(f"Found LAYER_BYTES payload for missing layer index {index} in {log_path}")
            if payload:
                layers[index]["raw_bytes"].extend(int(token) for token in payload.split())

    for index, payload in layers.items():
        expected = int(payload["byte_count"])
        actual = len(payload["raw_bytes"])
        if actual != expected:
            raise RuntimeError(
                f"GVSOC layer dump size mismatch for layer {index}: expected {expected}, got {actual}"
            )
    return layers


def decode_runtime_layer_values(raw_bytes: list[int], element_width_bytes: int) -> np.ndarray:
    data = bytes(int(value) & 0xFF for value in raw_bytes)
    if element_width_bytes == 1:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if element_width_bytes == 4:
        return np.frombuffer(data, dtype="<i4").astype(np.int64)
    if element_width_bytes == 2:
        return np.frombuffer(data, dtype="<i2").astype(np.int64)
    raise ValueError(f"Unsupported runtime element width: {element_width_bytes}")


def compare_runtime_layers(
    *,
    gvsoc_log_path: Path,
    layer_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(layer_manifest_path.read_text(encoding="utf-8"))
    gvsoc_layers = parse_gvsoc_layer_bytes(gvsoc_log_path)
    reports = []
    first_divergent = None

    for layer in manifest.get("layers") or []:
        index = int(layer["index"])
        gvsoc_payload = gvsoc_layers.get(index)
        if gvsoc_payload is None:
            reports.append(
                {
                    "index": index,
                    "layer_name": layer.get("layer_name"),
                    "status": "missing",
                    "reason": "No GVSOC layer dump found for this layer index.",
                }
            )
            continue

        golden_values = np.asarray(parse_numeric_artifact(Path(layer["golden_path"])), dtype=np.int64).reshape(-1)
        runtime_values = decode_runtime_layer_values(
            gvsoc_payload["raw_bytes"],
            int(layer["element_width_bytes"]),
        ).reshape(-1)
        if runtime_values.shape != golden_values.shape:
            status = "shape_mismatch"
            diff_report = {
                "expected_shape": [int(golden_values.size)],
                "actual_shape": [int(runtime_values.size)],
            }
        else:
            diff = np.abs(runtime_values.astype(np.float64) - golden_values.astype(np.float64))
            cosine = None
            runtime_norm = float(np.linalg.norm(runtime_values.astype(np.float64)))
            golden_norm = float(np.linalg.norm(golden_values.astype(np.float64)))
            if runtime_norm > 0.0 and golden_norm > 0.0:
                cosine = float(
                    np.dot(runtime_values.astype(np.float64), golden_values.astype(np.float64))
                    / (runtime_norm * golden_norm)
                )
            diff_report = {
                "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                "cosine_similarity": cosine,
                "exact_match": bool(np.array_equal(runtime_values, golden_values)),
            }
            status = "ok" if diff_report["exact_match"] else "warn"
            if first_divergent is None and status == "warn":
                first_divergent = {
                    "index": index,
                    "layer_name": layer.get("layer_name"),
                    "tensor_name": layer.get("tensor_name"),
                    **diff_report,
                }

        layer_report = {
            "index": index,
            "layer_name": layer.get("layer_name"),
            "tensor_name": layer.get("tensor_name"),
            "status": status,
            "element_count": int(layer["element_count"]),
            "element_width_bytes": int(layer["element_width_bytes"]),
            "golden_path": layer.get("golden_path"),
            "gvsoc_byte_count": int(gvsoc_payload["byte_count"]),
            "gvsoc_sum_mod32": int(gvsoc_payload["sum_mod32"]),
            "gvsoc_hash32": int(gvsoc_payload["hash32"]),
            "compare": diff_report,
        }
        reports.append(layer_report)

        layer_dir = output_dir / "runtime_layers" / f"{index:02d}_{layer.get('layer_name')}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        write_json(layer_dir / "report.json", layer_report)
        write_json(
            layer_dir / "gvsoc_values.json",
            {
                "index": index,
                "layer_name": layer.get("layer_name"),
                "values": runtime_values.tolist(),
            },
        )

    summary = {
        "gvsoc_log": str(gvsoc_log_path),
        "layer_manifest": str(layer_manifest_path),
        "layer_count": len(reports),
        "first_divergent_layer": first_divergent,
        "layers": reports,
    }
    write_json(output_dir / "runtime_layer_compare.json", summary)
    lines = [
        "# Runtime Layer Compare",
        "",
        f"- GVSOC log: `{gvsoc_log_path}`",
        f"- Layer manifest: `{layer_manifest_path}`",
        "",
    ]
    if first_divergent is None:
        lines.append("- First divergent layer: none")
    else:
        lines.append(
            "- First divergent layer: `{} {}` mean_abs_diff=`{:.6f}` max_abs_diff=`{:.6f}`".format(
                first_divergent["index"],
                first_divergent["layer_name"],
                float(first_divergent["mean_abs_diff"]),
                float(first_divergent["max_abs_diff"]),
            )
        )
    lines.extend(["", "| idx | layer | status | mean_abs | max_abs |", "| --- | --- | --- | ---: | ---: |"])
    for report in reports:
        compare = report.get("compare") or {}
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                report["index"],
                report.get("layer_name"),
                report["status"],
                "{:.6f}".format(float(compare.get("mean_abs_diff"))) if compare.get("mean_abs_diff") is not None else "n/a",
                "{:.6f}".format(float(compare.get("max_abs_diff"))) if compare.get("max_abs_diff") is not None else "n/a",
            )
        )
    (output_dir / "runtime_layer_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def decode_semantic_output(raw_semantic: list[float]) -> dict[str, float]:
    if len(raw_semantic) != 3:
        raise ValueError(f"Expected 3 semantic outputs, got {len(raw_semantic)}")
    return {
        "x_offset": float(raw_semantic[0]),
        "size_proxy": float(raw_semantic[1]),
        "visibility_logit": float(raw_semantic[2]),
        "visibility_confidence": float(sigmoid(float(raw_semantic[2]))),
    }


def to_stage_result(
    *,
    key: str,
    label: str,
    status: str,
    source: str | None = None,
    stage_tag: str | None = None,
    representation: str | None = None,
    input_mode: str | None = None,
    raw_native: np.ndarray | list[int] | list[float] | None = None,
    integer_output_scale: float | None = None,
    notes: list[str] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StageResult:
    raw_native_list: list[int | float] | None = None
    raw_semantic: list[float] | None = None
    decoded: dict[str, float] | None = None

    if raw_native is not None:
        raw_array = np.asarray(raw_native).reshape(-1)
        if raw_array.size != 3:
            raise ValueError(f"{label} output must reshape to 3 values, got shape {tuple(raw_array.shape)}")

        if representation == "fixed-point-int32":
            if integer_output_scale is None:
                raise ValueError("integer_output_scale is required for fixed-point outputs")
            raw_int = np.rint(raw_array).astype(np.int64)
            raw_native_list = [int(value) for value in raw_int.tolist()]
            raw_semantic = [float(value) / float(integer_output_scale) for value in raw_native_list]
        else:
            raw_float = raw_array.astype(np.float64)
            raw_native_list = [float(value) for value in raw_float.tolist()]
            raw_semantic = [float(value) for value in raw_native_list]

        decoded = decode_semantic_output(raw_semantic)

    return StageResult(
        key=key,
        label=label,
        status=status,
        source=str(source) if source else None,
        stage_tag=stage_tag,
        representation=representation,
        input_mode=input_mode,
        raw_native=raw_native_list,
        raw_semantic=raw_semantic,
        decoded=decoded,
        notes=notes or [],
        error=error,
        metadata=metadata or {},
    )


def preprocess_image_once(image_path: Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    transform = get_val_transforms(
        model_type="hybrid_follow",
        input_channels=1,
        image_size=(height, width),
    )
    with Image.open(image_path) as image:
        tensor, _ = transform(image, {})

    if tensor.ndim != 3 or tuple(tensor.shape) != (1, height, width):
        raise RuntimeError(
            f"Expected hybrid_follow preprocessing to produce [1,{height},{width}], got {tuple(tensor.shape)}"
        )

    x_float = tensor.unsqueeze(0).contiguous().to(dtype=torch.float32)
    x_uint8 = torch.round(torch.clamp(x_float, 0.0, 1.0) * 255.0).to(torch.uint8)
    return x_float, x_uint8


def save_preprocessed_artifacts(output_dir: Path, x_float: torch.Tensor, x_uint8: torch.Tensor) -> dict[str, Any]:
    preview_path = output_dir / "preprocessed_input_preview.png"
    float_dump_path = output_dir / "preprocessed_tensor_float.npy"
    uint8_dump_path = output_dir / "preprocessed_tensor_uint8.npy"
    metadata_path = output_dir / "preprocessed_input.json"

    x_float_np = x_float.detach().cpu().numpy()
    x_uint8_np = x_uint8.detach().cpu().numpy()

    Image.fromarray(x_uint8_np[0, 0], mode="L").save(preview_path)
    np.save(float_dump_path, x_float_np)
    np.save(uint8_dump_path, x_uint8_np)

    metadata = {
        "preprocess": PREPROCESS_DESCRIPTION,
        "float_shape": list(x_float_np.shape),
        "uint8_shape": list(x_uint8_np.shape),
        "float_min": float(x_float_np.min()),
        "float_max": float(x_float_np.max()),
        "uint8_min": int(x_uint8_np.min()),
        "uint8_max": int(x_uint8_np.max()),
        "preview_path": str(preview_path),
        "float_dump_path": str(float_dump_path),
        "uint8_dump_path": str(uint8_dump_path),
        "metadata_path": str(metadata_path),
    }
    write_json(metadata_path, metadata)
    return metadata


def infer_onnx_stage(onnx_path: Path, requested_stage: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    if requested_stage != "auto":
        return requested_stage, notes

    candidates = [
        onnx_path.with_name(f"{onnx_path.stem}_final_stage.txt"),
        onnx_path.parent / "hybrid_follow_final_stage.txt",
        onnx_path.parent / "final_stage.txt",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        stage_text = candidate.read_text(encoding="utf-8", errors="replace").strip().lower()
        if stage_text in {"fp", "fq", "qd", "id"}:
            notes.append(f"Resolved ONNX stage from {candidate.name}: {stage_text}")
            return stage_text, notes

    lower_name = onnx_path.name.lower()
    if "fp" in lower_name and "quant" not in lower_name:
        notes.append("Resolved ONNX stage heuristically from filename: fp")
        return "fp", notes
    if "fq" in lower_name:
        notes.append("Resolved ONNX stage heuristically from filename: fq")
        return "fq", notes
    if any(token in lower_name for token in ("qd", "id", "quant", "dory", "nomin", "noaffine", "notranspose")):
        notes.append("Resolved ONNX stage heuristically from filename: id")
        return "id", notes

    notes.append("Falling back to ONNX stage=fp because no explicit stage marker was found.")
    return "fp", notes


def onnx_stage_modes(stage_tag: str) -> tuple[str, str]:
    if stage_tag in {"qd", "id"}:
        return "staged_0_255", "fixed-point-int32"
    return "float_0_1", "float"


def run_pytorch_stage(ckpt_path: Path, x_float: torch.Tensor, args: argparse.Namespace) -> StageResult:
    model = build_hybrid_follow_model_for_checkpoint(ckpt_path, args, torch.device("cpu"))
    model = load_checkpoint(model, str(ckpt_path), torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        output = model(x_float)
    return to_stage_result(
        key="fp",
        label="PyTorch checkpoint",
        status="ok",
        source=str(ckpt_path),
        stage_tag="fp",
        representation="float",
        input_mode="float_0_1",
        raw_native=output.detach().cpu().numpy(),
        notes=["Original checkpoint path before NEMO export wrapping."],
        metadata={"shape": list(output.shape), "dtype": str(output.dtype)},
    )


def build_nemo_calib_args(args: argparse.Namespace) -> SimpleNamespace:
    calib_dir = resolve_repo_path(args.nemo_calib_dir) if args.nemo_calib_dir else None
    calib_tensor = resolve_repo_path(args.nemo_calib_tensor) if args.nemo_calib_tensor else None
    return SimpleNamespace(
        model_type="hybrid_follow",
        calib_dir=str(calib_dir) if calib_dir else None,
        calib_tensor=str(calib_tensor) if calib_tensor else None,
        calib_batches=args.nemo_calib_batches,
        calib_seed=args.nemo_calib_seed,
        mean=None,
        std=None,
        input_channels=args.input_channels,
        eps_in=args.nemo_eps_in,
    )


def build_nemo_stage_results(
    ckpt_path: Path,
    x_float: torch.Tensor,
    x_uint8: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, StageResult]:
    notes = []
    if args.nemo_stage == "auto":
        notes.append("Auto mode chose FQ for the legacy `nemo` alias. Explicit FQ/ID stages are collected together.")

    patch_model_to_graph_compat()
    import nemo

    model_fp = build_hybrid_follow_model_for_checkpoint(ckpt_path, args, torch.device("cpu"))
    model_fp = load_checkpoint(model_fp, str(ckpt_path), torch.device("cpu"))
    model_fp = maybe_fuse_hybrid_follow_for_export(model_fp)
    model_fp = maybe_convert_hybrid_follow_to_export_head(model_fp)
    model_fp.eval()

    dummy_input = torch.randn(1, args.input_channels, args.height, args.width, dtype=torch.float32)
    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
    model_q.eval()
    repair_hybrid_follow_fused_quant_graph(model_q)
    model_q.change_precision(bits=args.nemo_bits, scale_weights=True, scale_activations=True)

    calib_args = build_nemo_calib_args(args)
    calib_samples = collect_calib_samples(calib_args, (args.height, args.width), torch.device("cpu"))
    with torch.no_grad():
        with model_q.statistics_act():
            for sample in calib_samples:
                _ = model_q(sample["tensor"])
    model_q.reset_alpha_act()

    preset_name = normalize_hybrid_follow_export_preset(args.hybrid_follow_export_preset)
    preset_config = derive_hybrid_follow_export_preset_config(
        model_q,
        calib_samples,
        preset_name,
    )
    preset_report = apply_hybrid_follow_export_preset_config(model_q, preset_config)

    results: dict[str, StageResult] = {}
    with torch.no_grad():
        fq_output = model_q(x_float)

    eps_out = getattr(model_q, "eps_out", None)
    if torch.is_tensor(eps_out):
        eps_out = eps_out.detach().cpu().numpy().tolist()

    shared_metadata = {
        "bits": args.nemo_bits,
        "eps_in": args.nemo_eps_in,
        "calib_dir": calib_args.calib_dir,
        "calib_tensor": calib_args.calib_tensor,
        "calib_batches": calib_args.calib_batches,
        "hybrid_follow_export_preset": preset_name,
        "preset_report": preset_report,
    }
    results["fq"] = to_stage_result(
        key="fq",
        label="NEMO FQ (in-memory)",
        status="ok",
        source=str(ckpt_path),
        stage_tag="fq",
        representation="float",
        input_mode="float_0_1",
        raw_native=fq_output.detach().cpu().numpy(),
        integer_output_scale=args.integer_output_scale,
        notes=notes + ["Built from the shared calibrated preset snapshot used for explicit FQ/ID collection."],
        metadata={
            **shared_metadata,
            "shape": list(fq_output.shape),
            "dtype": str(fq_output.dtype),
            "eps_out": eps_out,
        },
    )

    with integer_add_scale_selection_scope(
        HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY,
        (preset_report or {}).get("integer_add_operator_overrides"),
    ):
        try:
            model_q.reset_alpha_weights()
        except Exception:
            notes.append("reset_alpha_weights() raised, so weight alpha reset was skipped.")

        model_q.qd_stage(eps_in=args.nemo_eps_in)
        repair_hybrid_follow_fused_quant_graph(model_q)
        with torch.no_grad():
            qd_output = model_q(x_uint8.to(dtype=torch.float32))

        qd_eps_out = getattr(model_q, "eps_out", None)
        if torch.is_tensor(qd_eps_out):
            qd_eps_out = qd_eps_out.detach().cpu().numpy().tolist()

        results["qd"] = to_stage_result(
            key="qd",
            label="NEMO QD (in-memory)",
            status="ok",
            source=str(ckpt_path),
            stage_tag="qd",
            representation="fixed-point-int32",
            input_mode="staged_0_255",
            raw_native=qd_output.detach().cpu().numpy(),
            integer_output_scale=args.integer_output_scale,
            notes=notes + ["Built from the shared calibrated preset snapshot used for explicit FQ/ID collection."],
            metadata={
                **shared_metadata,
                "shape": list(qd_output.shape),
                "dtype": str(qd_output.dtype),
                "eps_out": qd_eps_out,
            },
        )

        model_q.id_stage()
        normalize_integer_requant_tensors(model_q)
        integerize_deploy_conv_biases(model_q)
        with torch.no_grad():
            id_output = model_q(x_uint8.to(dtype=torch.float32))

    id_eps_out = getattr(model_q, "eps_out", None)
    if torch.is_tensor(id_eps_out):
        id_eps_out = id_eps_out.detach().cpu().numpy().tolist()

    results["id"] = to_stage_result(
        key="id",
        label="NEMO ID (in-memory)",
        status="ok",
        source=str(ckpt_path),
        stage_tag="id",
        representation="fixed-point-int32",
        input_mode="staged_0_255",
        raw_native=id_output.detach().cpu().numpy(),
        integer_output_scale=args.integer_output_scale,
        notes=notes + ["Built from the shared calibrated preset snapshot used for explicit FQ/ID collection."],
        metadata={
            **shared_metadata,
            "shape": list(id_output.shape),
            "dtype": str(id_output.dtype),
            "eps_out": id_eps_out,
        },
    )
    return results


def run_nemo_stage(
    ckpt_path: Path,
    x_float: torch.Tensor,
    x_uint8: torch.Tensor,
    args: argparse.Namespace,
    requested_stage: str | None = None,
) -> StageResult:
    requested_stage = requested_stage or ("fq" if args.nemo_stage == "auto" else args.nemo_stage)
    if requested_stage == "skip":
        return to_stage_result(
            key=requested_stage,
            label="NEMO quantized",
            status="skipped",
            stage_tag="skip",
            notes=["Skipped because --nemo-stage=skip was requested."],
        )
    results = build_nemo_stage_results(ckpt_path, x_float, x_uint8, args)
    return results[requested_stage]


def run_onnx_stage(
    onnx_path: Path,
    x_float: torch.Tensor,
    x_uint8: torch.Tensor,
    args: argparse.Namespace,
) -> StageResult:
    stage_tag, notes = infer_onnx_stage(onnx_path, args.onnx_stage)
    input_mode, representation = onnx_stage_modes(stage_tag)
    model = onnx.load(str(onnx_path))
    input_name, elem_type, input_shape = _get_model_input_meta(model, args.height, args.width)

    if input_mode == "staged_0_255":
        feed_values = x_uint8.detach().cpu().numpy().reshape(-1)
    else:
        feed_values = x_float.detach().cpu().numpy().reshape(-1)
    x_feed = _convert_input_for_model(feed_values, input_shape, elem_type)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if len(output_names) != 1:
        raise RuntimeError(f"Expected one output tensor, found {len(output_names)}: {output_names}")
    output_name = output_names[0]
    output_value = np.asarray(session.run([output_name], {session.get_inputs()[0].name: x_feed})[0])

    return to_stage_result(
        key="onnx",
        label=f"Exported ONNX ({stage_tag.upper()})",
        status="ok",
        source=str(onnx_path),
        stage_tag=stage_tag,
        representation=representation,
        input_mode=input_mode,
        raw_native=output_value,
        integer_output_scale=args.integer_output_scale,
        notes=notes,
        metadata={
            "input_name": input_name,
            "input_shape": list(input_shape),
            "input_dtype": session.get_inputs()[0].type,
            "output_name": output_name,
            "output_shape": list(output_value.shape),
            "output_dtype": str(output_value.dtype),
        },
    )


def save_generated_golden_artifact(output_dir: Path, stage: StageResult) -> Path:
    if stage.raw_native is None:
        raise ValueError("Cannot generate a golden artifact without ONNX raw output values.")
    path = output_dir / "generated_golden_output.txt"
    values = [int(value) for value in stage.raw_native]
    write_values_txt(path, values, "generated golden output (shape [1, 3])")
    return path


def load_fixed_point_artifact_stage(
    *,
    key: str,
    label: str,
    source_path: Path,
    args: argparse.Namespace,
    notes: list[str] | None = None,
) -> StageResult:
    values = parse_numeric_artifact(source_path)
    return to_stage_result(
        key=key,
        label=label,
        status="ok",
        source=str(source_path),
        stage_tag="id",
        representation="fixed-point-int32",
        input_mode="staged_0_255",
        raw_native=values,
        integer_output_scale=args.integer_output_scale,
        notes=notes,
        metadata={"decode_scale": args.integer_output_scale},
    )


def safe_run_stage(stage_key: str, runner) -> StageResult:
    try:
        return runner()
    except FileNotFoundError as exc:
        return to_stage_result(
            key=stage_key,
            label=stage_key.replace("_", " "),
            status="skipped",
            error=str(exc),
            notes=["Stage was skipped because its input artifact was not found."],
        )
    except Exception as exc:
        return to_stage_result(
            key=stage_key,
            label=stage_key.replace("_", " "),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            notes=[traceback.format_exc(limit=4)],
        )


def symmetric_relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-8)
    return abs(a - b) / denom


def compare_stage_pair(left: StageResult, right: StageResult, thresholds: Thresholds) -> dict[str, Any]:
    if left.raw_semantic is None or right.raw_semantic is None or left.decoded is None or right.decoded is None:
        raise ValueError("Cannot compare stages without semantic raw outputs and decoded values.")

    raw_names = ["x_offset", "size_proxy", "visibility_logit"]
    raw_abs = {}
    raw_rel = {}
    for index, name in enumerate(raw_names):
        raw_abs[name] = abs(float(left.raw_semantic[index]) - float(right.raw_semantic[index]))
        raw_rel[name] = symmetric_relative_diff(float(left.raw_semantic[index]), float(right.raw_semantic[index]))

    decoded_abs = {
        "x_offset": abs(left.decoded["x_offset"] - right.decoded["x_offset"]),
        "size_proxy": abs(left.decoded["size_proxy"] - right.decoded["size_proxy"]),
        "visibility_confidence": abs(
            left.decoded["visibility_confidence"] - right.decoded["visibility_confidence"]
        ),
    }
    decoded_rel = {
        key: symmetric_relative_diff(left.decoded[key], right.decoded[key])
        for key in decoded_abs
    }

    warnings = []
    if decoded_abs["x_offset"] > thresholds.x_abs_diff:
        warnings.append(f"WARNING: x_offset abs diff {decoded_abs['x_offset']:.6f} exceeds {thresholds.x_abs_diff:.6f}")
    if decoded_abs["size_proxy"] > thresholds.size_abs_diff:
        warnings.append(
            f"WARNING: size_proxy abs diff {decoded_abs['size_proxy']:.6f} exceeds {thresholds.size_abs_diff:.6f}"
        )
    if decoded_abs["visibility_confidence"] > thresholds.vis_conf_abs_diff:
        warnings.append(
            "WARNING: visibility_confidence abs diff "
            f"{decoded_abs['visibility_confidence']:.6f} exceeds {thresholds.vis_conf_abs_diff:.6f}"
        )

    return {
        "left_stage": left.key,
        "left_label": left.label,
        "right_stage": right.key,
        "right_label": right.label,
        "status": "warn" if warnings else "ok",
        "raw_head_abs_diff": raw_abs,
        "raw_head_relative_diff": raw_rel,
        "decoded_abs_diff": decoded_abs,
        "decoded_relative_diff": decoded_rel,
        "warnings": warnings,
    }


def build_pairwise_reports(stages: dict[str, StageResult], thresholds: Thresholds) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for left_key, right_key, report_key in PAIRWISE_SPECS:
        left = stages.get(left_key)
        right = stages.get(right_key)
        if left is None or right is None:
            reports[report_key] = {
                "left_stage": left_key,
                "right_stage": right_key,
                "status": "skipped",
                "reason": "One or both stages are unavailable.",
            }
            continue
        if left.status != "ok" or right.status != "ok":
            reports[report_key] = {
                "left_stage": left_key,
                "right_stage": right_key,
                "status": "skipped",
                "reason": "One or both stages did not complete successfully.",
            }
            continue
        reports[report_key] = compare_stage_pair(left, right, thresholds)
    return reports


def summarize_drift_onset(stages: dict[str, StageResult], pairwise_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available = [key for key in STAGE_ORDER if key in stages and stages[key].status == "ok"]
    for index in range(len(available) - 1):
        left_key = available[index]
        right_key = available[index + 1]
        for report in pairwise_reports.values():
            if report.get("left_stage") == left_key and report.get("right_stage") == right_key:
                if report.get("status") == "warn":
                    return {
                        "status": "warn",
                        "left_stage": left_key,
                        "right_stage": right_key,
                        "message": (
                            f"Configured drift thresholds first trip between {stages[left_key].label} "
                            f"and {stages[right_key].label}."
                        ),
                        "warnings": report.get("warnings", []),
                    }
    return {
        "status": "ok",
        "message": "No configured threshold breaches were found across the available adjacent stages.",
    }


def write_stage_json_artifacts(output_dir: Path, stages: dict[str, StageResult]) -> None:
    raw_dir = output_dir / "raw_outputs"
    decoded_dir = output_dir / "decoded_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    for key, stage in stages.items():
        write_json(raw_dir / f"{key}.json", stage.to_dict())
        write_json(
            decoded_dir / f"{key}.json",
            {
                "key": stage.key,
                "label": stage.label,
                "status": stage.status,
                "decoded": stage.decoded,
                "representation": stage.representation,
                "stage_tag": stage.stage_tag,
                "source": stage.source,
                "notes": stage.notes,
                "error": stage.error,
            },
        )


def stage_status_line(stage: StageResult) -> str:
    if stage.status == "ok" and stage.decoded is not None:
        return (
            f"{stage.label}: x={stage.decoded['x_offset']:.6f}, "
            f"size={stage.decoded['size_proxy']:.6f}, "
            f"vis_logit={stage.decoded['visibility_logit']:.6f}, "
            f"vis_conf={stage.decoded['visibility_confidence']:.6f}"
        )
    if stage.error:
        return f"{stage.label}: {stage.status.upper()} ({stage.error})"
    return f"{stage.label}: {stage.status.upper()}"


def build_markdown_summary(
    *,
    image_path: Path,
    output_dir: Path,
    thresholds: Thresholds,
    stages: dict[str, StageResult],
    pairwise_reports: dict[str, dict[str, Any]],
    drift_onset: dict[str, Any],
    preprocess_artifacts: dict[str, Any],
    generated_golden_path: Path | None,
    integer_output_scale: float,
    runtime_layer_compare: dict[str, Any] | None,
) -> str:
    lines = [
        "# Hybrid Follow Stage Drift Summary",
        "",
        f"- Image: `{image_path}`",
        f"- Output dir: `{output_dir}`",
        f"- Preprocess: {PREPROCESS_DESCRIPTION}",
        f"- Integer decode scale: `{integer_output_scale}`",
        f"- Warning thresholds: `{json.dumps(asdict(thresholds))}`",
    ]
    if generated_golden_path is not None:
        lines.append(f"- Golden artifact: generated for this run at `{generated_golden_path}`")

    lines.extend(["", "## Drift Onset", "", f"- {drift_onset['message']}"])
    for warning in drift_onset.get("warnings", []):
        lines.append(f"- {warning}")

    lines.extend(["", "## Stage Outputs", ""])
    for stage_key in STAGE_ORDER:
        stage = stages.get(stage_key)
        if stage is not None:
            lines.append(f"- {stage_status_line(stage)}")

    lines.extend(["", "## Pairwise Diffs", ""])
    for report_key, report in pairwise_reports.items():
        if report.get("status") == "skipped":
            lines.append(f"- {report_key}: SKIPPED ({report.get('reason')})")
            continue
        lines.append(
            "- "
            f"{report_key}: status={report['status'].upper()} "
            f"x_abs={report['decoded_abs_diff']['x_offset']:.6f} "
            f"size_abs={report['decoded_abs_diff']['size_proxy']:.6f} "
            f"vis_conf_abs={report['decoded_abs_diff']['visibility_confidence']:.6f}"
        )
        for warning in report.get("warnings", []):
            lines.append(f"- {warning}")

    if runtime_layer_compare is not None:
        first_divergent = runtime_layer_compare.get("first_divergent_layer")
        lines.extend(["", "## Runtime Layers", ""])
        if first_divergent is None:
            lines.append("- First divergent GVSOC layer: none")
        else:
            lines.append(
                "- First divergent GVSOC layer: `{} {}` mean_abs=`{:.6f}` max_abs=`{:.6f}`".format(
                    first_divergent["index"],
                    first_divergent["layer_name"],
                    float(first_divergent["mean_abs_diff"]),
                    float(first_divergent["max_abs_diff"]),
                )
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Preview: `{preprocess_artifacts['preview_path']}`",
            f"- Float tensor dump: `{preprocess_artifacts['float_dump_path']}`",
            f"- Uint8 tensor dump: `{preprocess_artifacts['uint8_dump_path']}`",
            f"- Raw per-stage JSON: `{output_dir / 'raw_outputs'}`",
            f"- Decoded per-stage JSON: `{output_dir / 'decoded_outputs'}`",
            f"- Pairwise JSON: `{output_dir / 'pairwise_diff_report.json'}`",
        ]
    )
    if runtime_layer_compare is not None:
        lines.append(f"- Runtime layer compare: `{output_dir / 'runtime_layer_compare.json'}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_path = resolve_repo_path(args.image)
    ckpt_path = resolve_repo_path(args.ckpt)
    onnx_path = resolve_repo_path(args.onnx) if args.onnx else None
    golden_path = resolve_repo_path(args.golden) if args.golden else None
    gvsoc_path = resolve_repo_path(args.gvsoc_json) if args.gvsoc_json else None
    gvsoc_log_path = resolve_repo_path(args.gvsoc_log) if args.gvsoc_log else None
    layer_manifest_path = resolve_repo_path(args.layer_manifest) if args.layer_manifest else None
    output_dir = resolve_repo_path(args.output_dir)

    if image_path is None or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if output_dir is None:
        raise RuntimeError("Output directory could not be resolved.")
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    x_float, x_uint8 = preprocess_image_once(image_path, args.height, args.width)
    preprocess_artifacts = save_preprocessed_artifacts(output_dir, x_float, x_uint8)
    thresholds = Thresholds(args.warn_x_abs_diff, args.warn_size_abs_diff, args.warn_vis_conf_abs_diff)

    stages: dict[str, StageResult] = {}

    if ckpt_path is not None and ckpt_path.is_file():
        stages["fp"] = safe_run_stage("fp", lambda: run_pytorch_stage(ckpt_path, x_float, args))
        try:
            nemo_results = build_nemo_stage_results(ckpt_path, x_float, x_uint8, args)
        except Exception as exc:
            nemo_results = {
                stage_key: to_stage_result(
                    key=stage_key,
                    label=f"NEMO {stage_key.upper()}",
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    notes=[traceback.format_exc(limit=6)],
                )
                for stage_key in ("fq", "qd", "id")
            }
        stages["fq"] = nemo_results["fq"]
        stages["id"] = nemo_results["id"]
    else:
        error = f"Checkpoint file not found: {ckpt_path}"
        stages["fp"] = to_stage_result(
            key="fp",
            label="PyTorch checkpoint",
            status="skipped",
            source=str(ckpt_path) if ckpt_path else None,
            error=error,
            notes=["PyTorch stage skipped because the checkpoint file was not found."],
        )
        stages["fq"] = to_stage_result(
            key="fq",
            label="NEMO FQ",
            status="skipped",
            source=str(ckpt_path) if ckpt_path else None,
            error=error,
            notes=["NEMO FQ stage skipped because the checkpoint file was not found."],
        )
        stages["id"] = to_stage_result(
            key="id",
            label="NEMO ID",
            status="skipped",
            source=str(ckpt_path) if ckpt_path else None,
            error=error,
            notes=["NEMO ID stage skipped because the checkpoint file was not found."],
        )

    if onnx_path is not None and onnx_path.is_file():
        stages["onnx"] = safe_run_stage("onnx", lambda: run_onnx_stage(onnx_path, x_float, x_uint8, args))
    else:
        stages["onnx"] = to_stage_result(
            key="onnx",
            label="Exported ONNX",
            status="skipped",
            source=str(onnx_path) if onnx_path else None,
            error=f"ONNX file not found: {onnx_path}",
            notes=["ONNX stage skipped because the model file was not found."],
        )

    generated_golden_path: Path | None = None
    if golden_path is not None and golden_path.is_file():
        stages["golden"] = safe_run_stage(
            "golden",
            lambda: load_fixed_point_artifact_stage(
                key="golden",
                label="Golden output artifact",
                source_path=golden_path,
                args=args,
                notes=["Loaded from the export-side text/json artifact."],
            ),
        )
        if generated_golden_path is not None and stages["golden"].status == "ok":
            stages["golden"].notes.append("This golden artifact was generated from the ONNX result during this run.")
    else:
        stages["golden"] = to_stage_result(
            key="golden",
            label="Golden output artifact",
            status="skipped",
            source=str(golden_path) if golden_path else None,
            error=f"Golden artifact not found: {golden_path}" if golden_path else None,
            notes=["Golden stage skipped because no compatible artifact path was provided."],
        )

    if gvsoc_path is not None and gvsoc_path.is_file():
        stages["gvsoc"] = safe_run_stage(
            "gvsoc",
            lambda: load_fixed_point_artifact_stage(
                key="gvsoc",
                label="GVSOC final tensor",
                source_path=gvsoc_path,
                args=args,
                notes=["Loaded from gvsoc_final_tensor.json or an equivalent fixed-point artifact."],
            ),
        )
    else:
        stages["gvsoc"] = to_stage_result(
            key="gvsoc",
            label="GVSOC final tensor",
            status="skipped",
            source=str(gvsoc_path) if gvsoc_path else None,
            error=f"GVSOC artifact not found: {gvsoc_path}" if gvsoc_path else None,
            notes=["GVSOC stage skipped because no gvsoc_final_tensor.json path was provided."],
        )

    legacy_nemo_stage = "fq" if args.nemo_stage in {"auto", "fq", "skip"} else args.nemo_stage
    stages["pytorch"] = stages["fp"]
    stages["nemo"] = nemo_results.get(legacy_nemo_stage, stages["fq"]) if ckpt_path is not None and ckpt_path.is_file() else stages["fq"]

    write_stage_json_artifacts(output_dir, stages)
    pairwise_reports = build_pairwise_reports(stages, thresholds)
    drift_onset = summarize_drift_onset(stages, pairwise_reports)

    runtime_layer_compare = None
    if (
        gvsoc_log_path is not None
        and gvsoc_log_path.is_file()
        and layer_manifest_path is not None
        and layer_manifest_path.is_file()
    ):
        runtime_layer_compare = compare_runtime_layers(
            gvsoc_log_path=gvsoc_log_path,
            layer_manifest_path=layer_manifest_path,
            output_dir=output_dir,
        )

    aggregate_report = {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "preprocess": PREPROCESS_DESCRIPTION,
        "integer_output_scale": args.integer_output_scale,
        "thresholds": asdict(thresholds),
        "preprocessed_input": preprocess_artifacts,
        "stages": {key: stage.to_dict() for key, stage in stages.items()},
        "pairwise": pairwise_reports,
        "drift_onset": drift_onset,
        "runtime_layer_compare": runtime_layer_compare,
    }
    required_stage_keys = ("fp", "fq", "id", "onnx", "golden", "gvsoc")
    required_stage_failures = [
        {
            "stage": stage_key,
            "status": stages[stage_key].status,
            "error": stages[stage_key].error,
        }
        for stage_key in required_stage_keys
        if stage_key not in stages or stages[stage_key].status != "ok"
    ]
    aggregate_report["required_stage_failures"] = required_stage_failures
    if runtime_layer_compare is None:
        aggregate_report["runtime_layer_compare_failure"] = (
            "Runtime layer compare requires --gvsoc-log and --layer-manifest."
        )
    write_json(output_dir / "stage_drift_report.json", aggregate_report)
    write_json(output_dir / "pairwise_diff_report.json", pairwise_reports)

    summary_text = build_markdown_summary(
        image_path=image_path,
        output_dir=output_dir,
        thresholds=thresholds,
        stages=stages,
        pairwise_reports=pairwise_reports,
        drift_onset=drift_onset,
        preprocess_artifacts=preprocess_artifacts,
        generated_golden_path=generated_golden_path,
        integer_output_scale=args.integer_output_scale,
        runtime_layer_compare=runtime_layer_compare,
    )
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    if required_stage_failures:
        raise RuntimeError(
            "Stage drift requires FP, FQ, ID, ONNX, golden, and GVSOC stages. "
            f"Failures: {required_stage_failures}"
        )
    if runtime_layer_compare is None:
        raise RuntimeError(
            "Stage drift requires runtime layer comparison. "
            "Provide both --gvsoc-log and --layer-manifest."
        )

    print(f"Image: {image_path}")
    print(f"Output dir: {output_dir}")
    for stage_key in STAGE_ORDER:
        if stage_key in stages:
            print(stage_status_line(stages[stage_key]))
    print(drift_onset["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

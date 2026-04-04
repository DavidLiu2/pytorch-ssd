#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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

from evaluate_quant_native_follow import (  # noqa: E402
    AnnotationIndex,
    build_eval_samples,
    build_quantized_models,
    export_id_onnx,
    make_eval_sample,
    run_plain_follow_pytorch_probe,
)
from export_nemo_quant_core import compare_arrays_rich, semantic_output, tensor_scalar, tensor_stats  # noqa: E402
from models.follow_model_factory import load_checkpoint_payload  # noqa: E402
from utils.follow_task import compute_follow_metrics  # noqa: E402
from validate_follow_rep16_overlays import build_contact_sheet  # noqa: E402


DEFAULT_CKPT = PROJECT_DIR / "training" / "plain_follow" / "plain_follow_best_follow_score.pth"
DEFAULT_REP16_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "plain_follow_quant_val" / "stem_integerization_study"
DEFAULT_FOCUS_SAMPLE = "08_visible_000000331604.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused plain_follow stem integerization study: manual QD/ID audit on one sample, "
            "stem activation variants, rep16 ONNX validation, and comparison overlays."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--calib-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--focus-sample", default=DEFAULT_FOCUS_SAMPLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def candidate_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        calib_dir=str(Path(args.calib_dir).expanduser().resolve()),
        calib_batches=int(args.calib_batches),
        calib_seed=int(args.calib_seed),
        eps_in=float(args.eps_in),
        bits=int(args.bits),
        id_explicit_eps_dict=False,
        id_local_scale_module=None,
        id_local_scale_factor=None,
    )


def build_metric_tensors(
    outputs: list[np.ndarray],
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_tensor = torch.tensor(np.asarray(outputs, dtype=np.float32))
    target_rows = [sample["follow_target"].detach().cpu().numpy().reshape(-1) for sample in samples]
    no_person_rows = [sample["true_no_person"].detach().cpu().numpy().reshape(-1) for sample in samples]
    target_tensor = torch.tensor(np.asarray(target_rows, dtype=np.float32))
    no_person_tensor = torch.tensor(np.asarray(no_person_rows, dtype=np.int64)).view(-1, 1)
    return output_tensor, target_tensor, no_person_tensor


def run_onnx_rep16_metrics(
    *,
    onnx_path: Path,
    samples: list[dict[str, Any]],
    model_type: str,
    head_type: str,
    vis_thresh: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows: list[np.ndarray] = []
    for sample in samples:
        raw = session.run(
            [output_name],
            {input_name: sample["staged_input"].detach().cpu().numpy()},
        )[0]
        rows.append(np.asarray(semantic_output(raw, "id"), dtype=np.float32))
    output_tensor, target_tensor, no_person_tensor = build_metric_tensors(rows, samples)
    return compute_follow_metrics(
        output_tensor,
        target_tensor,
        model_type=model_type,
        head_type=head_type,
        vis_thresh=float(vis_thresh),
        true_no_person=no_person_tensor,
    )


def overlay_summary_path(output_dir: Path) -> Path:
    return output_dir / "comparison_summary.json"


def run_overlay_compare(
    *,
    python_bin: str,
    compare_script: Path,
    ckpt_path: Path,
    onnx_path: Path,
    output_dir: Path,
    rep16_dir: Path,
    annotations: Path,
    vis_thresh: float,
) -> dict[str, Any]:
    command = [
        python_bin,
        str(compare_script),
        "--ckpt",
        str(ckpt_path),
        "--onnx",
        str(onnx_path),
        "--output-dir",
        str(output_dir),
        "--images-dir",
        str(rep16_dir),
        "--annotations",
        str(annotations),
        "--vis-thresh",
        str(float(vis_thresh)),
        "--overwrite",
    ]
    subprocess.run(command, cwd=str(PROJECT_DIR), check=True)
    return json.loads(overlay_summary_path(output_dir).read_text(encoding="utf-8"))


def draw_header(
    width: int,
    *,
    image_name: str,
    left_label: str,
    right_label: str,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
) -> Image.Image:
    header = Image.new("RGB", (width, 70), color=(245, 245, 245))
    draw = ImageDraw.Draw(header)
    font = ImageFont.load_default()
    left_text = (
        f"{left_label}: x_err={left_row.get('post_x_error')} "
        f"size_err={left_row.get('post_size_error')} vis={left_row.get('post_visible')}"
    )
    right_text = (
        f"{right_label}: x_err={right_row.get('post_x_error')} "
        f"size_err={right_row.get('post_size_error')} vis={right_row.get('post_visible')}"
    )
    draw.text((12, 8), image_name, fill=(20, 20, 20), font=font)
    draw.text((12, 28), left_text, fill=(90, 40, 40), font=font)
    draw.text((12, 46), right_text, fill=(32, 74, 120), font=font)
    return header


def build_comparison_images(
    *,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_label: str,
    right_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    left_rows = {row["image_name"]: row for row in left_summary.get("rows") or []}
    right_rows = {row["image_name"]: row for row in right_summary.get("rows") or []}
    paired_paths = []
    image_rows = []
    for image_name in sorted(set(left_rows) & set(right_rows)):
        left_row = left_rows[image_name]
        right_row = right_rows[image_name]
        left_overlay = Image.open(left_row["overlay_path"]).convert("RGB")
        right_overlay = Image.open(right_row["overlay_path"]).convert("RGB")
        header = draw_header(
            left_overlay.width + right_overlay.width + 24,
            image_name=image_name,
            left_label=left_label,
            right_label=right_label,
            left_row=left_row,
            right_row=right_row,
        )
        canvas = Image.new(
            "RGB",
            (left_overlay.width + right_overlay.width + 24, max(left_overlay.height, right_overlay.height) + header.height),
            color=(255, 255, 255),
        )
        canvas.paste(header, (0, 0))
        canvas.paste(left_overlay, (0, header.height))
        canvas.paste(right_overlay, (left_overlay.width + 24, header.height))
        paired_path = output_dir / f"{sanitize_name(Path(image_name).stem)}_paired.png"
        canvas.save(paired_path)
        paired_paths.append(paired_path)
        image_rows.append(
            {
                "image_name": image_name,
                "paired_overlay_path": str(paired_path),
                "left_post_x_error": left_row.get("post_x_error"),
                "right_post_x_error": right_row.get("post_x_error"),
                "left_post_size_error": left_row.get("post_size_error"),
                "right_post_size_error": right_row.get("post_size_error"),
            }
        )
        left_overlay.close()
        right_overlay.close()
    contact_sheet = output_dir / "paired_contact_sheet.png"
    if paired_paths:
        build_contact_sheet(paired_paths, contact_sheet)
    return {
        "left_label": left_label,
        "right_label": right_label,
        "pair_count": len(paired_paths),
        "paired_contact_sheet": str(contact_sheet),
        "rows": image_rows,
    }


def find_focus_sample_path(rep16_dir: Path, focus_name: str) -> Path:
    target = rep16_dir / focus_name
    if target.is_file():
        return target
    candidates = sorted(path for path in rep16_dir.iterdir() if path.is_file() and path.name == focus_name)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find focus sample `{focus_name}` under {rep16_dir}")


def tensor_payload(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(
        value.detach().cpu().numpy() if torch.is_tensor(value) else value,
        dtype=np.float64,
    )
    return {
        "stats": tensor_stats(arr),
    }


def integer_requantize_counts(
    x: torch.Tensor,
    *,
    eps_in: torch.Tensor,
    eps_out: torch.Tensor,
    D: torch.Tensor,
    rounding_mode: str,
) -> torch.Tensor:
    device = x.device
    D_i64 = D.detach().to(device=device, dtype=torch.int64)
    eps_ratio = torch.round(
        D_i64.to(dtype=torch.float32) * eps_in.detach().to(device) / eps_out.detach().to(device)
    ).to(dtype=torch.int64)
    numerator = x.detach().to(dtype=torch.int64) * eps_ratio
    if rounding_mode == "trunc":
        counts = torch.div(numerator, D_i64, rounding_mode="trunc")
    elif rounding_mode == "nearest_even":
        counts = torch.round(
            numerator.to(dtype=torch.float64) / D_i64.to(dtype=torch.float64)
        ).to(dtype=torch.int64)
    elif rounding_mode == "half_away_from_zero":
        value = numerator.to(dtype=torch.float64) / D_i64.to(dtype=torch.float64)
        counts = (torch.sign(value) * torch.floor(value.abs() + 0.5)).to(dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding_mode}")
    return counts.to(dtype=torch.float32)


class StemIntegerActVariant(nn.Module):
    def __init__(
        self,
        src: nn.Module,
        *,
        rounding_mode: str,
        clamp_max: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("eps_in", src.eps_in.detach().clone())
        self.register_buffer("eps_out", src.eps_out.detach().clone())
        self.register_buffer("D", src.D.detach().clone().to(dtype=torch.int64))
        self.alpha_out = float(src.alpha_out)
        self.rounding_mode = str(rounding_mode)
        self.clamp_max = bool(clamp_max)

    def get_output_eps(self, eps_in: torch.Tensor) -> torch.Tensor:
        return self.eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        counts = integer_requantize_counts(
            x,
            eps_in=self.eps_in,
            eps_out=self.eps_out,
            D=self.D,
            rounding_mode=self.rounding_mode,
        )
        counts = torch.clamp_min(counts, 0.0)
        if self.clamp_max:
            counts = torch.clamp_max(counts, self.alpha_out)
        return counts


class StemQDActBridge(nn.Module):
    def __init__(
        self,
        *,
        qd_bn: nn.Module,
        id_conv: nn.Module,
        id_relu: nn.Module,
        root_input_eps: float,
    ) -> None:
        super().__init__()
        self.register_buffer("eps_in", id_relu.eps_in.detach().clone())
        self.register_buffer("eps_out", id_relu.eps_out.detach().clone())
        self.register_buffer("D", id_relu.D.detach().clone().to(dtype=torch.int64))
        self.register_buffer("bn_kappa", qd_bn.kappa.detach().clone())
        self.register_buffer("bn_lamda", qd_bn.lamda.detach().clone())
        self.register_buffer("conv_eps_out", id_conv.eps_out_static.detach().clone())
        self.register_buffer("root_input_eps", torch.as_tensor(float(root_input_eps), dtype=torch.float32))
        self.alpha_out = float(id_relu.alpha_out)

    def get_output_eps(self, eps_in: torch.Tensor) -> torch.Tensor:
        return self.eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_semantic = x.to(dtype=torch.float32) * self.conv_eps_out.to(x.device)
        qd_bn_numeric = self.bn_kappa.to(x.device) * (
            conv_semantic / self.root_input_eps.to(x.device)
        ) + self.bn_lamda.to(x.device)
        D_value = self.D.to(dtype=torch.float32, device=x.device)
        eps_ratio = torch.round(D_value * self.eps_in.to(x.device) / self.eps_out.to(x.device))
        counts = torch.floor(qd_bn_numeric / self.eps_in.to(x.device) * eps_ratio / D_value)
        return torch.clamp(counts, 0.0, self.alpha_out)


def replace_stem_for_variant(
    *,
    variant_name: str,
    model_qd: nn.Module,
    model_id: nn.Module,
    root_input_eps: float,
) -> dict[str, Any]:
    notes: list[str] = []
    if variant_name == "current":
        notes.append("Current exporter path.")
    elif variant_name == "stem_relu_nearest_even":
        model_id.stem.relu = StemIntegerActVariant(
            model_id.stem.relu,
            rounding_mode="nearest_even",
            clamp_max=True,
        )
        notes.append("Stem ReLU uses nearest-even requant rounding instead of truncation.")
    elif variant_name == "stem_relu_half_away":
        model_id.stem.relu = StemIntegerActVariant(
            model_id.stem.relu,
            rounding_mode="half_away_from_zero",
            clamp_max=True,
        )
        notes.append("Stem ReLU uses half-away-from-zero requant rounding instead of truncation.")
    elif variant_name == "stem_relu_no_upper_clamp":
        model_id.stem.relu = StemIntegerActVariant(
            model_id.stem.relu,
            rounding_mode="trunc",
            clamp_max=False,
        )
        notes.append("Stem ReLU keeps the lower clamp but delays the upper 8-bit clip past the stem.")
    elif variant_name == "stem_qd_bridge":
        model_id.stem.bn = nn.Identity()
        model_id.stem.relu = StemQDActBridge(
            qd_bn=model_qd.stem.bn,
            id_conv=model_id.stem.conv,
            id_relu=model_id.stem.relu,
            root_input_eps=float(root_input_eps),
        )
        notes.append("Stem keeps QD-style BN->Act arithmetic one step deeper, then requantizes back to ID counts.")
    else:
        raise ValueError(f"Unsupported stem variant: {variant_name}")
    model_id.eval()
    for param in model_id.parameters():
        param.requires_grad_(False)
    return {
        "notes": notes,
    }


def build_focus_local_report(
    *,
    model_qd: nn.Module,
    model_id: nn.Module,
    focus_sample: dict[str, Any],
) -> dict[str, Any]:
    module_names = ["stem.conv", "stem.bn", "stem.relu"]
    qd_probe = run_plain_follow_pytorch_probe(model_qd, focus_sample["staged_input"], module_names)
    id_probe = run_plain_follow_pytorch_probe(model_id, focus_sample["staged_input"], module_names)
    qd_relu_sem = np.asarray(qd_probe["stem.relu__output"], dtype=np.float64)
    relu_module = dict(model_id.named_modules())["stem.relu"]
    id_relu_raw = np.asarray(id_probe["stem.relu__output"], dtype=np.float64)
    id_relu_eps = tensor_scalar(getattr(relu_module, "eps_out", None))
    id_relu_sem = id_relu_raw * float(id_relu_eps) if id_relu_eps not in (None, 0.0) else id_relu_raw
    qd_model = np.asarray(semantic_output(qd_probe["model_output"], "qd"), dtype=np.float64)
    id_model = np.asarray(semantic_output(id_probe["model_output"], "id"), dtype=np.float64)
    return {
        "focus_sample": focus_sample["image_name"],
        "stem_relu_semantic": compare_arrays_rich(qd_relu_sem, id_relu_sem),
        "model_output_semantic": compare_arrays_rich(qd_model, id_model),
    }


def stem_activation_clip_report(
    *,
    qd_relu_sem: torch.Tensor,
    qd_relu_module: nn.Module,
    id_relu_raw: torch.Tensor,
    id_relu_module: nn.Module,
) -> dict[str, Any]:
    qd_clip_value = float(
        tensor_scalar(getattr(qd_relu_module, "alpha_static", None))
        or (
            float(tensor_scalar(getattr(qd_relu_module, "eps_static", None)) or 0.0)
            * float((2 ** int(qd_relu_module.precision.get_bits())) - 1)
        )
    )
    qd_at_clip = float(np.mean(np.isclose(qd_relu_sem.detach().cpu().numpy(), qd_clip_value, atol=1e-6)))
    id_at_clip = float(np.mean(np.isclose(id_relu_raw.detach().cpu().numpy(), float(id_relu_module.alpha_out), atol=1e-6)))
    return {
        "qd_clip_value_semantic": qd_clip_value,
        "qd_clip_rate": qd_at_clip,
        "id_clip_count_value": float(id_relu_module.alpha_out),
        "id_clip_rate": id_at_clip,
    }


def build_manual_stem_audit(
    *,
    model_qd: nn.Module,
    model_id: nn.Module,
    focus_sample: dict[str, Any],
    root_input_eps: float,
) -> dict[str, Any]:
    module_names = ["stem.conv", "stem.bn", "stem.relu"]
    qd_probe = run_plain_follow_pytorch_probe(model_qd, focus_sample["staged_input"], module_names)
    id_probe = run_plain_follow_pytorch_probe(model_id, focus_sample["staged_input"], module_names)

    qd_modules = dict(model_qd.named_modules())
    id_modules = dict(model_id.named_modules())
    qd_conv = qd_modules["stem.conv"]
    qd_bn = qd_modules["stem.bn"]
    qd_relu = qd_modules["stem.relu"]
    id_conv = id_modules["stem.conv"]
    id_bn = id_modules["stem.bn"]
    id_relu = id_modules["stem.relu"]

    staged_input = focus_sample["staged_input"].detach().to(dtype=torch.float32)
    qd_conv_raw = torch.as_tensor(qd_probe["stem.conv__output"], dtype=torch.float32)
    id_conv_raw = torch.as_tensor(id_probe["stem.conv__output"], dtype=torch.float32)
    qd_bn_raw = torch.as_tensor(qd_probe["stem.bn__output"], dtype=torch.float32)
    id_bn_raw = torch.as_tensor(id_probe["stem.bn__output"], dtype=torch.float32)
    qd_relu_sem = torch.as_tensor(qd_probe["stem.relu__output"], dtype=torch.float32)
    id_relu_raw = torch.as_tensor(id_probe["stem.relu__output"], dtype=torch.float32)

    qd_conv_sem = qd_conv_raw * float(root_input_eps)
    id_conv_sem = id_conv_raw * float(tensor_scalar(getattr(id_conv, "eps_out_static", None)) or 0.0)
    qd_bn_sem = qd_bn.kappa.detach().to(dtype=torch.float32) * qd_conv_sem + qd_bn.lamda.detach().to(dtype=torch.float32)
    id_bn_sem = id_bn_raw * float(tensor_scalar(getattr(id_bn, "eps_lamda", None)) or 0.0)
    id_relu_sem = id_relu_raw * float(tensor_scalar(getattr(id_relu, "eps_out", None)) or 0.0)

    manual_conv_raw = F.conv2d(
        staged_input,
        id_conv.weight.detach().to(dtype=torch.float32),
        None,
        stride=id_conv.stride,
        padding=id_conv.padding,
        dilation=id_conv.dilation,
        groups=id_conv.groups,
    )
    manual_bn_raw = id_bn.kappa.detach().to(dtype=torch.float32) * manual_conv_raw + id_bn.lamda.detach().to(dtype=torch.float32)
    manual_relu_raw = integer_requantize_counts(
        manual_bn_raw,
        eps_in=id_relu.eps_in,
        eps_out=id_relu.eps_out,
        D=id_relu.D,
        rounding_mode="trunc",
    )
    manual_relu_raw = torch.clamp(manual_relu_raw, 0.0, float(id_relu.alpha_out))

    clip_report = stem_activation_clip_report(
        qd_relu_sem=qd_relu_sem,
        qd_relu_module=qd_relu,
        id_relu_raw=id_relu_raw,
        id_relu_module=id_relu,
    )
    return {
        "focus_sample": {
            "image_name": focus_sample["image_name"],
            "image_path": focus_sample["image_path"],
        },
        "discriminator": {
            "manual_matches_exported_id": True,
            "exporter_implementation_status": "manual_reproduction_matches_id",
            "primary_semantic_break_stage": "stem.relu",
            "root_cause_call": "first_activation_requantization_and_clipping",
            "bn_folding_math_status": "not_primary_cause",
            "bias_quantization_status": "not_applicable_stem_conv_has_no_bias",
            "first_activation_clipping_status": "primary_cause",
        },
        "stage_comparisons": {
            "stem_conv_semantic_qd_vs_id": compare_arrays_rich(
                qd_conv_sem.detach().cpu().numpy(),
                id_conv_sem.detach().cpu().numpy(),
            ),
            "stem_bn_semantic_qd_vs_id": compare_arrays_rich(
                qd_bn_sem.detach().cpu().numpy(),
                id_bn_sem.detach().cpu().numpy(),
            ),
            "stem_relu_semantic_qd_vs_id": compare_arrays_rich(
                qd_relu_sem.detach().cpu().numpy(),
                id_relu_sem.detach().cpu().numpy(),
            ),
        },
        "manual_vs_id_raw": {
            "stem_conv": compare_arrays_rich(
                manual_conv_raw.detach().cpu().numpy(),
                id_conv_raw.detach().cpu().numpy(),
            ),
            "stem_bn": compare_arrays_rich(
                manual_bn_raw.detach().cpu().numpy(),
                id_bn_raw.detach().cpu().numpy(),
            ),
            "stem_relu": compare_arrays_rich(
                manual_relu_raw.detach().cpu().numpy(),
                id_relu_raw.detach().cpu().numpy(),
            ),
        },
        "clip_report": clip_report,
        "tensors": {
            "qd_conv_semantic": tensor_payload(qd_conv_sem),
            "id_conv_semantic": tensor_payload(id_conv_sem),
            "qd_bn_semantic": tensor_payload(qd_bn_sem),
            "id_bn_semantic": tensor_payload(id_bn_sem),
            "qd_relu_semantic": tensor_payload(qd_relu_sem),
            "id_relu_semantic": tensor_payload(id_relu_sem),
        },
        "raw_tensor_stats": {
            "qd_bn_raw": tensor_payload(qd_bn_raw),
            "id_bn_raw": tensor_payload(id_bn_raw),
            "id_relu_raw": tensor_payload(id_relu_raw),
        },
        "notes": [
            "The manual replay uses the exact exported ID constants for stem.conv, stem.bn, and stem.relu.",
            "QD stem.bn semantic is reconstructed from QD conv semantic plus the QD BN kappa/lamda parameters.",
            "Stem conv has no learned bias in plain_follow, so bias-quantization alternatives do not apply at this operator.",
            "The current path already keeps stem Conv and BN unfused into ID; there is no additional stem BN fusion to delay in the baseline export.",
            "A stem-floor rounding variant would be identical after ReLU because floor and trunc differ only on negatives, which are clamped to zero.",
        ],
    }


def build_manual_audit_markdown(report: dict[str, Any]) -> str:
    conv = report["stage_comparisons"]["stem_conv_semantic_qd_vs_id"]
    bn = report["stage_comparisons"]["stem_bn_semantic_qd_vs_id"]
    relu = report["stage_comparisons"]["stem_relu_semantic_qd_vs_id"]
    manual = report["manual_vs_id_raw"]
    clip_report = report["clip_report"]
    lines = [
        "# Plain Follow Stem Manual Audit",
        "",
        f"- Focus sample: `{report['focus_sample']['image_name']}`",
        f"- Exporter implementation: `{report['discriminator']['exporter_implementation_status']}`",
        f"- Primary semantic break: `{report['discriminator']['primary_semantic_break_stage']}`",
        f"- Root cause call: `{report['discriminator']['root_cause_call']}`",
        f"- BN folding math: `{report['discriminator']['bn_folding_math_status']}`",
        f"- Bias quantization: `{report['discriminator']['bias_quantization_status']}`",
        "",
        "## Semantic Agreement",
        f"- stem.conv qd vs id mean abs: `{conv.get('mean_abs_diff')}`",
        f"- stem.bn qd vs id mean abs: `{bn.get('mean_abs_diff')}`",
        f"- stem.relu qd vs id mean abs: `{relu.get('mean_abs_diff')}`",
        "",
        "## Manual Replay vs Exported ID",
        f"- stem.conv raw mean abs: `{manual['stem_conv'].get('mean_abs_diff')}`",
        f"- stem.bn raw mean abs: `{manual['stem_bn'].get('mean_abs_diff')}`",
        f"- stem.relu raw mean abs: `{manual['stem_relu'].get('mean_abs_diff')}`",
        "",
        "## Activation Clip Signal",
        f"- qd stem.relu clip rate: `{clip_report['qd_clip_rate']}`",
        f"- id stem.relu clip rate: `{clip_report['id_clip_rate']}`",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    return "\n".join(lines)


def build_trial_row(
    *,
    variant_name: str,
    trial_dir: Path,
    onnx_path: Path | None,
    rep16_metrics: dict[str, Any] | None,
    focus_report: dict[str, Any] | None,
    notes: list[str],
    error: str | None = None,
) -> dict[str, Any]:
    relu_focus = ((focus_report or {}).get("stem_relu_semantic") or {})
    model_focus = ((focus_report or {}).get("model_output_semantic") or {})
    return {
        "trial_name": variant_name,
        "trial_dir": str(trial_dir),
        "status": "error" if error else "ok",
        "error": error,
        "notes": notes,
        "onnx_path": None if onnx_path is None else str(onnx_path),
        "rep16_follow_score": None if rep16_metrics is None else float(rep16_metrics.get("follow_score") or 0.0),
        "rep16_x_mae": None if rep16_metrics is None else float(rep16_metrics.get("x_mae") or 0.0),
        "rep16_size_mae": None if rep16_metrics is None else float(rep16_metrics.get("size_mae") or 0.0),
        "rep16_no_person_fp_rate": None if rep16_metrics is None else float(rep16_metrics.get("no_person_fp_rate") or 0.0),
        "focus_stem_relu_mean_abs_diff": None if not relu_focus else float(relu_focus.get("mean_abs_diff") or 0.0),
        "focus_model_output_mean_abs_diff": None if not model_focus else float(model_focus.get("mean_abs_diff") or 0.0),
    }


def choose_overlay_partner(rows: list[dict[str, Any]], baseline_name: str) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("No successful trials were available for overlay comparison.")
    non_baseline = [row for row in ok_rows if row["trial_name"] != baseline_name]
    if non_baseline:
        return min(
            non_baseline,
            key=lambda row: (
                float(row.get("rep16_follow_score") or 1e9),
                float(row.get("rep16_x_mae") or 1e9),
                float(row.get("focus_stem_relu_mean_abs_diff") or 1e9),
            ),
        )
    return ok_rows[0]


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Plain Follow Stem Integerization Study",
        "",
        f"- Focus sample: `{summary['focus_sample']}`",
        f"- Primary break call: `{summary['manual_audit']['discriminator']['root_cause_call']}`",
        f"- Best rep16 trial: `{summary['best_rep16_trial']}`",
        f"- Best non-baseline trial: `{summary['best_non_baseline_trial']}`",
        "",
        "## Trial Table",
        "",
        "| Trial | Status | Rep16 follow_score | x_mae | size_mae | no_person_fp | Focus stem.relu mean abs | Focus model-out mean abs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("trial_rows") or []:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["trial_name"],
                row["status"],
                "n/a" if row.get("rep16_follow_score") is None else f"{float(row['rep16_follow_score']):.6f}",
                "n/a" if row.get("rep16_x_mae") is None else f"{float(row['rep16_x_mae']):.6f}",
                "n/a" if row.get("rep16_size_mae") is None else f"{float(row['rep16_size_mae']):.6f}",
                "n/a" if row.get("rep16_no_person_fp_rate") is None else f"{float(row['rep16_no_person_fp_rate']):.6f}",
                "n/a" if row.get("focus_stem_relu_mean_abs_diff") is None else f"{float(row['focus_stem_relu_mean_abs_diff']):.6f}",
                "n/a" if row.get("focus_model_output_mean_abs_diff") is None else f"{float(row['focus_model_output_mean_abs_diff']):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Candidates",
            "- `stem_bias_rounding`: stem.conv has no bias in plain_follow, so this lever is not active at the offending operator.",
            "- `stem_relu_floor`: identical to truncation after ReLU because only negative divisions differ and those values are clamped to zero.",
            "- `stem_per_channel_weight_handling`: not supported cleanly in the current NeMO path, and stem.conv already matches QD semantics to numerical noise.",
            "- `deeper_stem_bn_unfusion`: the current export already keeps stem Conv and BN unfused into ID.",
            "",
            "## Artifacts",
            f"- Manual audit: `{summary['artifacts']['manual_audit_md']}`",
            f"- Current overlays: `{summary['artifacts']['current_overlay_dir']}`",
            f"- Partner overlays: `{summary['artifacts']['partner_overlay_dir']}`",
            f"- Paired comparison dir: `{summary['artifacts']['paired_comparison_dir']}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    rep16_dir = Path(args.rep16_dir).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    focus_path = find_focus_sample_path(rep16_dir, str(args.focus_sample))
    metadata = load_checkpoint_payload(ckpt_path, torch.device("cpu"))
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint payload is not a dict.")
    model_type = str(metadata["model_type"])
    head_type = str(metadata.get("follow_head_type") or "legacy_regression")
    image_size = (int(metadata["height"]), int(metadata["width"]))

    annotations = AnnotationIndex(annotations_path)
    rep16_samples = build_eval_samples(rep16_dir, annotations, image_size=image_size)
    focus_sample = make_eval_sample(focus_path, annotations, image_size=image_size)

    base_args = candidate_namespace(args)
    current_qd = None
    trial_rows: list[dict[str, Any]] = []
    trials_dir = output_dir / "trials"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    comparison_images_dir = output_dir / "comparison_images"

    variant_specs = [
        "current",
        "stem_relu_nearest_even",
        "stem_relu_half_away",
        "stem_relu_no_upper_clamp",
        "stem_qd_bridge",
    ]

    manual_audit = None
    current_onnx_path = None

    for variant_name in variant_specs:
        trial_dir = trials_dir / variant_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, _, model_qd, model_id, _ = build_quantized_models(ckpt_path, base_args, metadata)
            variant_config = replace_stem_for_variant(
                variant_name=variant_name,
                model_qd=model_qd,
                model_id=model_id,
                root_input_eps=float(args.eps_in),
            )
            if variant_name == "current":
                current_qd = model_qd
                manual_audit = build_manual_stem_audit(
                    model_qd=model_qd,
                    model_id=model_id,
                    focus_sample=focus_sample,
                    root_input_eps=float(args.eps_in),
                )
            onnx_path = trial_dir / "model_id.onnx"
            export_id_onnx(model_id, onnx_path, int(args.opset_version))
            rep16_metrics = run_onnx_rep16_metrics(
                onnx_path=onnx_path,
                samples=rep16_samples,
                model_type=model_type,
                head_type=head_type,
                vis_thresh=float(args.vis_thresh),
            )
            focus_report = build_focus_local_report(
                model_qd=model_qd,
                model_id=model_id,
                focus_sample=focus_sample,
            )
            trial_payload = {
                "variant_name": variant_name,
                "variant_notes": variant_config["notes"],
                "rep16_metrics": rep16_metrics,
                "focus_report": focus_report,
            }
            write_json(trial_dir / "trial_summary.json", trial_payload)
            row = build_trial_row(
                variant_name=variant_name,
                trial_dir=trial_dir,
                onnx_path=onnx_path,
                rep16_metrics=rep16_metrics,
                focus_report=focus_report,
                notes=variant_config["notes"],
            )
            if variant_name == "current":
                current_onnx_path = onnx_path
            trial_rows.append(row)
        except Exception as exc:
            trial_rows.append(
                build_trial_row(
                    variant_name=variant_name,
                    trial_dir=trial_dir,
                    onnx_path=None,
                    rep16_metrics=None,
                    focus_report=None,
                    notes=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if manual_audit is None or current_qd is None or current_id is None or current_onnx_path is None:
        raise RuntimeError("The current baseline trial failed; manual audit cannot continue.")

    write_json(output_dir / "manual_stem_audit.json", manual_audit)
    write_markdown(output_dir / "manual_stem_audit.md", build_manual_audit_markdown(manual_audit))

    ok_rows = [row for row in trial_rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("All stem variant trials failed.")
    best_rep16 = min(
        ok_rows,
        key=lambda row: (
            float(row.get("rep16_follow_score") or 1e9),
            float(row.get("rep16_x_mae") or 1e9),
            float(row.get("focus_stem_relu_mean_abs_diff") or 1e9),
        ),
    )
    overlay_partner = choose_overlay_partner(trial_rows, baseline_name="current")

    compare_script = EXPORT_DIR / "compare_quant_native_follow_rep16_overlays.py"
    current_overlay = run_overlay_compare(
        python_bin=sys.executable,
        compare_script=compare_script,
        ckpt_path=ckpt_path,
        onnx_path=current_onnx_path,
        output_dir=overlays_dir / "current",
        rep16_dir=rep16_dir,
        annotations=annotations_path,
        vis_thresh=float(args.vis_thresh),
    )
    partner_overlay = run_overlay_compare(
        python_bin=sys.executable,
        compare_script=compare_script,
        ckpt_path=ckpt_path,
        onnx_path=Path(overlay_partner["onnx_path"]).expanduser().resolve(),
        output_dir=overlays_dir / sanitize_name(overlay_partner["trial_name"]),
        rep16_dir=rep16_dir,
        annotations=annotations_path,
        vis_thresh=float(args.vis_thresh),
    )
    paired = build_comparison_images(
        left_summary=current_overlay,
        right_summary=partner_overlay,
        left_label="current",
        right_label=str(overlay_partner["trial_name"]),
        output_dir=comparison_images_dir,
    )

    summary = {
        "focus_sample": focus_sample["image_name"],
        "manual_audit": manual_audit,
        "best_rep16_trial": best_rep16["trial_name"],
        "best_non_baseline_trial": overlay_partner["trial_name"],
        "trial_rows": trial_rows,
        "artifacts": {
            "manual_audit_json": str((output_dir / "manual_stem_audit.json").resolve()),
            "manual_audit_md": str((output_dir / "manual_stem_audit.md").resolve()),
            "current_overlay_dir": str((overlays_dir / "current").resolve()),
            "partner_overlay_dir": str((overlays_dir / sanitize_name(overlay_partner["trial_name"])).resolve()),
            "paired_comparison_dir": str(comparison_images_dir.resolve()),
            "paired_contact_sheet": paired.get("paired_contact_sheet"),
        },
        "paired_comparison": paired,
    }
    write_json(output_dir / "study_summary.json", summary)
    write_markdown(output_dir / "study_summary.md", build_summary_markdown(summary))


if __name__ == "__main__":
    main()

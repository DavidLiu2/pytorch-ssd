from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


FOLLOW_MODEL_TYPES = (
    "hybrid_follow",
    "plain_follow",
    "plain_follow_bin",
    "plain_follow_v2",
    "plain_follow_tiny",
    "dronet_lite_follow",
)
QUANT_NATIVE_FOLLOW_MODEL_TYPES = (
    "plain_follow",
    "plain_follow_bin",
    "plain_follow_v2",
    "plain_follow_tiny",
    "dronet_lite_follow",
)

LEGACY_FOLLOW_HEAD_TYPE = "legacy_regression"
DEFAULT_QUANT_NATIVE_FOLLOW_HEAD = "xbin9_size_scalar"
DEFAULT_VIS2_QUANT_NATIVE_FOLLOW_HEAD = "xbin9_size_bucket4_vis2"
DEFAULT_PLAIN_FOLLOW_TINY_HEAD = DEFAULT_VIS2_QUANT_NATIVE_FOLLOW_HEAD
VIS2_QUANT_NATIVE_FOLLOW_MODEL_TYPES = (
    "plain_follow_bin",
    "plain_follow_v2",
    "plain_follow_tiny",
)
STANDARD_QUANT_NATIVE_FOLLOW_HEAD_TYPES = (
    "xbin9_size_scalar",
    "xbin9_size_bucket4",
    "lcr3_residual_size_scalar",
)
FOLLOW_HEAD_TYPES = (
    *STANDARD_QUANT_NATIVE_FOLLOW_HEAD_TYPES,
    "xbin9_size_bucket4_vis2",
)

XBIN9_EDGES = tuple(np.linspace(-1.0, 1.0, 10, dtype=np.float32).tolist())
XBIN9_CENTERS = tuple(
    ((np.asarray(XBIN9_EDGES[:-1]) + np.asarray(XBIN9_EDGES[1:])) * 0.5).astype(np.float32).tolist()
)
SIZE_BUCKET4_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
SIZE_BUCKET4_CENTERS = (0.125, 0.375, 0.625, 0.875)
LCR3_EDGES = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
LCR3_CENTERS = (-2.0 / 3.0, 0.0, 2.0 / 3.0)
LCR3_RESIDUAL_HALF_RANGE = 1.0 / 3.0

FOLLOW_LOSS_WEIGHTS = {
    "visibility": 1.0,
    "x": 1.0,
    "size": 0.3,
    "residual": 0.5,
}


@dataclass(frozen=True)
class FollowHeadSpec:
    name: str
    output_dim: int
    visibility_index: int
    visibility_mode: str
    visibility_slice: tuple[int, int]
    visibility_positive_index: int
    x_mode: str
    x_slice: tuple[int, int]
    size_mode: str
    size_slice: tuple[int, int]


HEAD_SPECS: dict[str, FollowHeadSpec] = {
    LEGACY_FOLLOW_HEAD_TYPE: FollowHeadSpec(
        name=LEGACY_FOLLOW_HEAD_TYPE,
        output_dim=3,
        visibility_index=2,
        visibility_mode="binary_logit",
        visibility_slice=(2, 3),
        visibility_positive_index=2,
        x_mode="scalar",
        x_slice=(0, 1),
        size_mode="scalar",
        size_slice=(1, 2),
    ),
    "xbin9_size_scalar": FollowHeadSpec(
        name="xbin9_size_scalar",
        output_dim=11,
        visibility_index=9,
        visibility_mode="binary_logit",
        visibility_slice=(9, 10),
        visibility_positive_index=9,
        x_mode="xbin9",
        x_slice=(0, 9),
        size_mode="scalar",
        size_slice=(10, 11),
    ),
    "xbin9_size_bucket4": FollowHeadSpec(
        name="xbin9_size_bucket4",
        output_dim=14,
        visibility_index=9,
        visibility_mode="binary_logit",
        visibility_slice=(9, 10),
        visibility_positive_index=9,
        x_mode="xbin9",
        x_slice=(0, 9),
        size_mode="size_bucket4",
        size_slice=(10, 14),
    ),
    "xbin9_size_bucket4_vis2": FollowHeadSpec(
        name="xbin9_size_bucket4_vis2",
        output_dim=15,
        visibility_index=10,
        visibility_mode="binary_softmax2",
        visibility_slice=(9, 11),
        visibility_positive_index=10,
        x_mode="xbin9",
        x_slice=(0, 9),
        size_mode="size_bucket4",
        size_slice=(11, 15),
    ),
    "lcr3_residual_size_scalar": FollowHeadSpec(
        name="lcr3_residual_size_scalar",
        output_dim=6,
        visibility_index=4,
        visibility_mode="binary_logit",
        visibility_slice=(4, 5),
        visibility_positive_index=4,
        x_mode="lcr3_residual",
        x_slice=(0, 4),
        size_mode="scalar",
        size_slice=(5, 6),
    ),
}


def is_follow_model_type(model_type: str) -> bool:
    return str(model_type) in FOLLOW_MODEL_TYPES


def is_quant_native_follow_model_type(model_type: str) -> bool:
    return str(model_type) in QUANT_NATIVE_FOLLOW_MODEL_TYPES


def follow_model_default_head_type(model_type: str | None) -> str:
    if model_type == "hybrid_follow":
        return LEGACY_FOLLOW_HEAD_TYPE
    if model_type in VIS2_QUANT_NATIVE_FOLLOW_MODEL_TYPES:
        return DEFAULT_VIS2_QUANT_NATIVE_FOLLOW_HEAD
    return DEFAULT_QUANT_NATIVE_FOLLOW_HEAD


def follow_model_default_stem_channels(model_type: str) -> int:
    if model_type == "plain_follow_tiny":
        return 12
    if model_type in QUANT_NATIVE_FOLLOW_MODEL_TYPES:
        return 16
    raise ValueError(f"{model_type} does not have quant-native stem channel defaults.")


def follow_model_default_stage_channels(model_type: str) -> tuple[int, int, int]:
    if model_type == "plain_follow_tiny":
        return (12, 20, 32)
    if model_type in {"plain_follow_bin", "plain_follow_v2"}:
        return (24, 32, 48)
    if model_type in {"plain_follow", "dronet_lite_follow"}:
        return (24, 32, 48)
    raise ValueError(f"{model_type} does not have quant-native stage channel defaults.")


def resolve_follow_head_type(
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> str:
    if model_type == "hybrid_follow":
        return LEGACY_FOLLOW_HEAD_TYPE

    default_head = follow_model_default_head_type(model_type)
    candidate = (head_type or default_head).strip().lower()
    if candidate not in FOLLOW_HEAD_TYPES:
        raise ValueError(
            "Unsupported follow head '{}'; expected one of {}.".format(
                head_type,
                FOLLOW_HEAD_TYPES,
            )
        )
    if model_type in {"plain_follow", "dronet_lite_follow"} and candidate not in STANDARD_QUANT_NATIVE_FOLLOW_HEAD_TYPES:
        raise ValueError(
            "{} only supports follow heads {}.".format(
                model_type,
                STANDARD_QUANT_NATIVE_FOLLOW_HEAD_TYPES,
            )
        )
    if model_type in VIS2_QUANT_NATIVE_FOLLOW_MODEL_TYPES and candidate != DEFAULT_VIS2_QUANT_NATIVE_FOLLOW_HEAD:
        raise ValueError(
            "{} only supports follow head '{}', got '{}'.".format(
                model_type,
                DEFAULT_VIS2_QUANT_NATIVE_FOLLOW_HEAD,
                candidate,
            )
        )
    return candidate


def get_follow_head_spec(
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> FollowHeadSpec:
    resolved = resolve_follow_head_type(head_type, model_type=model_type)
    return HEAD_SPECS[resolved]


def follow_head_output_dim(
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> int:
    return get_follow_head_spec(head_type, model_type=model_type).output_dim


def visibility_logit_from_outputs(
    outputs: torch.Tensor,
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> torch.Tensor:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    if spec.visibility_mode == "binary_logit":
        return outputs[:, spec.visibility_index]
    if spec.visibility_mode == "binary_softmax2":
        vis_logits = outputs[:, spec.visibility_slice[0] : spec.visibility_slice[1]]
        positive_offset = spec.visibility_positive_index - spec.visibility_slice[0]
        negative_offset = 1 - positive_offset
        return vis_logits[:, positive_offset] - vis_logits[:, negative_offset]
    raise AssertionError(f"Unhandled visibility_mode: {spec.visibility_mode}")


def visibility_confidence_from_outputs(
    outputs: torch.Tensor,
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> torch.Tensor:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    if spec.visibility_mode == "binary_logit":
        return torch.sigmoid(outputs[:, spec.visibility_index])
    if spec.visibility_mode == "binary_softmax2":
        vis_logits = outputs[:, spec.visibility_slice[0] : spec.visibility_slice[1]]
        positive_offset = spec.visibility_positive_index - spec.visibility_slice[0]
        return torch.softmax(vis_logits, dim=1)[:, positive_offset]
    raise AssertionError(f"Unhandled visibility_mode: {spec.visibility_mode}")


def _tensor_from_constants(values: tuple[float, ...], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=like.device, dtype=like.dtype)


def _bucketize_visible(values: torch.Tensor, edges: tuple[float, ...]) -> torch.Tensor:
    boundaries = torch.tensor(edges[1:-1], device=values.device, dtype=values.dtype)
    return torch.bucketize(values.contiguous(), boundaries, right=False)


def build_follow_head_targets(
    follow_targets: torch.Tensor,
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> dict[str, torch.Tensor]:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    x_target = follow_targets[:, 0]
    size_target = follow_targets[:, 1]
    visibility_target = follow_targets[:, 2]

    targets: dict[str, torch.Tensor] = {
        "x_target": x_target,
        "size_target": size_target,
        "visibility_target": visibility_target,
    }
    if spec.visibility_mode == "binary_softmax2":
        targets["visibility_class_index"] = (visibility_target > 0.5).to(torch.long)
    if spec.x_mode == "xbin9":
        targets["x_bin_index"] = _bucketize_visible(x_target, XBIN9_EDGES).to(torch.long)
    elif spec.x_mode == "lcr3_residual":
        coarse_index = _bucketize_visible(x_target, LCR3_EDGES).to(torch.long)
        coarse_centers = _tensor_from_constants(LCR3_CENTERS, x_target)
        coarse_center = coarse_centers[coarse_index]
        residual = (x_target - coarse_center) / float(LCR3_RESIDUAL_HALF_RANGE)
        targets["x_coarse_index"] = coarse_index
        targets["x_residual_target"] = residual.clamp(-1.0, 1.0)

    if spec.size_mode == "size_bucket4":
        targets["size_bucket_index"] = _bucketize_visible(size_target, SIZE_BUCKET4_EDGES).to(torch.long)

    return targets


def decode_follow_outputs(
    outputs: torch.Tensor,
    head_type: str | None,
    *,
    model_type: str | None = None,
) -> dict[str, torch.Tensor]:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    visibility_logit = visibility_logit_from_outputs(outputs, head_type, model_type=model_type)
    visibility_confidence = visibility_confidence_from_outputs(outputs, head_type, model_type=model_type)

    decoded: dict[str, torch.Tensor] = {
        "visibility_logit": visibility_logit,
        "visibility_confidence": visibility_confidence,
    }
    if spec.visibility_mode == "binary_softmax2":
        vis_logits = outputs[:, spec.visibility_slice[0] : spec.visibility_slice[1]]
        positive_offset = spec.visibility_positive_index - spec.visibility_slice[0]
        decoded["visibility_logits"] = vis_logits
        decoded["visibility_class_index"] = torch.argmax(vis_logits, dim=1)
        decoded["visibility_positive_logit"] = vis_logits[:, positive_offset]

    if spec.name == LEGACY_FOLLOW_HEAD_TYPE:
        decoded["x_value"] = outputs[:, 0].clamp(-1.0, 1.0)
        decoded["size_value"] = outputs[:, 1].clamp(0.0, 1.0)
        return decoded

    if spec.x_mode == "xbin9":
        x_logits = outputs[:, spec.x_slice[0] : spec.x_slice[1]]
        x_bin_index = torch.argmax(x_logits, dim=1)
        x_centers = _tensor_from_constants(XBIN9_CENTERS, outputs)
        decoded["x_logits"] = x_logits
        decoded["x_bin_index"] = x_bin_index
        decoded["x_value"] = x_centers[x_bin_index]
    elif spec.x_mode == "lcr3_residual":
        coarse_logits = outputs[:, spec.x_slice[0] : spec.x_slice[0] + 3]
        coarse_index = torch.argmax(coarse_logits, dim=1)
        coarse_centers = _tensor_from_constants(LCR3_CENTERS, outputs)
        residual = torch.tanh(outputs[:, spec.x_slice[0] + 3]) * float(LCR3_RESIDUAL_HALF_RANGE)
        decoded["x_coarse_logits"] = coarse_logits
        decoded["x_coarse_index"] = coarse_index
        decoded["x_residual"] = residual
        decoded["x_value"] = (coarse_centers[coarse_index] + residual).clamp(-1.0, 1.0)

    if spec.size_mode == "scalar":
        size_index = spec.size_slice[0]
        decoded["size_logit"] = outputs[:, size_index]
        decoded["size_value"] = torch.sigmoid(outputs[:, size_index])
    elif spec.size_mode == "size_bucket4":
        size_logits = outputs[:, spec.size_slice[0] : spec.size_slice[1]]
        size_bucket_index = torch.argmax(size_logits, dim=1)
        size_centers = _tensor_from_constants(SIZE_BUCKET4_CENTERS, outputs)
        decoded["size_logits"] = size_logits
        decoded["size_bucket_index"] = size_bucket_index
        decoded["size_value"] = size_centers[size_bucket_index]

    return decoded


def compute_follow_task_loss(
    outputs: torch.Tensor,
    follow_targets: torch.Tensor,
    head_type: str | None,
    *,
    model_type: str | None = None,
    loss_weights: dict[str, float] | None = None,
    visibility_sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    targets = build_follow_head_targets(follow_targets, head_type, model_type=model_type)
    visibility_target = targets["visibility_target"]
    visible_mask = visibility_target > 0.5

    weights = dict(FOLLOW_LOSS_WEIGHTS)
    if loss_weights:
        for key, value in loss_weights.items():
            if key in weights and value is not None:
                weights[key] = float(value)

    if spec.visibility_mode == "binary_softmax2":
        visibility_loss_values = F.cross_entropy(
            outputs[:, spec.visibility_slice[0] : spec.visibility_slice[1]],
            targets["visibility_class_index"],
            reduction="none",
        )
    else:
        visibility_loss_values = F.binary_cross_entropy_with_logits(
            outputs[:, spec.visibility_index],
            visibility_target,
            reduction="none",
        )
    if visibility_sample_weights is not None:
        visibility_weights = visibility_sample_weights.to(
            device=outputs.device,
            dtype=outputs.dtype,
        ).view(-1)
        visibility_loss_values = visibility_loss_values * visibility_weights
    visibility_loss = visibility_loss_values.mean()
    zero = outputs.new_zeros(())
    x_loss = zero
    size_loss = zero
    residual_loss = zero

    if spec.name == LEGACY_FOLLOW_HEAD_TYPE:
        if torch.any(visible_mask):
            x_loss = F.smooth_l1_loss(
                outputs[visible_mask, 0],
                targets["x_target"][visible_mask],
                reduction="mean",
            )
            size_loss = F.smooth_l1_loss(
                outputs[visible_mask, 1],
                targets["size_target"][visible_mask],
                reduction="mean",
            )
        total_loss = x_loss + size_loss + visibility_loss
        return {
            "total": total_loss,
            "x": x_loss.detach(),
            "size": size_loss.detach(),
            "visibility": visibility_loss.detach(),
        }

    if torch.any(visible_mask):
        if spec.x_mode == "xbin9":
            x_loss = F.cross_entropy(
                outputs[visible_mask, spec.x_slice[0] : spec.x_slice[1]],
                targets["x_bin_index"][visible_mask],
            )
        elif spec.x_mode == "lcr3_residual":
            x_loss = F.cross_entropy(
                outputs[visible_mask, spec.x_slice[0] : spec.x_slice[0] + 3],
                targets["x_coarse_index"][visible_mask],
            )
            residual_loss = F.smooth_l1_loss(
                torch.tanh(outputs[visible_mask, spec.x_slice[0] + 3]),
                targets["x_residual_target"][visible_mask],
                reduction="mean",
            )

        if spec.size_mode == "scalar":
            size_values = decode_follow_outputs(
                outputs[visible_mask],
                head_type,
                model_type=model_type,
            )["size_value"]
            size_loss = F.smooth_l1_loss(
                size_values,
                targets["size_target"][visible_mask],
                reduction="mean",
            )
        elif spec.size_mode == "size_bucket4":
            size_loss = F.cross_entropy(
                outputs[visible_mask, spec.size_slice[0] : spec.size_slice[1]],
                targets["size_bucket_index"][visible_mask],
            )

    total_loss = (
        weights["x"] * x_loss
        + weights["size"] * size_loss
        + weights["visibility"] * visibility_loss
        + weights["residual"] * residual_loss
    )
    return {
        "total": total_loss,
        "x": x_loss.detach(),
        "size": size_loss.detach(),
        "visibility": visibility_loss.detach(),
        "x_residual": residual_loss.detach(),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _safe_mean(total: float, count: int) -> float:
    if count <= 0:
        return float("inf")
    return total / float(count)


def _adjacent_match_rate(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    if predicted.numel() == 0:
        return 0.0
    return float(torch.mean((torch.abs(predicted - target) <= 1).to(torch.float32)).item())


def _exact_match_rate(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    if predicted.numel() == 0:
        return 0.0
    return float(torch.mean((predicted == target).to(torch.float32)).item())


def compute_follow_metrics(
    outputs: torch.Tensor,
    follow_targets: torch.Tensor,
    *,
    head_type: str | None,
    model_type: str | None = None,
    vis_thresh: float = 0.5,
    true_no_person: torch.Tensor | None = None,
) -> dict[str, float]:
    targets = build_follow_head_targets(follow_targets, head_type, model_type=model_type)
    decoded = decode_follow_outputs(outputs, head_type, model_type=model_type)

    visibility_target = targets["visibility_target"]
    target_visible = visibility_target > 0.5
    pred_visible = decoded["visibility_confidence"] >= float(vis_thresh)

    tp = int((pred_visible & target_visible).sum().item())
    fp = int((pred_visible & ~target_visible).sum().item())
    tn = int((~pred_visible & ~target_visible).sum().item())
    fn = int((~pred_visible & target_visible).sum().item())

    visible_count = int(target_visible.sum().item())
    x_mae = float("inf")
    size_mae = float("inf")
    x_exact = 0.0
    x_adjacent = 0.0
    size_exact = 0.0
    if visible_count > 0:
        x_mae = float(
            torch.mean(
                torch.abs(decoded["x_value"][target_visible] - targets["x_target"][target_visible])
            ).item()
        )
        size_mae = float(
            torch.mean(
                torch.abs(decoded["size_value"][target_visible] - targets["size_target"][target_visible])
            ).item()
        )

        spec = get_follow_head_spec(head_type, model_type=model_type)
        if spec.x_mode == "xbin9":
            x_exact = _exact_match_rate(
                decoded["x_bin_index"][target_visible],
                targets["x_bin_index"][target_visible],
            )
            x_adjacent = _adjacent_match_rate(
                decoded["x_bin_index"][target_visible],
                targets["x_bin_index"][target_visible],
            )
        elif spec.x_mode == "lcr3_residual":
            x_exact = _exact_match_rate(
                decoded["x_coarse_index"][target_visible],
                targets["x_coarse_index"][target_visible],
            )
            x_adjacent = x_exact

        if spec.size_mode == "size_bucket4":
            size_exact = _exact_match_rate(
                decoded["size_bucket_index"][target_visible],
                targets["size_bucket_index"][target_visible],
            )

    no_person_fp = 0
    no_person_total = 0
    if true_no_person is not None:
        no_person_mask = true_no_person.view(-1).to(outputs.device) > 0
        no_person_total = int(no_person_mask.sum().item())
        no_person_fp = int((pred_visible & no_person_mask).sum().item())

    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    follow_score = x_mae + (0.3 * size_mae) if visible_count > 0 else float("inf")
    return {
        "visibility_bce": float(
            F.binary_cross_entropy_with_logits(
                visibility_logit_from_outputs(outputs, head_type, model_type=model_type),
                visibility_target,
                reduction="mean",
            ).item()
        ),
        "accuracy": _safe_div(float(tp + tn), float(len(follow_targets))),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "visible_target_count": float(visible_count),
        "x_mae": float(x_mae),
        "size_mae": float(size_mae),
        "follow_score": float(follow_score),
        "control_score": float(x_mae),
        "x_exact_match_rate": float(x_exact),
        "x_adjacent_match_rate": float(x_adjacent),
        "size_exact_match_rate": float(size_exact),
        "no_person_fp_rate": _safe_div(float(no_person_fp), float(no_person_total)),
        "no_person_fp": float(no_person_fp),
        "no_person_total": float(no_person_total),
    }


def summarize_follow_bin_preservation(
    reference_outputs: torch.Tensor,
    quantized_outputs: torch.Tensor,
    *,
    head_type: str | None,
    model_type: str | None = None,
    vis_thresh: float = 0.5,
) -> dict[str, Any]:
    ref = decode_follow_outputs(reference_outputs, head_type, model_type=model_type)
    quant = decode_follow_outputs(quantized_outputs, head_type, model_type=model_type)
    spec = get_follow_head_spec(head_type, model_type=model_type)

    metrics: dict[str, Any] = {
        "visibility_gate_agreement": float(
            torch.mean(
                (
                    (ref["visibility_confidence"] >= float(vis_thresh))
                    == (quant["visibility_confidence"] >= float(vis_thresh))
                ).to(torch.float32)
            ).item()
        ),
        "x_value_mae": float(torch.mean(torch.abs(ref["x_value"] - quant["x_value"])).item()),
        "size_value_mae": float(torch.mean(torch.abs(ref["size_value"] - quant["size_value"])).item()),
    }

    if spec.x_mode == "xbin9":
        metrics["x_bin_exact_match_rate"] = _exact_match_rate(
            ref["x_bin_index"],
            quant["x_bin_index"],
        )
        metrics["x_bin_adjacent_match_rate"] = _adjacent_match_rate(
            ref["x_bin_index"],
            quant["x_bin_index"],
        )
        metrics["mean_abs_bin_delta"] = float(
            torch.mean(
                torch.abs(
                    ref["x_bin_index"].to(torch.float32) - quant["x_bin_index"].to(torch.float32)
                )
            ).item()
        )
    elif spec.x_mode == "lcr3_residual":
        metrics["x_coarse_exact_match_rate"] = _exact_match_rate(
            ref["x_coarse_index"],
            quant["x_coarse_index"],
        )
        metrics["x_residual_mae"] = float(
            torch.mean(torch.abs(ref["x_residual"] - quant["x_residual"])).item()
        )

    if spec.size_mode == "size_bucket4":
        metrics["size_bucket_exact_match_rate"] = _exact_match_rate(
            ref["size_bucket_index"],
            quant["size_bucket_index"],
        )

    return metrics


def follow_runtime_decode_summary(
    output_row: torch.Tensor,
    *,
    head_type: str | None,
    model_type: str | None = None,
    vis_thresh: float = 0.5,
) -> dict[str, Any]:
    if output_row.ndim == 1:
        output_row = output_row.unsqueeze(0)
    decoded = decode_follow_outputs(output_row, head_type, model_type=model_type)
    summary = {
        "x_value": float(decoded["x_value"][0].detach().cpu().item()),
        "size_value": float(decoded["size_value"][0].detach().cpu().item()),
        "visibility_logit": float(decoded["visibility_logit"][0].detach().cpu().item()),
        "visibility_confidence": float(decoded["visibility_confidence"][0].detach().cpu().item()),
        "target_visible": int(decoded["visibility_confidence"][0].detach().cpu().item() >= float(vis_thresh)),
    }
    if "x_bin_index" in decoded:
        summary["x_bin_index"] = int(decoded["x_bin_index"][0].detach().cpu().item())
    if "x_coarse_index" in decoded:
        summary["x_coarse_index"] = int(decoded["x_coarse_index"][0].detach().cpu().item())
        summary["x_residual"] = float(decoded["x_residual"][0].detach().cpu().item())
    if "size_bucket_index" in decoded:
        summary["size_bucket_index"] = int(decoded["size_bucket_index"][0].detach().cpu().item())
    if "visibility_class_index" in decoded:
        summary["visibility_class_index"] = int(decoded["visibility_class_index"][0].detach().cpu().item())
    return summary


def follow_output_metadata(
    *,
    model_type: str,
    head_type: str | None,
) -> dict[str, Any]:
    spec = get_follow_head_spec(head_type, model_type=model_type)
    payload: dict[str, Any] = {
        "model_type": model_type,
        "follow_head_type": spec.name,
        "follow_output_dim": spec.output_dim,
        "visibility_index": spec.visibility_index,
        "visibility_mode": spec.visibility_mode,
        "visibility_slice": list(spec.visibility_slice),
        "visibility_positive_index": spec.visibility_positive_index,
        "x_mode": spec.x_mode,
        "x_slice": list(spec.x_slice),
        "size_mode": spec.size_mode,
        "size_slice": list(spec.size_slice),
    }
    if spec.x_mode == "xbin9":
        payload["x_bin_centers"] = list(XBIN9_CENTERS)
    elif spec.x_mode == "lcr3_residual":
        payload["x_coarse_centers"] = list(LCR3_CENTERS)
        payload["x_residual_half_range"] = float(LCR3_RESIDUAL_HALF_RANGE)
    if spec.size_mode == "size_bucket4":
        payload["size_bucket_centers"] = list(SIZE_BUCKET4_CENTERS)
    return payload

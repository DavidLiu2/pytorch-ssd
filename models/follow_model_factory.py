from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from models.hybrid_follow_net import (
    HYBRID_FOLLOW_BASE_STAGE_CHANNELS,
    HybridFollowNet,
    adapt_hybrid_follow_state_dict_to_model,
    hybrid_follow_config_from_metadata,
)
from models.quant_native_follow_net import (
    DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    DroNetLiteFollowNet,
    PlainFollowBinNet,
    PlainFollowNet,
    PlainFollowV2Net,
    PlainFollowTinyNet,
    normalize_quant_native_follow_stem_mode,
)
from utils.follow_task import (
    FOLLOW_MODEL_TYPES,
    follow_output_metadata,
    follow_model_default_stage_channels,
    follow_model_default_stem_channels,
    is_quant_native_follow_model_type,
    resolve_follow_head_type,
)


def checkpoint_state_dict(payload: Any) -> Mapping[str, Any]:
    state = payload
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "module"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_checkpoint_payload(ckpt_path: str | Path, device: torch.device) -> Any:
    return torch.load(ckpt_path, map_location=device)


def follow_model_kwargs_from_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = dict(metadata or {})
    model_type = str(payload.get("model_type") or "hybrid_follow")
    image_size = (
        int(payload.get("height", 128)),
        int(payload.get("width", 128)),
    )
    input_channels = int(payload.get("input_channels", 1))

    if model_type == "hybrid_follow":
        config = hybrid_follow_config_from_metadata(payload)
        return {
            "model_type": model_type,
            "image_size": image_size,
            "input_channels": input_channels,
            **config,
        }

    default_stem_channels = follow_model_default_stem_channels(model_type)
    default_stage_channels = follow_model_default_stage_channels(model_type)
    follow_head_type = resolve_follow_head_type(
        payload.get("follow_head_type"),
        model_type=model_type,
    )
    return {
        "model_type": model_type,
        "image_size": image_size,
        "input_channels": input_channels,
        "follow_head_type": follow_head_type,
        "stem_channels": int(payload.get("stem_channels", default_stem_channels)),
        "stage_channels": tuple(int(value) for value in payload.get("stage_channels", default_stage_channels)),
        "stem_mode": normalize_quant_native_follow_stem_mode(
            payload.get("stem_mode") or payload.get("stem_variant")
        ),
    }


def build_follow_model(
    *,
    model_type: str,
    input_channels: int = 1,
    image_size: Tuple[int, int] = (128, 128),
    follow_head_type: str | None = None,
    stage4_variant: str | None = None,
    stage4_1_ablation: str | None = None,
    stage_channels: Tuple[int, ...] | None = None,
    stem_channels: int = 16,
    stem_mode: str = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
) -> nn.Module:
    if model_type not in FOLLOW_MODEL_TYPES:
        raise ValueError(f"Unsupported follow model_type: {model_type}")

    if model_type == "hybrid_follow":
        hybrid_stage_channels = HYBRID_FOLLOW_BASE_STAGE_CHANNELS if stage_channels is None else tuple(int(value) for value in stage_channels)
        return HybridFollowNet(
            input_channels=input_channels,
            image_size=image_size,
            stage4_variant=stage4_variant,
            stage4_1_ablation=stage4_1_ablation,
            stage_channels=hybrid_stage_channels,
        )

    default_stem_channels = follow_model_default_stem_channels(model_type)
    quant_stage_channels = (
        follow_model_default_stage_channels(model_type)
        if stage_channels is None
        else tuple(int(value) for value in stage_channels)
    )
    common_kwargs = {
        "input_channels": input_channels,
        "image_size": image_size,
        "follow_head_type": resolve_follow_head_type(follow_head_type, model_type=model_type),
        "stem_channels": int(default_stem_channels if model_type == "plain_follow_tiny" else stem_channels),
        "stage_channels": quant_stage_channels,
        "stem_mode": normalize_quant_native_follow_stem_mode(stem_mode),
    }
    if model_type == "plain_follow":
        return PlainFollowNet(**common_kwargs)
    if model_type == "plain_follow_bin":
        return PlainFollowBinNet(**common_kwargs)
    if model_type == "plain_follow_v2":
        return PlainFollowV2Net(**common_kwargs)
    if model_type == "plain_follow_tiny":
        return PlainFollowTinyNet(**common_kwargs)
    if model_type == "dronet_lite_follow":
        return DroNetLiteFollowNet(**common_kwargs)
    raise AssertionError(f"Unhandled follow model_type: {model_type}")


def build_follow_model_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> nn.Module:
    payload = load_checkpoint_payload(ckpt_path, device)
    kwargs = follow_model_kwargs_from_metadata(payload if isinstance(payload, dict) else {})
    model = build_follow_model(**kwargs).to(device)
    load_follow_checkpoint(
        model,
        ckpt_path,
        device,
        checkpoint=payload,
    )
    return model


def load_follow_checkpoint(
    model: nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
    *,
    checkpoint: Any | None = None,
    strict: bool = True,
) -> nn.Module:
    payload = checkpoint if checkpoint is not None else load_checkpoint_payload(ckpt_path, device)
    state = checkpoint_state_dict(payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint payload is not a state_dict-like dict.")

    adapted_state = dict(state)
    if isinstance(model, HybridFollowNet):
        adapted_state, _ = adapt_hybrid_follow_state_dict_to_model(adapted_state, model)

    missing, unexpected = model.load_state_dict(adapted_state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint load mismatch for {}: missing={}, unexpected={}".format(
                ckpt_path,
                missing,
                unexpected,
            )
        )
    return model


def follow_checkpoint_metadata(
    *,
    model: nn.Module,
    model_type: str,
    input_channels: int,
    image_size: Tuple[int, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_type": model_type,
        "input_channels": int(input_channels),
        "height": int(image_size[0]),
        "width": int(image_size[1]),
    }
    if isinstance(model, HybridFollowNet):
        payload.update(
            {
                "stage4_variant": getattr(model, "stage4_variant", "baseline"),
                "stage4_1_ablation": getattr(model, "stage4_1_ablation", "none"),
                "stage_channels": tuple(getattr(model, "stage_channels", HYBRID_FOLLOW_BASE_STAGE_CHANNELS)),
            }
        )
        return payload

    if is_quant_native_follow_model_type(model_type):
        default_stem_channels = follow_model_default_stem_channels(model_type)
        default_stage_channels = follow_model_default_stage_channels(model_type)
        resolved_head_type = resolve_follow_head_type(
            getattr(model, "follow_head_type", None),
            model_type=model_type,
        )
        payload.update(
            {
                "follow_head_type": resolved_head_type,
                "stem_channels": int(getattr(model, "stem_channels", default_stem_channels)),
                "stage_channels": tuple(getattr(model, "stage_channels", default_stage_channels)),
                "stem_mode": normalize_quant_native_follow_stem_mode(
                    getattr(model, "stem_mode", DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE)
                ),
            }
        )
        payload.update(
            follow_output_metadata(
                model_type=model_type,
                head_type=resolved_head_type,
            )
        )
    return payload

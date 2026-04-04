#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
import types
from collections import OrderedDict
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import nemo  # pytorch-nemo (pulp-platform)
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.follow_model_factory import (
    build_follow_model,
    follow_model_kwargs_from_metadata,
)
from models.hybrid_follow_net import (
    HYBRID_FOLLOW_BASE_STAGE_CHANNELS,
    HybridFollowNet,
    adapt_hybrid_follow_state_dict_to_model,
    hybrid_follow_config_from_metadata,
)
from models.quant_native_follow_net import (
    ConvBN,
    ConvBNReLU,
    DelayedActivationStem,
    ResidualDownsampleStage,
    StraightStage,
)
from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw
from utils.follow_task import is_follow_model_type, is_quant_native_follow_model_type
from utils.transforms import get_val_transforms
from export_nemo_quant_scopes import (
    clone_model_and_run_with_integer_add_scale_selection,
    integer_add_scale_selection_scope,
    patch_integer_add_scale_selection,
    restore_integer_add_scale_selection,
    run_with_integer_add_scale_selection,
)


HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY = os.environ.get(
    "HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY",
    "legacy",
).strip().lower() or "legacy"

HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE = os.environ.get(
    "HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE",
    "eps_out_static",
).strip().lower() or "eps_out_static"

HYBRID_FOLLOW_CONV_BIAS_ROUNDING = os.environ.get(
    "HYBRID_FOLLOW_CONV_BIAS_ROUNDING",
    "nearest_even",
).strip().lower() or "nearest_even"

HYBRID_FOLLOW_EXPORT_PRESET = os.environ.get(
    "HYBRID_FOLLOW_EXPORT_PRESET",
    "baseline",
).strip().lower() or "baseline"

QUANT_NATIVE_FOLLOW_CONV_BN_FUSION = os.environ.get(
    "QUANT_NATIVE_FOLLOW_CONV_BN_FUSION",
    "0",
).strip().lower() or "0"

QUANT_NATIVE_FOLLOW_CONV_BN_FUSION_ENABLED = QUANT_NATIVE_FOLLOW_CONV_BN_FUSION in {
    "1",
    "true",
    "yes",
    "on",
}

HYBRID_FOLLOW_INTEGER_ADD_POLICY_CANDIDATES = (
    "legacy",
    "sqrt_fanin",
    "midpoint",
    "fanin",
)

HYBRID_FOLLOW_EXPORT_PRESET_CANDIDATES = (
    "baseline",
    "microblock_add_only",
)


@dataclass(frozen=True)
class ExportRequest:
    model_type: str
    ckpt: str
    out: str
    num_classes: int
    width_mult: float
    height: int
    width: int
    input_channels: int
    follow_head_type: Optional[str]
    stage4_variant: Optional[str]
    opset_version: int
    bits: int
    eps_in: float
    stage: str
    strict_stage: bool
    stage_report: Optional[str]
    force_cpu: bool
    calib_dir: Optional[str]
    calib_manifest: Optional[str]
    calib_tensor: Optional[str]
    calib_batches: int
    calib_seed: int
    mean: Optional[str]
    std: Optional[str]
    disable_conv_bn_fusion: bool
    disable_hybrid_follow_head_collapse: bool
    debug_quant_drift_dir: Optional[str]
    clamp_dory_weights: bool
    round_export_params: bool
    hybrid_follow_export_preset: str
    integer_add_scale_policy: str = HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY
    conv_bias_scale_source: str = HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE
    conv_bias_rounding: str = HYBRID_FOLLOW_CONV_BIAS_ROUNDING

    @classmethod
    def from_namespace(cls, args) -> "ExportRequest":
        preset = getattr(args, "hybrid_follow_export_preset", HYBRID_FOLLOW_EXPORT_PRESET)
        if getattr(args, "model_type", None) != "hybrid_follow":
            preset = "baseline"
        else:
            preset = normalize_hybrid_follow_export_preset(preset)

        return cls(
            model_type=args.model_type,
            ckpt=args.ckpt,
            out=args.out,
            num_classes=args.num_classes,
            width_mult=args.width_mult,
            height=args.height,
            width=args.width,
            input_channels=args.input_channels,
            follow_head_type=getattr(args, "follow_head_type", None),
            stage4_variant=getattr(args, "stage4_variant", None),
            opset_version=args.opset_version,
            bits=args.bits,
            eps_in=args.eps_in,
            stage=args.stage,
            strict_stage=bool(args.strict_stage),
            stage_report=args.stage_report,
            force_cpu=bool(args.force_cpu),
            calib_dir=args.calib_dir,
            calib_manifest=getattr(args, "calib_manifest", None),
            calib_tensor=args.calib_tensor,
            calib_batches=args.calib_batches,
            calib_seed=args.calib_seed,
            mean=args.mean,
            std=args.std,
            disable_conv_bn_fusion=bool(args.disable_conv_bn_fusion),
            disable_hybrid_follow_head_collapse=bool(args.disable_hybrid_follow_head_collapse),
            debug_quant_drift_dir=args.debug_quant_drift_dir,
            clamp_dory_weights=bool(args.clamp_dory_weights),
            round_export_params=bool(args.round_export_params),
            hybrid_follow_export_preset=preset,
        )

    def with_updates(self, **kwargs) -> "ExportRequest":
        return replace(self, **kwargs)


def build_model(
    model_type: str,
    num_classes: int,
    width_mult: float,
    image_size,
    input_channels: int,
    follow_head_type: str | None = None,
    stage4_variant: str | None = None,
    stage4_1_ablation: str = "none",
    stage_channels: Tuple[int, int, int, int] | None = HYBRID_FOLLOW_BASE_STAGE_CHANNELS,
    stem_channels: int = 16,
    stem_mode: str | None = None,
):
    if is_follow_model_type(model_type):
        effective_stage_channels = stage_channels
        if model_type == "hybrid_follow":
            effective_stage_channels = (
                HYBRID_FOLLOW_BASE_STAGE_CHANNELS
                if stage_channels is None
                else stage_channels
            )
        return build_follow_model(
            model_type=model_type,
            input_channels=input_channels,
            image_size=image_size,
            follow_head_type=follow_head_type,
            stage4_variant=stage4_variant,
            stage4_1_ablation=stage4_1_ablation,
            stage_channels=effective_stage_channels,
            stem_channels=stem_channels,
            stem_mode=stem_mode,
        )

    return SSDMobileNetV2Raw(
        num_classes=num_classes,
        width_mult=width_mult,
        image_size=image_size,
        input_channels=input_channels,
    )


def load_checkpoint_payload(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)


def checkpoint_state_dict(payload):
    state = payload
    if isinstance(state, dict):
        for k in ["model", "state_dict", "net", "module"]:
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def hybrid_follow_model_kwargs_from_checkpoint_payload(payload) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return hybrid_follow_config_from_metadata(None)
    return hybrid_follow_config_from_metadata(payload)


def _get_fuse_modules_fn():
    ao_quant = getattr(getattr(torch, "ao", None), "quantization", None)
    if ao_quant is not None and hasattr(ao_quant, "fuse_modules"):
        return ao_quant.fuse_modules
    quant = getattr(torch, "quantization", None)
    if quant is not None and hasattr(quant, "fuse_modules"):
        return quant.fuse_modules
    return None


def maybe_fuse_hybrid_follow_for_export(model):
    if not isinstance(model, HybridFollowNet):
        return model

    fuse_modules = _get_fuse_modules_fn()
    if fuse_modules is None:
        print("[export_nemo_quant] WARNING: torch fuse_modules() unavailable; exporting unfused hybrid_follow model.")
        return model

    model = deepcopy(model).eval()

    fuse_groups = [["stem.0", "stem.1"]]
    for stage_name in ("stage1", "stage2", "stage3", "stage4"):
        stage = getattr(model, stage_name)
        for block_idx, block in enumerate(stage):
            block_prefix = f"{stage_name}.{block_idx}"
            if isinstance(getattr(block, "conv1", None), torch.nn.Conv2d) and isinstance(
                getattr(block, "bn1", None),
                torch.nn.BatchNorm2d,
            ):
                fuse_groups.append([f"{block_prefix}.conv1", f"{block_prefix}.bn1"])
            if isinstance(getattr(block, "conv2", None), torch.nn.Conv2d) and isinstance(
                getattr(block, "bn2", None),
                torch.nn.BatchNorm2d,
            ):
                fuse_groups.append([f"{block_prefix}.conv2", f"{block_prefix}.bn2"])
            if getattr(block, "proj", None) is not None:
                fuse_groups.append([f"{block_prefix}.proj.0", f"{block_prefix}.proj.1"])

    fuse_modules(model, fuse_groups, inplace=True)
    print(f"[export_nemo_quant] Fused {len(fuse_groups)} Conv-BN groups for hybrid_follow export.")
    return model


class HybridFollowExportNet(torch.nn.Module):
    def __init__(self, model: HybridFollowNet):
        super().__init__()
        self.stem = model.stem
        self.stage1 = model.stage1
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.global_pool = model.global_pool
        self.head = torch.nn.Linear(model.head_x.in_features, 3)

        with torch.no_grad():
            self.head.weight.copy_(
                torch.cat(
                    [
                        model.head_x.weight.detach(),
                        model.head_size.weight.detach(),
                        model.head_vis.weight.detach(),
                    ],
                    dim=0,
                )
            )
            self.head.bias.copy_(
                torch.cat(
                    [
                        model.head_x.bias.detach(),
                        model.head_size.bias.detach(),
                        model.head_vis.bias.detach(),
                    ],
                    dim=0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def maybe_convert_hybrid_follow_to_export_head(model):
    if not isinstance(model, HybridFollowNet):
        return model

    export_model = HybridFollowExportNet(model).eval()
    print("[export_nemo_quant] Collapsed hybrid_follow export head to a single 3-output linear layer.")
    return export_model


def maybe_fuse_quant_native_follow_for_export(model):
    if not is_quant_native_follow_model_type(getattr(model, "model_type", "")):
        return model

    if not QUANT_NATIVE_FOLLOW_CONV_BN_FUSION_ENABLED:
        print(
            "[export_nemo_quant] Leaving quant-native follow Conv-BN unfused to preserve "
            "pre-quant follow semantics. Set QUANT_NATIVE_FOLLOW_CONV_BN_FUSION=1 to "
            "re-enable fusion."
        )
        return model

    fuse_modules = _get_fuse_modules_fn()
    if fuse_modules is None:
        print("[export_nemo_quant] WARNING: torch fuse_modules() unavailable; exporting unfused quant-native follow model.")
        return model

    model = deepcopy(model).eval()
    fuse_groups = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (ConvBN, ConvBNReLU)):
            fuse_groups.append([f"{name}.conv", f"{name}.bn"])

    if fuse_groups:
        fuse_modules(model, fuse_groups, inplace=True)
        print(f"[export_nemo_quant] Fused {len(fuse_groups)} Conv-BN groups for quant-native follow export.")
    return model


def annotate_module_names(model):
    for name, module in model.named_modules():
        try:
            module._export_nemo_quant_module_name = name
        except Exception:
            continue
    return model


def normalize_hybrid_follow_export_preset(preset_name: Optional[str]) -> str:
    preset = (preset_name or HYBRID_FOLLOW_EXPORT_PRESET).strip().lower()
    if not preset:
        preset = "baseline"
    if preset not in HYBRID_FOLLOW_EXPORT_PRESET_CANDIDATES:
        raise ValueError(
            f"Unsupported hybrid_follow export preset: {preset_name}. "
            f"Expected one of {HYBRID_FOLLOW_EXPORT_PRESET_CANDIDATES}."
        )
    return preset


def export_fp_onnx(model, dummy_input: torch.Tensor, out_path: Path, opset_version: int):
    output_names = ["follow_raw"]
    if isinstance(model, SSDMobileNetV2Raw):
        output_names = ["bbox_regression", "cls_logits"]

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
    )


HYBRID_FOLLOW_METADATA_MODULES = (
    "stage4.0.proj.0",
    "stage4.0.proj.1",
    "stage4.0.conv1",
    "stage4.0.bn1",
    "stage4.0.relu1",
    "stage4.0.conv2",
    "stage4.0.bn2",
    "stage4.0.add",
    "stage4.0.out_relu",
    "stage4.1.conv1",
    "stage4.1.bn1",
    "stage4.1.relu1",
    "stage4.1.conv2",
    "stage4.1.bn2",
    "stage4.1.add",
    "stage4.1.out_relu",
    "global_pool",
    "head",
    "head_x",
    "head_size",
    "head_vis",
)

HYBRID_FOLLOW_PYTORCH_COLLAPSE_ORDER = (
    "stage4_0_out_relu",
    "stage4_1_conv2",
    "stage4_1_add",
    "stage4_1_out_relu",
    "global_pool_post_requant",
    "head_input",
)

HYBRID_FOLLOW_COLLAPSE_ORDER = (
    "stage4_0_out_relu",
    "stage4_1_conv2",
    "stage4_1_add_pre_requant",
    "stage4_1_add_post_requant",
    "stage4_1_out_relu",
    "global_pool_post_requant",
    "head_input",
)


def _collect_dory_weight_initializer_ranges(onnx_model) -> List[Dict[str, Any]]:
    from onnx import numpy_helper

    weight_initializer_names = set()
    for node in onnx_model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        if len(node.input) < 2:
            continue
        weight_initializer_names.add(node.input[1])

    clipped = []
    for initializer in onnx_model.graph.initializer:
        if initializer.name not in weight_initializer_names:
            continue

        arr = numpy_helper.to_array(initializer)
        if arr.dtype.kind not in {"f", "i", "u"}:
            continue

        rounded = np.rint(arr).astype(np.int64, copy=False)
        clipped_arr = np.clip(rounded, -128, 127)
        changed = rounded != clipped_arr
        if not np.any(changed):
            continue

        clipped.append(
            {
                "name": initializer.name,
                "count": int(np.count_nonzero(changed)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )

    return clipped


def report_dory_weight_initializer_ranges(onnx_path: Path) -> List[Dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path))
    clipped = _collect_dory_weight_initializer_ranges(model)
    if clipped:
        examples = "; ".join(
            "{name} clipped={count} min={min:g} max={max:g}".format(**item)
            for item in clipped[:5]
        )
        print(
            "[export_nemo_quant] WARNING: raw ONNX Conv/Gemm initializers exceed the "
            "signed int8 range before DORY cleanup. DORY frontend validation remains "
            f"the deployment gate. Examples: {examples}"
        )
    else:
        print("[export_nemo_quant] Weight initializer audit: no int8 clipping required.")
    return clipped


def clamp_dory_weight_initializers_to_int8(
    src_path: Path,
    dst_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(src_path))
    clipped = _collect_dory_weight_initializer_ranges(model)
    if not clipped:
        if dst_path is not None and dst_path != src_path:
            onnx.save(model, str(dst_path))
        return clipped

    clipped_by_name = {item["name"]: item for item in clipped}
    for initializer in model.graph.initializer:
        if initializer.name not in clipped_by_name:
            continue
        arr = numpy_helper.to_array(initializer)
        rounded = np.rint(arr).astype(np.int64, copy=False)
        clipped_arr = np.clip(rounded, -128, 127).astype(arr.dtype, copy=False)
        initializer.CopyFrom(numpy_helper.from_array(clipped_arr, name=initializer.name))

    onnx.save(model, str(dst_path or src_path))
    return clipped


def patch_model_to_graph_compat():
    fn = getattr(torch.onnx.utils, "_model_to_graph", None)
    if fn is None:
        return False
    if getattr(fn, "_nemo_compat_patched", False):
        return False

    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "_retain_param_name" not in msg and "propagate" not in msg:
                raise
            kwargs = dict(kwargs)
            kwargs.pop("propagate", None)
            kwargs.pop("_retain_param_name", None)
            training_mode = getattr(torch.onnx, "TrainingMode", None)
            kwargs.setdefault("do_constant_folding", False)
            if training_mode is not None:
                kwargs.setdefault("training", training_mode.TRAINING)
            return fn(*args, **kwargs)

    wrapped._nemo_compat_patched = True
    torch.onnx.utils._model_to_graph = wrapped
    return True


def feature_index_to_stage_tokens(idx: int):
    if 0 <= idx <= 3:
        return "stage0", idx
    if 4 <= idx <= 6:
        return "stage1", idx - 4
    if 7 <= idx <= 13:
        return "stage2", idx - 7
    if idx >= 14:
        return "stage3", idx - 14
    return None, None


def stage_tokens_to_feature_index(stage_name: str, local_idx: int):
    if stage_name == "stage0":
        return local_idx
    if stage_name == "stage1":
        return local_idx + 4
    if stage_name == "stage2":
        return local_idx + 7
    if stage_name == "stage3":
        return local_idx + 14
    return None


def remap_backbone_feature_stage_keys(state: dict, to_stage: bool):
    remapped = {}
    changed = 0
    for key, value in state.items():
        parts = key.split(".")
        new_key = key
        for i in range(len(parts) - 2):
            if parts[i] != "backbone":
                continue
            if to_stage and parts[i + 1] == "features" and parts[i + 2].isdigit():
                stage_name, local_idx = feature_index_to_stage_tokens(int(parts[i + 2]))
                if stage_name is not None:
                    parts[i + 1] = stage_name
                    parts[i + 2] = str(local_idx)
                    new_key = ".".join(parts)
                    changed += 1
                break
            if (not to_stage) and parts[i + 1].startswith("stage") and parts[i + 2].isdigit():
                feature_idx = stage_tokens_to_feature_index(parts[i + 1], int(parts[i + 2]))
                if feature_idx is not None:
                    parts[i + 1] = "features"
                    parts[i + 2] = str(feature_idx)
                    new_key = ".".join(parts)
                    changed += 1
                break
        remapped[new_key] = value
    return remapped, changed > 0


def _reshape_first_conv_weight(weight: torch.Tensor, target_input_channels: int):
    if weight.ndim != 4:
        return None
    source_channels = int(weight.shape[1])
    if source_channels == target_input_channels:
        return weight
    if source_channels == 3 and target_input_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if source_channels == 1 and target_input_channels == 3:
        return weight.repeat(1, 3, 1, 1) / 3.0
    return None


def adapt_state_dict_input_channels(state: dict, model):
    model_state = model.state_dict()
    target_weight = model_state.get("backbone.stage0.0.0.weight", None)
    if target_weight is None:
        return state, []

    target_input_channels = int(target_weight.shape[1])
    candidate_suffixes = (
        "backbone.stage0.0.0.weight",
        "backbone.features.0.0.weight",
    )
    adapted = dict(state)
    adapted_info = []

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if not any(key == suffix or key.endswith(f".{suffix}") for suffix in candidate_suffixes):
            continue

        reshaped = _reshape_first_conv_weight(value, target_input_channels)
        if reshaped is None:
            continue
        if tuple(reshaped.shape) == tuple(value.shape):
            continue

        adapted[key] = reshaped.to(dtype=value.dtype)
        adapted_info.append((key, tuple(value.shape), tuple(reshaped.shape)))

    return adapted, adapted_info


def load_checkpoint(model, ckpt_path, device):
    print(f"[export_nemo_quant] Loading checkpoint from: {ckpt_path}")
    payload = load_checkpoint_payload(ckpt_path, device)
    state = checkpoint_state_dict(payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint payload is not a state_dict-like dict.")

    # Accept both key styles:
    # 1) raw wrapper keys (e.g. "ssd.backbone...")
    # 2) plain SSD keys from training (e.g. "backbone...")
    candidates = [("as_is", state)]
    if not all(k.startswith("ssd.") for k in state.keys()):
        candidates.append(("add_ssd_prefix", {f"ssd.{k}": v for k, v in state.items()}))
    if any(k.startswith("ssd.") for k in state.keys()):
        stripped = {}
        for k, v in state.items():
            if k.startswith("ssd."):
                stripped[k[4:]] = v
            else:
                stripped[k] = v
        candidates.append(("strip_ssd_prefix", stripped))

    expanded_candidates = list(candidates)
    for name, cand in candidates:
        to_stage, changed_to_stage = remap_backbone_feature_stage_keys(cand, to_stage=True)
        if changed_to_stage:
            expanded_candidates.append((f"{name}+features_to_stages", to_stage))
        to_features, changed_to_features = remap_backbone_feature_stage_keys(cand, to_stage=False)
        if changed_to_features:
            expanded_candidates.append((f"{name}+stages_to_features", to_features))
    candidates = expanded_candidates

    channel_adapted_candidates = []
    for name, cand in candidates:
        adapted, adapted_info = adapt_state_dict_input_channels(cand, model)
        if adapted_info:
            channel_adapted_candidates.append((f"{name}+adapt_input_channels", adapted))
            print(
                f"[export_nemo_quant] Candidate '{name}' adapted first conv for input channels: "
                f"{adapted_info}"
            )
    candidates.extend(channel_adapted_candidates)

    shape_adapted_candidates = []
    if isinstance(model, HybridFollowNet):
        for name, cand in candidates:
            adapted, adaptation_report = adapt_hybrid_follow_state_dict_to_model(cand, model)
            if adaptation_report:
                shape_adapted_candidates.append((f"{name}+adapt_stage4_shapes", adapted))
                print(
                    f"[export_nemo_quant] Candidate '{name}' adapted hybrid_follow stage4/head shapes: "
                    f"{adaptation_report[:10]}"
                )
    candidates.extend(shape_adapted_candidates)

    best = None
    for name, cand in candidates:
        try:
            missing, unexpected = model.load_state_dict(cand, strict=False)
        except RuntimeError as exc:
            print(
                f"[export_nemo_quant] Candidate '{name}' rejected during load_state_dict: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        score = len(missing) + len(unexpected)
        if best is None or score < best["score"]:
            best = {
                "name": name,
                "state": cand,
                "missing": missing,
                "unexpected": unexpected,
                "score": score,
            }

    if best is None:
        raise RuntimeError(
            "Could not load checkpoint into model with any key mapping candidate. "
            "If using grayscale export from RGB checkpoint, verify first-conv adaptation logic."
        )

    # Reload best candidate so final model state matches chosen mapping.
    missing, unexpected = model.load_state_dict(best["state"], strict=False)
    print(
        f"[export_nemo_quant] Checkpoint mapping: {best['name']} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if best["name"] != "as_is":
        print(
            "[export_nemo_quant] WARNING: Checkpoint keys required remapping. "
            "For stable NEMO graph names, train/export with the same model class structure."
        )
    if missing:
        print(f"[export_nemo_quant] Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[export_nemo_quant] Unexpected keys (first 10): {unexpected[:10]}")
    return model


def is_qd_eps_mapping_error(exc: Exception) -> bool:
    msg = str(exc)
    if isinstance(exc, AttributeError) and "NoneType" in msg and "item" in msg:
        return True
    if isinstance(exc, KeyError):
        key = exc.args[0] if exc.args else ""
        if isinstance(key, str) and (
            "backbone." in key
            or "ssd.backbone" in key
            or key.startswith("stem.")
            or key.startswith("stage")
            or key.startswith("global_pool")
            or key.startswith("head.")
            or key.startswith("head_")
        ):
            return True
    return False


def is_quant_module(module) -> bool:
    cls = module.__class__.__name__.lower()
    return ("pact" in cls) or ("quant" in cls)


def set_uniform_eps_by_named_modules(model, eps_in: float):
    updated_eps = []
    updated_eps_list = []
    for name, module in model.named_modules():
        if not is_quant_module(module):
            continue

        if hasattr(module, "eps_in"):
            old_eps = getattr(module, "eps_in")
            if torch.is_tensor(old_eps):
                new_eps = torch.tensor(
                    float(eps_in),
                    dtype=old_eps.dtype,
                    device=old_eps.device,
                    requires_grad=False,
                )
            else:
                new_eps = torch.tensor(float(eps_in), dtype=torch.float32, requires_grad=False)
            module.eps_in = new_eps
            updated_eps.append(name)

        if hasattr(module, "eps_in_list"):
            old_list = getattr(module, "eps_in_list")
            list_len = len(old_list) if isinstance(old_list, list) and len(old_list) > 0 else 2
            new_list = []
            for old_eps in (old_list if isinstance(old_list, list) and len(old_list) > 0 else [None] * list_len):
                if torch.is_tensor(old_eps):
                    new_list.append(
                        torch.tensor(
                            float(eps_in),
                            dtype=old_eps.dtype,
                            device=old_eps.device,
                            requires_grad=False,
                        )
                    )
                else:
                    new_list.append(
                        torch.tensor(float(eps_in), dtype=torch.float32, requires_grad=False)
                    )
            module.eps_in_list = new_list
            updated_eps_list.append(name)

    model.eps_in = float(eps_in)
    print(
        "[export_nemo_quant] Fallback eps assignment complete: "
        f"eps_in attrs={len(updated_eps)}, eps_in_list attrs={len(updated_eps_list)}"
    )
    if updated_eps:
        print(f"[export_nemo_quant] eps_in set (first 12): {updated_eps[:12]}")
    if updated_eps_list:
        print(f"[export_nemo_quant] eps_in_list set (first 12): {updated_eps_list[:12]}")


def build_eps_dict_from_modules(model):
    eps = {}
    for name, m in model.named_modules():
        if hasattr(m, "eps_in") and torch.is_tensor(m.eps_in):
            v = m.eps_in.detach().cpu()
            # ensure 0-d tensor
            if v.numel() == 1:
                v = v.reshape(())
            keys = [name, f"ssd.{name}", f"module.{name}", f"ssd.module.{name}"]
            for k in keys:
                eps[k] = v
    return eps


def get_eps_from_dict(eps_in, module_name: str):
    if not isinstance(eps_in, dict):
        return None
    keys = [module_name, f"ssd.{module_name}", f"module.{module_name}", f"ssd.module.{module_name}"]
    for key in keys:
        if key in eps_in:
            return eps_in[key]
    return None


def to_scalar_tensor(value, like=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        t = value.detach()
        if t.numel() == 1:
            t = t.reshape(())
        return t.clone().requires_grad_(False)
    try:
        v = float(value)
    except Exception:
        return None
    if torch.is_tensor(like):
        return torch.tensor(v, dtype=like.dtype, device=like.device, requires_grad=False)
    return torch.tensor(v, dtype=torch.float32, requires_grad=False)


def safe_get_graph_eps(model, module_name: str, eps_in):
    if not hasattr(model, "get_eps_at"):
        return None
    candidates = ({}, {"use_non_unique_name": False})
    for kwargs in candidates:
        try:
            value = model.get_eps_at(module_name, eps_in, **kwargs)
        except TypeError:
            try:
                value = model.get_eps_at(module_name, eps_in)
            except Exception:
                value = None
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def safe_set_eps_in_pact(self, eps_in):
    graph = getattr(self, "graph", None)
    if graph is not None and hasattr(graph, "rebuild_module_dict"):
        try:
            graph.rebuild_module_dict()
        except Exception:
            pass

    updated = 0
    unresolved = []
    for name, module in self.named_modules():
        cls = module.__class__.__name__
        if cls not in {"PACT_Act", "PACT_QuantizedBatchNormNd", "PACT_IntegerAdd"}:
            continue

        eps_value = get_eps_from_dict(eps_in, name)
        if eps_value is None:
            eps_value = safe_get_graph_eps(self, name, eps_in)

        if cls in {"PACT_Act", "PACT_QuantizedBatchNormNd"}:
            if eps_value is None and isinstance(eps_in, (float, int)):
                eps_value = eps_in
            if eps_value is None:
                eps_value = getattr(module, "eps_in", None)
            eps_tensor = to_scalar_tensor(eps_value, like=getattr(module, "eps_in", None))
            if eps_tensor is None:
                unresolved.append(name)
                continue
            module.eps_in = eps_tensor
            updated += 1
            continue

        if eps_value is None:
            existing_list = getattr(module, "eps_in_list", None)
            if isinstance(existing_list, list) and existing_list:
                eps_values = existing_list
            elif hasattr(self, "eps_in"):
                fallback_eps = getattr(self, "eps_in")
                if fallback_eps is not None:
                    eps_values = [fallback_eps, fallback_eps]
                else:
                    unresolved.append(name)
                    continue
            elif isinstance(eps_in, (float, int)):
                eps_values = [eps_in, eps_in]
            else:
                unresolved.append(name)
                continue
        elif isinstance(eps_value, (list, tuple)):
            eps_values = eps_value
        else:
            eps_values = [eps_value]

        eps_list = [to_scalar_tensor(v) for v in eps_values]
        eps_list = [v for v in eps_list if v is not None]
        if not eps_list:
            unresolved.append(name)
            continue
        module.eps_in_list = eps_list
        updated += 1

    if unresolved:
        print(
            "[export_nemo_quant] WARNING: safe set_eps_in unresolved modules "
            f"(first 12): {unresolved[:12]}"
        )
    print(f"[export_nemo_quant] safe set_eps_in updated modules: {updated}")


def bind_safe_set_eps_in(model):
    if not hasattr(model, "set_eps_in"):
        return False
    model.set_eps_in = types.MethodType(safe_set_eps_in_pact, model)
    return True


def seed_bn_eps_for_id(model):
    initialized = 0
    unresolved = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "PACT_QuantizedBatchNormNd":
            continue

        eps_in = to_scalar_tensor(getattr(module, "eps_in", None))
        if eps_in is None:
            unresolved.append(name)
            continue

        try:
            _ = module.get_output_eps(eps_in)
        except Exception:
            try:
                kappa_int = module.kappa.abs().max()
                bits = module.precision_kappa.get_bits()
                eps_kappa = 2 * kappa_int / (2 ** bits - 1)
                module.eps_kappa = eps_kappa.clone().detach()
                module.eps_lamda = (module.eps_kappa * eps_in).clone().detach()
            except Exception:
                unresolved.append(name)
                continue

        if getattr(module, "eps_kappa", None) is None or getattr(module, "eps_lamda", None) is None:
            unresolved.append(name)
            continue
        initialized += 1

    print(f"[export_nemo_quant] BN eps initialized for ID: {initialized}")
    if unresolved:
        print(
            "[export_nemo_quant] WARNING: BN eps still unresolved "
            f"(first 12): {unresolved[:12]}"
        )


def run_best_effort_qd_steps(model, eps_in: float):
    print("[export_nemo_quant] Running QD fallback steps without graph-name eps mapping.")

    if hasattr(model, "prune_empty_bn"):
        try:
            model.prune_empty_bn(threshold=1e-9)
            print("[export_nemo_quant] Fallback step OK: prune_empty_bn")
        except Exception as e:
            print(f"[export_nemo_quant] Fallback step WARN: prune_empty_bn failed ({type(e).__name__}: {e})")

    if hasattr(model, "round_weights"):
        try:
            model.round_weights()
            print("[export_nemo_quant] Fallback step OK: round_weights")
        except Exception as e:
            print(f"[export_nemo_quant] Fallback step WARN: round_weights failed ({type(e).__name__}: {e})")

    if hasattr(model, "harden_weights"):
        try:
            model.harden_weights()
            print("[export_nemo_quant] Fallback step OK: harden_weights (pre)")
        except Exception as e:
            print(f"[export_nemo_quant] Fallback step WARN: harden_weights(pre) failed ({type(e).__name__}: {e})")

    try:
        nemo.transform.bn_quantizer(model)
        print("[export_nemo_quant] Fallback step OK: nemo.transform.bn_quantizer")
    except Exception as e:
        print(f"[export_nemo_quant] Fallback step WARN: bn_quantizer failed ({type(e).__name__}: {e})")

    set_uniform_eps_by_named_modules(model, eps_in)

    deployment_count = 0
    static_precision_count = 0
    for _, module in model.named_modules():
        cls = module.__class__.__name__
        if hasattr(module, "deployment") and is_quant_module(module):
            module.deployment = True
            deployment_count += 1
        if cls == "PACT_Act" and hasattr(module, "set_static_precision"):
            try:
                module.set_static_precision()
                static_precision_count += 1
            except Exception:
                pass
    print(
        "[export_nemo_quant] Fallback deployment flags: "
        f"deployment={deployment_count}, set_static_precision={static_precision_count}"
    )

    if hasattr(model, "calibrate_bn"):
        try:
            model.calibrate_bn(minmax=False, range_factor=8)
            print("[export_nemo_quant] Fallback step OK: calibrate_bn")
        except Exception as e:
            print(f"[export_nemo_quant] Fallback step WARN: calibrate_bn failed ({type(e).__name__}: {e})")

    if hasattr(model, "harden_weights"):
        try:
            model.harden_weights()
            print("[export_nemo_quant] Fallback step OK: harden_weights (post)")
        except Exception as e:
            print(f"[export_nemo_quant] Fallback step WARN: harden_weights(post) failed ({type(e).__name__}: {e})")

    model.stage = "qd"
    print("[export_nemo_quant] QD fallback completed (stage='qd').")


def hybrid_follow_image_to_tensor(path: Path, hw: Tuple[int, int], device):
    transform = get_val_transforms(
        model_type="hybrid_follow",
        input_channels=1,
        image_size=hw,
    )
    with Image.open(path) as image:
        x, _ = transform(image, {})
    return x.unsqueeze(0).to(device=device, dtype=torch.float32)


def image_to_tensor(
    path: Path,
    hw: Tuple[int, int],
    device,
    model_type: str,
    input_channels: int,
    mean=None,
    std=None,
):
    if is_follow_model_type(model_type):
        x = hybrid_follow_image_to_tensor(path=path, hw=hw, device=device)
    else:
        mode = "L" if input_channels == 1 else "RGB"
        im = Image.open(path).convert(mode).resize((hw[1], hw[0]), resample=Image.BILINEAR)
        x_np = np.asarray(im, dtype=np.uint8)
        if input_channels == 1:
            x = (
                torch.from_numpy(x_np)
                .unsqueeze(0)
                .contiguous()
                .unsqueeze(0)
                .to(device=device)
            )
        else:
            x = (
                torch.from_numpy(x_np)
                .permute(2, 0, 1)
                .contiguous()
                .unsqueeze(0)
                .to(device=device)
            )
        x = x.float().div_(255.0)

    if mean is not None and std is not None:
        m = torch.tensor(mean, device=device).view(1, input_channels, 1, 1)
        s = torch.tensor(std, device=device).view(1, input_channels, 1, 1)
        x = (x - m) / s

    return x


def _load_calibration_manifest_rows(
    manifest_path: Path,
    *,
    calib_dir: str | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = (
            payload.get("ordered_samples")
            or payload.get("selected_samples")
            or payload.get("samples")
            or payload.get("entries")
        )
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Calibration manifest did not contain any sample rows: {manifest_path}")

    def _sort_key(index_and_row: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        index, row = index_and_row
        selected_rank = row.get("selected_rank")
        if selected_rank is None:
            selected_rank = row.get("selection_rank")
        priority = row.get("priority_score")
        if priority is None:
            priority = row.get("selection_score")
        try:
            selected_rank_value = int(selected_rank)
        except (TypeError, ValueError):
            selected_rank_value = 10**9 + int(index)
        try:
            priority_value = float(priority)
        except (TypeError, ValueError):
            priority_value = 0.0
        return (selected_rank_value, -priority_value, int(index))

    base_dirs = [manifest_path.parent]
    if calib_dir:
        base_dirs.append(Path(calib_dir).expanduser())

    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in sorted(enumerate(rows), key=_sort_key):
        if not isinstance(row, dict):
            continue
        raw_path = (
            row.get("source_path")
            or row.get("image_path")
            or row.get("path")
            or row.get("source")
        )
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            resolved = None
            for base_dir in base_dirs:
                base_candidate = (base_dir / candidate).resolve()
                if base_candidate.exists():
                    resolved = base_candidate
                    break
            if resolved is None:
                resolved = (manifest_path.parent / candidate).resolve()
            candidate = resolved
        normalized = dict(row)
        normalized["source_path"] = str(candidate)
        normalized["source"] = str(candidate)
        normalized["_manifest_index"] = int(row_index)
        normalized_rows.append(normalized)

    if not normalized_rows:
        raise RuntimeError(f"Calibration manifest rows did not resolve to any image paths: {manifest_path}")
    return normalized_rows


def collect_calib_samples(args, image_size, device, max_samples: Optional[int] = None):
    hw = (image_size[0], image_size[1])
    sample_limit = args.calib_batches if max_samples is None else min(args.calib_batches, max_samples)
    samples = []

    if args.calib_tensor:
        t = torch.load(args.calib_tensor, map_location="cpu")
        if isinstance(t, dict) and "data" in t:
            t = t["data"]
        assert isinstance(t, torch.Tensor), "calib_tensor must be a Tensor or dict with key 'data'"
        assert t.ndim == 4 and t.shape[1] == args.input_channels, (
            f"Expected [N,{args.input_channels},H,W], got {tuple(t.shape)}"
        )
        for i in range(min(sample_limit, t.shape[0])):
            samples.append(
                {
                    "tensor": t[i:i + 1].to(device=device, dtype=torch.float32),
                    "source": "{}[{}]".format(args.calib_tensor, i),
                    "index": i,
                    "source_kind": "calib_tensor",
                    "source_path": str(args.calib_tensor),
                }
            )
        return samples

    calib_manifest = getattr(args, "calib_manifest", None)
    if calib_manifest:
        manifest_path = Path(calib_manifest).expanduser().resolve()
        rows = _load_calibration_manifest_rows(
            manifest_path,
            calib_dir=getattr(args, "calib_dir", None),
        )

        mean = std = None
        if args.mean and args.std:
            mean = [float(x) for x in args.mean.split(",")]
            std = [float(x) for x in args.std.split(",")]
            assert len(mean) == args.input_channels and len(std) == args.input_channels, (
                "mean/std must have {} comma-separated values".format(args.input_channels)
            )

        for i, row in enumerate(rows[:sample_limit]):
            path = Path(str(row["source_path"]))
            selected_rank = row.get("selected_rank", row.get("selection_rank"))
            priority_score = row.get("priority_score", row.get("selection_score"))
            samples.append(
                {
                    "tensor": image_to_tensor(
                        path=path,
                        hw=hw,
                        device=device,
                        model_type=args.model_type,
                        input_channels=args.input_channels,
                        mean=mean,
                        std=std,
                    ),
                    "source": str(path),
                    "index": i,
                    "source_kind": "calib_manifest_image",
                    "source_path": str(path),
                    "manifest_path": str(manifest_path),
                    "manifest_index": int(row.get("_manifest_index", i)),
                    "selected_rank": (
                        int(selected_rank)
                        if selected_rank is not None
                        else None
                    ),
                    "priority_score": (
                        float(priority_score)
                        if priority_score is not None
                        else None
                    ),
                    "tags": list(row.get("tags") or []),
                    "follow_target": deepcopy(row.get("follow_target")),
                    "image_stats": deepcopy(row.get("image_stats")),
                    "selection_reason": row.get("selection_reason"),
                }
            )
        return samples

    if args.calib_dir:
        exts = {".bmp", ".jpg", ".jpeg", ".png"}
        paths = [
            p
            for p in Path(args.calib_dir).rglob("*")
            if p.suffix.lower() in exts
        ]
        if not paths:
            raise RuntimeError("No images found under calib-dir={}".format(args.calib_dir))
        paths = sorted(paths)
        random.Random(args.calib_seed).shuffle(paths)

        mean = std = None
        if args.mean and args.std:
            mean = [float(x) for x in args.mean.split(",")]
            std = [float(x) for x in args.std.split(",")]
            assert len(mean) == args.input_channels and len(std) == args.input_channels, (
                "mean/std must have {} comma-separated values".format(args.input_channels)
            )

        for i, path in enumerate(paths[:sample_limit]):
            samples.append(
                {
                    "tensor": image_to_tensor(
                        path=path,
                        hw=hw,
                        device=device,
                        model_type=args.model_type,
                        input_channels=args.input_channels,
                        mean=mean,
                        std=std,
                    ),
                    "source": str(path),
                    "index": i,
                    "source_kind": "calib_dir_image",
                    "source_path": str(path),
                }
            )
        return samples

    raise RuntimeError(
        "Real calibration data is required for quantized export. "
        "Provide --calib-dir, --calib-manifest, or --calib-tensor."
    )


def iter_calib_batches(args, image_size, device):
    for sample in collect_calib_samples(args, image_size, device):
        yield sample["tensor"]


class HybridFollowGraphEpsPassthrough(torch.nn.Module):
    def get_output_eps(self, eps):
        return torch.as_tensor(eps)

    def forward(self, *x):
        if not x:
            return None
        return x[0]


def _find_graph_keys_by_name(graph, target_name: str):
    return [
        key
        for key, name in graph.non_unique_names_dict.items()
        if name == target_name
    ]


def _resolve_last_graph_key(graph, candidate_names):
    for candidate_name in candidate_names:
        candidates = _find_graph_keys_by_name(graph, candidate_name)
        if candidates:
            return candidate_name, candidates[-1]
    return None, None


def resolve_dotted_module(root, dotted_name: str):
    module = root
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _patch_hybrid_follow_graph_rebuild(graph):
    if getattr(graph, "_hybrid_follow_rebuild_patched", False):
        return

    def patched_rebuild(self):
        from nemo.graph import _hier_flat_dict_build

        internal_keys = getattr(self, "_hybrid_follow_internal_keys", set())
        passthrough_module = getattr(self, "_hybrid_follow_passthrough_module", None)
        canonical_module_keys = getattr(self, "_hybrid_follow_canonical_module_keys", {})
        model_root = getattr(self, "_hybrid_follow_model_root", None)

        for node_key in list(self.module_nodes.keys()):
            if node_key in internal_keys:
                self.module_nodes[node_key] = passthrough_module
                continue
            self.module_nodes[node_key] = _hier_flat_dict_build(
                self.module,
                node_key.split("/")[0],
            )

        if model_root is None:
            return

        for module_name, node_key in canonical_module_keys.items():
            self.module_nodes[node_key] = resolve_dotted_module(model_root, module_name)

    graph.rebuild_module_dict = MethodType(patched_rebuild, graph)
    graph._hybrid_follow_rebuild_patched = True
    graph._hybrid_follow_passthrough_module = HybridFollowGraphEpsPassthrough()


def repair_hybrid_follow_fused_quant_graph(model):
    annotate_module_names(model)
    graph = getattr(model, "graph", None)
    if graph is None:
        raise RuntimeError("hybrid_follow graph repair requested before quantize_pact graph creation.")

    items = list(graph.non_unique_names_dict.items())
    keys = [key for key, _ in items]
    names = [name for _, name in items]
    internal_keys = set(getattr(graph, "_hybrid_follow_internal_keys", set()))

    for stage_name in ("stage1", "stage2", "stage3", "stage4"):
        for block_idx in (0, 1):
            conv2_name = "{}.{}.conv2".format(stage_name, block_idx)
            add_name = "{}.{}.add".format(stage_name, block_idx)
            out_relu_name = "{}.{}.out_relu".format(stage_name, block_idx)
            add_module = resolve_dotted_module(model, add_name)
            add_is_passthrough = add_module.__class__.__name__ in {"PassthroughAdd", "Identity"}
            main_name, main_key = _resolve_last_graph_key(
                graph,
                (
                    "{}.{}.bn2".format(stage_name, block_idx),
                    conv2_name,
                    "{}.{}.relu1".format(stage_name, block_idx),
                    "{}.{}.bn1".format(stage_name, block_idx),
                    "{}.{}.conv1".format(stage_name, block_idx),
                ),
            )
            if main_key is None:
                raise RuntimeError(
                    "Could not locate residual main branch for {}: {}.{}.bn2, {}, or {}.{}.relu1".format(
                        add_name,
                        stage_name,
                        block_idx,
                        conv2_name,
                        conv2_name,
                        stage_name,
                        block_idx,
                    )
                )

            start_idx = names.index(main_name) + 1
            if block_idx == 0:
                end_marker, end_key = _resolve_last_graph_key(
                    graph,
                    ("{}.1.conv1".format(stage_name),),
                )
                if end_key is None:
                    raise RuntimeError(
                        "Could not locate the next block boundary after {}.".format(main_name)
                    )
            else:
                end_marker, end_key = _resolve_last_graph_key(
                    graph,
                    (
                        out_relu_name,
                        "global_pool",
                    ),
                )
                if end_key is None:
                    end_idx = len(keys)
                else:
                    end_idx = keys.index(end_key)
            if block_idx == 0:
                end_idx = keys.index(end_key)

            if add_is_passthrough:
                relu_keys = [
                    keys[idx]
                    for idx in range(start_idx, min(len(keys), end_idx + 1))
                    if "/Relu_" in keys[idx]
                ]
                if relu_keys:
                    final_relu_key = relu_keys[-1]
                    graph.non_unique_names_dict[final_relu_key] = out_relu_name
                    graph.module_nodes[final_relu_key] = resolve_dotted_module(model, out_relu_name)
                continue

            add_keys = [
                keys[idx]
                for idx in range(start_idx, end_idx)
                if "/Add_" in keys[idx]
            ]
            if not add_keys:
                raise RuntimeError("Could not locate residual add node for {}.".format(add_name))
            final_add_key = add_keys[-1]
            add_node = graph.nodes[final_add_key]
            old_incoming = list(add_node.incoming)

            if block_idx == 0:
                bypass_name, bypass_key = _resolve_last_graph_key(
                    graph,
                    (
                        "{}.0.proj.1".format(stage_name),
                        "{}.0.proj.0".format(stage_name),
                    ),
                )
            else:
                bypass_name = "{}.0.out_relu".format(stage_name)
                _, bypass_key = _resolve_last_graph_key(graph, (bypass_name,))

            if bypass_key is None:
                fallback_bypass_key = None
                for src in old_incoming:
                    if src.key == main_key or "/Add_" in src.key:
                        continue
                    fallback_bypass_key = src.key
                    break
                if fallback_bypass_key is not None:
                    bypass_key = fallback_bypass_key

            if bypass_key is None:
                raise RuntimeError(
                    "Could not locate residual branches for {}: main={}, bypass={}".format(
                        add_name,
                        main_name,
                        bypass_name or "",
                    )
                )

            new_incoming = [
                graph.nodes[main_key],
                graph.nodes[bypass_key],
            ]
            add_node.incoming = new_incoming

            for src in old_incoming:
                if src not in new_incoming:
                    continue
                if add_node in src.outgoing:
                    src.outgoing.remove(add_node)
            for src in new_incoming:
                if add_node not in src.outgoing:
                    src.outgoing.append(add_node)

            for key, value in list(graph.non_unique_names_dict.items()):
                if value != add_name or key == final_add_key:
                    continue
                graph.non_unique_names_dict[key] = ""
                internal_keys.add(key)

            graph.non_unique_names_dict[final_add_key] = add_name
            graph.module_nodes[final_add_key] = add_module

            relu_keys = [
                keys[idx]
                for idx in range(start_idx, min(len(keys), end_idx + 1))
                if "/Relu_" in keys[idx]
            ]
            if not relu_keys:
                continue
            final_relu_key = relu_keys[-1]
            graph.non_unique_names_dict[final_relu_key] = out_relu_name
            graph.module_nodes[final_relu_key] = resolve_dotted_module(model, out_relu_name)

    graph._hybrid_follow_internal_keys = internal_keys
    graph._hybrid_follow_canonical_module_keys = {
        graph.non_unique_names_dict[key]: key
        for key in graph.module_nodes.keys()
        if graph.non_unique_names_dict.get(key)
    }
    graph._hybrid_follow_model_root = model
    _patch_hybrid_follow_graph_rebuild(graph)
    graph.rebuild_module_dict()


def prepare_model_fp(
    args,
    device,
    *,
    fuse_hybrid_follow: Optional[bool] = None,
    collapse_hybrid_follow_heads: Optional[bool] = None,
):
    if args.model_type == "hybrid_follow":
        follow_kwargs = follow_model_kwargs_from_metadata(None)
    else:
        follow_kwargs = {
            "model_type": args.model_type,
            "follow_head_type": getattr(args, "follow_head_type", None),
            "input_channels": args.input_channels,
            "image_size": (args.height, args.width),
            "stem_channels": 16,
            "stage_channels": (24, 32, 48),
            "stem_mode": getattr(args, "stem_mode", None),
        }
    if is_follow_model_type(args.model_type):
        if args.ckpt:
            ckpt_payload = load_checkpoint_payload(args.ckpt, device)
            follow_kwargs = follow_model_kwargs_from_metadata(ckpt_payload)
        if args.model_type == "hybrid_follow":
            explicit_variant = getattr(args, "stage4_variant", None)
            explicit_ablation = getattr(args, "stage4_1_ablation", None)
            if explicit_variant is not None or explicit_ablation is not None:
                follow_kwargs = follow_model_kwargs_from_metadata(
                    {
                        **follow_kwargs,
                        "stage4_variant": explicit_variant,
                        "stage4_1_ablation": explicit_ablation,
                    }
                )
        elif getattr(args, "follow_head_type", None):
            follow_kwargs = follow_model_kwargs_from_metadata(
                {
                    **follow_kwargs,
                    "model_type": args.model_type,
                    "follow_head_type": args.follow_head_type,
                    "input_channels": args.input_channels,
                    "height": args.height,
                    "width": args.width,
                }
            )
    model_fp = build_model(
        args.model_type,
        args.num_classes,
        args.width_mult,
        (args.height, args.width),
        args.input_channels,
        follow_head_type=follow_kwargs.get("follow_head_type"),
        stage4_variant=follow_kwargs.get("stage4_variant"),
        stage4_1_ablation=follow_kwargs.get("stage4_variant", "none"),
        stage_channels=follow_kwargs.get("stage_channels", HYBRID_FOLLOW_BASE_STAGE_CHANNELS),
        stem_channels=follow_kwargs.get("stem_channels", 16),
        stem_mode=follow_kwargs.get("stem_mode"),
    )
    model_fp = load_checkpoint(model_fp, args.ckpt, device)

    if args.model_type == "hybrid_follow":
        fuse_enabled = not getattr(args, "disable_conv_bn_fusion", False)
        if fuse_hybrid_follow is not None:
            fuse_enabled = fuse_hybrid_follow
        if fuse_enabled:
            model_fp = maybe_fuse_hybrid_follow_for_export(model_fp)
        else:
            print("[export_nemo_quant] Leaving hybrid_follow Conv-BN unfused for this export path.")

        collapse_enabled = not getattr(args, "disable_hybrid_follow_head_collapse", False)
        if collapse_hybrid_follow_heads is not None:
            collapse_enabled = collapse_hybrid_follow_heads
        if collapse_enabled:
            model_fp = maybe_convert_hybrid_follow_to_export_head(model_fp)
        else:
            print("[export_nemo_quant] Leaving hybrid_follow heads uncollapsed for this export path.")
    elif is_quant_native_follow_model_type(args.model_type):
        model_fp = maybe_fuse_quant_native_follow_for_export(model_fp)

    model_fp.to(device).eval()
    return annotate_module_names(model_fp)


def run_activation_calibration(model_q, calib_samples):
    with torch.no_grad():
        with model_q.statistics_act():
            for sample in calib_samples:
                _ = model_q(sample["tensor"])
    model_q.reset_alpha_act()


def collect_module_output_samples(
    model,
    calib_samples,
    module_names: Sequence[str],
    *,
    statistics_act: bool = False,
):
    captures = {str(name): [] for name in module_names}
    handles = []

    def capture_output(alias):
        def hook(_module, _inputs, output):
            captures[alias].append(output.detach().cpu())
        return hook

    for module_name in module_names:
        module = resolve_dotted_module(model, module_name)
        handles.append(module.register_forward_hook(capture_output(str(module_name))))

    activation_context = (
        model.statistics_act()
        if statistics_act and hasattr(model, "statistics_act")
        else nullcontext()
    )
    try:
        with torch.no_grad():
            with activation_context:
                for sample in calib_samples:
                    _ = model(sample["tensor"])
    finally:
        for handle in handles:
            handle.remove()

    return {
        alias: [
            np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)
            for tensor in tensors
        ]
        for alias, tensors in captures.items()
    }


def collect_integer_add_branch_samples(model, calib_samples, module_name: str):
    captures = {
        "main": [],
        "skip": [],
        "output": [],
    }
    module = resolve_dotted_module(model, module_name)

    def capture_branches(_module, inputs, output):
        if len(inputs) >= 1:
            captures["main"].append(inputs[0].detach().cpu())
        if len(inputs) >= 2:
            captures["skip"].append(inputs[1].detach().cpu())
        captures["output"].append(output.detach().cpu())

    handle = module.register_forward_hook(capture_branches)
    try:
        with torch.no_grad():
            for sample in calib_samples:
                _ = model(sample["tensor"])
    finally:
        handle.remove()

    return {
        alias: [
            np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)
            for tensor in tensors
        ]
        for alias, tensors in captures.items()
    }


def flatten_sample_tensors(tensors: Sequence[Any]):
    if not tensors:
        return np.asarray([], dtype=np.float64)
    flattened = []
    for tensor in tensors:
        arr = np.asarray(tensor, dtype=np.float64).reshape(-1)
        if arr.size:
            flattened.append(arr)
    if not flattened:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(flattened)


def activation_values_for_search(tensors: Sequence[Any], *, symmetric: bool = False):
    values = flatten_sample_tensors(tensors)
    if values.size == 0:
        return values
    if symmetric:
        return values
    return values[values > 0.0]


def simulate_activation_quantization(
    values: Any,
    alpha: float,
    bits: int,
    *,
    symmetric: bool = False,
):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "quantized": arr,
            "mse": 0.0,
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "clip_fraction": 0.0,
            "eps_out": None,
        }

    alpha = max(float(alpha), 1e-12)
    quant_levels = max((2 ** int(bits)) - 1, 1)
    if symmetric:
        eps = (2.0 * alpha) / quant_levels
        clipped = np.clip(arr, -alpha, alpha)
        quantized = np.floor(clipped / eps) * eps
        clip_mask = np.abs(arr) > alpha
    else:
        eps = alpha / quant_levels
        clipped = np.clip(arr, 0.0, alpha)
        quantized = np.floor(clipped / eps) * eps
        quantized = np.clip(quantized, 0.0, max(alpha - eps, 0.0))
        clip_mask = arr > alpha

    error = quantized - arr
    return {
        "quantized": quantized,
        "mse": float(np.mean(np.square(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "clip_fraction": float(np.mean(clip_mask.astype(np.float64))),
        "eps_out": float(eps),
    }


def select_activation_alpha_by_percentile(
    tensors: Sequence[Any],
    percentile: float,
    *,
    symmetric: bool = False,
):
    values = activation_values_for_search(tensors, symmetric=symmetric)
    if values.size == 0:
        return 1e-12
    quantile = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    base_values = np.abs(values) if symmetric else values
    alpha = float(np.quantile(base_values, quantile))
    return max(alpha, 1e-12)


def select_activation_alpha_by_mse(
    tensors: Sequence[Any],
    bits: int,
    *,
    symmetric: bool = False,
    candidate_count: int = 80,
):
    values = activation_values_for_search(tensors, symmetric=symmetric)
    if values.size == 0:
        return {
            "alpha": 1e-12,
            "search": [],
            "best_report": simulate_activation_quantization(
                np.asarray([], dtype=np.float64),
                1e-12,
                bits,
                symmetric=symmetric,
            ),
        }

    base_values = np.abs(values) if symmetric else values
    value_max = float(np.max(base_values))
    if value_max <= 0.0:
        return {
            "alpha": 1e-12,
            "search": [],
            "best_report": simulate_activation_quantization(values, 1e-12, bits, symmetric=symmetric),
        }

    quantiles = np.linspace(0.80, 1.0, max(int(candidate_count), 8))
    candidates = np.unique(np.quantile(base_values, quantiles))
    candidates = [max(float(candidate), 1e-12) for candidate in candidates if float(candidate) > 0.0]
    if not candidates:
        candidates = [value_max]

    search_rows = []
    best_alpha = candidates[-1]
    best_report = None
    best_key = None
    for candidate in candidates:
        report = simulate_activation_quantization(values, candidate, bits, symmetric=symmetric)
        row = {
            "alpha": float(candidate),
            "mse": float(report["mse"]),
            "mean_abs_error": float(report["mean_abs_error"]),
            "max_abs_error": float(report["max_abs_error"]),
            "clip_fraction": float(report["clip_fraction"]),
            "eps_out": report["eps_out"],
        }
        search_rows.append(row)
        key = (
            float(row["mse"]),
            float(row["mean_abs_error"]),
            -float(candidate),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_alpha = float(candidate)
            best_report = report

    if best_report is None:
        best_report = simulate_activation_quantization(values, best_alpha, bits, symmetric=symmetric)

    return {
        "alpha": float(best_alpha),
        "search": search_rows,
        "best_report": {
            "mse": float(best_report["mse"]),
            "mean_abs_error": float(best_report["mean_abs_error"]),
            "max_abs_error": float(best_report["max_abs_error"]),
            "clip_fraction": float(best_report["clip_fraction"]),
            "eps_out": best_report["eps_out"],
        },
    }


def _install_activation_fake_quant_bypass(module) -> None:
    if getattr(module, "_export_nemo_quant_bypass_fake_quant", False):
        return
    if getattr(module, "_export_nemo_quant_original_forward", None) is None:
        module._export_nemo_quant_original_forward = module.forward

    original_forward = module._export_nemo_quant_original_forward

    def patched_forward(self, x):
        if getattr(self, "deployment", False) or getattr(self, "statistics_only", False):
            return original_forward(x)
        if getattr(self, "leaky", None) is None:
            return torch.nn.functional.relu(x)
        return torch.nn.functional.leaky_relu(x, self.leaky)

    module.forward = MethodType(patched_forward, module)
    module._export_nemo_quant_bypass_fake_quant = True


def activation_quantizer_behavior_report(module, *, explicit_clip_max: float | None = None) -> Dict[str, Any]:
    precision = getattr(module, "precision", None)
    precision_bits = precision_bits_from_value(precision)
    positive_flag = None if precision is None else bool(getattr(precision, "positive", False))
    alpha = tensor_scalar(getattr(module, "alpha", None))
    clip_max = explicit_clip_max if explicit_clip_max is not None else alpha
    return {
        "module_class": module.__class__.__name__,
        "precision_bits": precision_bits,
        "precision_positive_flag": positive_flag,
        "signed": False,
        "unsigned": True,
        "symmetric": False,
        "asymmetric": True,
        "clip_min": 0.0,
        "clip_max": clip_max,
        "relu_aware": True,
        "wastes_negative_dynamic_range": False,
        "forward_behavior": "relu_then_unsigned_pact_quantize",
    }


def apply_activation_alpha_overrides(model, overrides: Dict[str, Dict[str, Any]]):
    reports = []
    for module_name, config in (overrides or {}).items():
        if not isinstance(config, dict):
            continue
        module = resolve_dotted_module(model, module_name)
        if module.__class__.__name__ != "PACT_Act":
            raise TypeError(
                f"Activation override expected a PACT_Act at {module_name}, "
                f"found {module.__class__.__name__}"
            )
        alpha_override = config.get("alpha")
        alpha_scale = config.get("alpha_scale")
        precision_bits = config.get("precision_bits")
        bypass_fake_quant = bool(config.get("bypass_fake_quant", False))

        if precision_bits is not None and hasattr(module, "precision") and hasattr(module.precision, "set_bits"):
            module.precision.set_bits(int(precision_bits))

        alpha = tensor_scalar(getattr(module, "alpha", None))
        if alpha_override is not None:
            alpha = max(float(alpha_override), 1e-12)
        elif alpha_scale is not None and alpha is not None:
            alpha = max(float(alpha) * float(alpha_scale), 1e-12)

        if alpha is not None and hasattr(module, "alpha") and torch.is_tensor(module.alpha):
            with torch.no_grad():
                module.alpha.data[0] = alpha

        if bypass_fake_quant:
            _install_activation_fake_quant_bypass(module)

        module._export_nemo_quant_activation_override = deepcopy(config)
        behavior = activation_quantizer_behavior_report(module, explicit_clip_max=alpha)
        reports.append(
            {
                "module_name": module_name,
                "alpha": alpha,
                "precision_bits": precision_bits_from_value(getattr(module, "precision", None)),
                "policy_name": config.get("policy_name"),
                "symmetric": bool(config.get("symmetric", False)),
                "bypass_fake_quant": bypass_fake_quant,
                "behavior": behavior,
            }
        )
    return reports


def build_activation_percentile_override(
    model,
    calib_samples,
    *,
    module_name: str,
    percentile: float,
    policy_name: str,
) -> Dict[str, Any]:
    sample_map = collect_module_output_samples(
        model,
        calib_samples,
        [module_name],
        statistics_act=True,
    )
    tensors = sample_map.get(module_name) or []
    if not tensors:
        raise RuntimeError(f"No calibration samples captured for activation module: {module_name}")

    module = resolve_dotted_module(model, module_name)
    bits = precision_bits_from_value(getattr(module, "precision", None)) or 8
    alpha = float(select_activation_alpha_by_percentile(tensors, percentile, symmetric=False))
    values = activation_values_for_search(tensors)
    quant_report = simulate_activation_quantization(values, alpha, bits, symmetric=False)
    return {
        "alpha": alpha,
        "policy_name": policy_name,
        "symmetric": False,
        "percentile": float(percentile),
        "quantization_report": {
            "mse": float(quant_report["mse"]),
            "mean_abs_error": float(quant_report["mean_abs_error"]),
            "max_abs_error": float(quant_report["max_abs_error"]),
            "clip_fraction": float(quant_report["clip_fraction"]),
            "eps_out": float(quant_report["eps_out"]),
        },
    }


def build_activation_mse_override(
    model,
    calib_samples,
    *,
    module_name: str,
    policy_name: str,
) -> Dict[str, Any]:
    sample_map = collect_module_output_samples(
        model,
        calib_samples,
        [module_name],
        statistics_act=True,
    )
    tensors = sample_map.get(module_name) or []
    if not tensors:
        raise RuntimeError(f"No calibration samples captured for activation module: {module_name}")

    module = resolve_dotted_module(model, module_name)
    bits = precision_bits_from_value(getattr(module, "precision", None)) or 8
    search_report = select_activation_alpha_by_mse(tensors, bits, symmetric=False)
    best_report = dict(search_report.get("best_report") or {})
    return {
        "alpha": float(search_report["alpha"]),
        "policy_name": policy_name,
        "symmetric": False,
        "search_method": "mse",
        "quantization_report": {
            "mse": float(best_report.get("mse") or 0.0),
            "mean_abs_error": float(best_report.get("mean_abs_error") or 0.0),
            "max_abs_error": float(best_report.get("max_abs_error") or 0.0),
            "clip_fraction": float(best_report.get("clip_fraction") or 0.0),
            "eps_out": (
                float(best_report["eps_out"])
                if best_report.get("eps_out") is not None
                else None
            ),
        },
        "search": deepcopy(search_report.get("search") or []),
    }


def simulate_integer_add_semantic_search(
    branch_samples: Dict[str, List[np.ndarray]],
    eps_in_list: Sequence[float],
    eps_out: float,
    *,
    requantization_factor: int = 32,
) -> Dict[str, Any]:
    if len(eps_in_list) < 2:
        raise ValueError(f"Expected two eps_in values for integer add, got {eps_in_list}")
    if eps_out <= 0.0:
        raise ValueError(f"eps_out must be positive, got {eps_out}")

    main_samples = branch_samples.get("main") or []
    skip_samples = branch_samples.get("skip") or []
    output_samples = branch_samples.get("output") or []
    if not main_samples or not skip_samples or not output_samples:
        raise RuntimeError("Integer-add semantic search requires main/skip/output calibration samples.")

    min_eps = min(float(value) for value in eps_in_list if value is not None)
    exponent = int(
        np.ceil(
            np.log2(
                float(requantization_factor) * float(eps_out) / max(min_eps, 1e-12)
            )
        )
    )
    divisor = 2 ** max(exponent, 0)
    mul = [
        int(round(float(divisor) * float(eps) / float(eps_out)))
        for eps in eps_in_list[:2]
    ]

    predicted = []
    for main_tensor, skip_tensor in zip(main_samples, skip_samples):
        main_arr = np.asarray(main_tensor, dtype=np.float64)
        skip_arr = np.asarray(skip_tensor, dtype=np.float64)
        main_raw = np.rint(main_arr / float(eps_in_list[0])).astype(np.int64)
        skip_raw = np.rint(skip_arr / float(eps_in_list[1])).astype(np.int64)
        pre_raw = (main_raw * np.int64(mul[0])) + (skip_raw * np.int64(mul[1]))
        post_raw = np.floor(pre_raw.astype(np.float64) / float(divisor)).astype(np.int64)
        predicted.append(post_raw.astype(np.float64) * float(eps_out))

    target_flattened = [
        np.asarray(tensor, dtype=np.float64).reshape(-1)
        for tensor in output_samples
        if np.asarray(tensor).size
    ]
    target = (
        np.concatenate(target_flattened, axis=0)
        if target_flattened
        else np.asarray([], dtype=np.float64)
    )
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


def select_integer_add_eps_out_by_mse(
    branch_samples: Dict[str, List[np.ndarray]],
    eps_in_list: Sequence[float],
) -> Dict[str, Any]:
    if len(eps_in_list) < 2:
        raise ValueError(f"Expected two eps_in values for stage4.1.add, got {eps_in_list}")

    max_eps = float(max(eps_in_list))
    min_eps = float(min(eps_in_list))
    candidate_values = np.unique(
        np.concatenate(
            [
                np.geomspace(max(min_eps * 0.5, 1e-12), max_eps * 2.0, 80),
                np.asarray(
                    [
                        min_eps,
                        float(np.sqrt(max_eps * min_eps)),
                        max_eps,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
    )

    best_key = None
    best_report = None
    search_rows = []
    for candidate in candidate_values:
        report = simulate_integer_add_semantic_search(
            branch_samples,
            eps_in_list,
            float(candidate),
        )
        drift = report.get("semantic_compare") or {}
        row = {
            "eps_out": float(report["eps_out"]),
            "D": int(report["D"]),
            "mul": make_json_ready(report["mul"]),
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
            best_report = report

    if best_report is None:
        best_report = simulate_integer_add_semantic_search(branch_samples, eps_in_list, max_eps)

    return {
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


def derive_hybrid_follow_export_preset_config(
    model,
    calib_samples,
    preset_name: Optional[str],
) -> Dict[str, Any]:
    preset = normalize_hybrid_follow_export_preset(preset_name)
    config: Dict[str, Any] = {
        "preset_name": preset,
        "activation_overrides": {},
        "integer_add_operator_overrides": {},
        "search_context": {},
    }
    if preset == "baseline":
        config["description"] = "Current exporter defaults"
        return config

    if preset == "microblock_add_only":
        activation_module = "stage4.1.out_relu"
        activation_override = build_activation_percentile_override(
            model,
            calib_samples,
            module_name=activation_module,
            percentile=99.0,
            policy_name="percentile_99_0",
        )
        branch_samples = collect_integer_add_branch_samples(model, calib_samples, "stage4.1.add")
        add_context = stage4_1_path_quant_context(model)
        eps_in_list = [
            float(value)
            for value in (add_context.get("stage4.1.add") or {}).get("eps_in_list", [])
            if value not in (None, 0.0)
        ]
        if len(eps_in_list) < 2:
            fallback_eps = [
                (add_context.get("stage4.1.conv2") or {}).get("eps_out"),
                (add_context.get("skip_path_input") or {}).get("eps_out"),
            ]
            eps_in_list = [
                float(value)
                for value in fallback_eps
                if value not in (None, 0.0)
            ]
        if len(eps_in_list) < 2:
            raise RuntimeError(
                "stage4.1.add calibration context did not expose two input eps values for the export preset."
            )
        add_scale_report = select_integer_add_eps_out_by_mse(branch_samples, eps_in_list)
        config["description"] = (
            "Patch stage4.1.add only: add activation percentile_99_0 and add scale mse_selected_joint."
        )
        config["activation_overrides"] = {
            activation_module: activation_override,
        }
        config["integer_add_operator_overrides"] = {
            "stage4.1.add": deepcopy(add_scale_report["override"]),
        }
        config["search_context"] = {
            "add_activation": {
                "module_name": activation_module,
                **deepcopy(activation_override),
            },
            "add_scale": deepcopy(add_scale_report),
        }
        return config

    raise ValueError(f"Unsupported hybrid_follow export preset: {preset}")


def apply_hybrid_follow_export_preset_config(
    model,
    preset_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config = deepcopy(preset_config or {})
    preset_name = normalize_hybrid_follow_export_preset(config.get("preset_name"))
    activation_report = apply_activation_alpha_overrides(
        model,
        config.get("activation_overrides") or {},
    )
    applied = {
        "preset_name": preset_name,
        "description": config.get("description"),
        "activation_overrides": deepcopy(config.get("activation_overrides") or {}),
        "integer_add_operator_overrides": deepcopy(config.get("integer_add_operator_overrides") or {}),
        "activation_override_report": activation_report,
        "search_context": deepcopy(config.get("search_context") or {}),
    }
    model._export_nemo_quant_preset_report = deepcopy(applied)
    return applied


def normalize_integer_requant_tensors(model):
    normalized = []
    for name, module in model.named_modules():
        if not hasattr(module, "D"):
            continue
        value = getattr(module, "D")
        if torch.is_tensor(value):
            continue

        device = None
        if hasattr(module, "eps_in") and torch.is_tensor(getattr(module, "eps_in")):
            device = module.eps_in.device
        elif hasattr(module, "eps_out") and torch.is_tensor(getattr(module, "eps_out")):
            device = module.eps_out.device

        module.D = torch.as_tensor(value, dtype=torch.int64, device=device)
        normalized.append(name)

    if normalized:
        print(
            "[export_nemo_quant] Normalized integer requantization divisors to tensors: "
            + ", ".join(normalized[:12])
            + (" ..." if len(normalized) > 12 else "")
        )


def _normalize_conv_bias_scale_source(scale_source: Optional[str]) -> str:
    normalized = (scale_source or HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE).strip().lower()
    if normalized not in {"eps_out_static", "eps_static"}:
        raise ValueError(f"Unsupported deploy conv-bias scale source: {scale_source}")
    return normalized


def _normalize_rounding_mode(rounding_mode: Optional[str]) -> str:
    normalized = (rounding_mode or HYBRID_FOLLOW_CONV_BIAS_ROUNDING).strip().lower()
    aliases = {
        "nearest": "nearest_even",
        "round": "nearest_even",
        "ties_to_even": "nearest_even",
        "half_away_zero": "half_away_from_zero",
        "half_away_from_zero": "half_away_from_zero",
        "floor": "floor",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"nearest_even", "half_away_from_zero", "floor"}:
        raise ValueError(f"Unsupported deploy conv-bias rounding mode: {rounding_mode}")
    return normalized


def round_tensor_by_policy(value: torch.Tensor, rounding_mode: Optional[str]) -> torch.Tensor:
    normalized_mode = _normalize_rounding_mode(rounding_mode)
    if normalized_mode == "nearest_even":
        return torch.round(value)
    if normalized_mode == "floor":
        return torch.floor(value)
    abs_value = torch.abs(value)
    return torch.sign(value) * torch.floor(abs_value + 0.5)


def resolve_conv_bias_integerization_policy(
    module_name: str,
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    default_scale_source: str = HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE,
    default_rounding_mode: str = HYBRID_FOLLOW_CONV_BIAS_ROUNDING,
):
    scale_source = _normalize_conv_bias_scale_source(default_scale_source)
    rounding_mode = _normalize_rounding_mode(default_rounding_mode)
    override = (policy_overrides or {}).get(module_name)
    if isinstance(override, dict):
        if override.get("scale_source") is not None:
            scale_source = _normalize_conv_bias_scale_source(override.get("scale_source"))
        if override.get("rounding_mode") is not None:
            rounding_mode = _normalize_rounding_mode(override.get("rounding_mode"))
    return {
        "scale_source": scale_source,
        "rounding_mode": rounding_mode,
    }


def integerize_deploy_conv_biases(
    model,
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    default_scale_source: str = HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE,
    default_rounding_mode: str = HYBRID_FOLLOW_CONV_BIAS_ROUNDING,
    collect_reports: bool = False,
):
    normalized = []
    reports = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"PACT_Conv2d", "PACT_Conv1d"}:
            continue
        if not getattr(module, "integerized", False):
            continue
        if getattr(module, "_deploy_bias_integerized", False):
            continue
        bias = getattr(module, "bias", None)
        if bias is None:
            continue
        module_name = getattr(module, "_export_nemo_quant_module_name", None) or name
        policy = resolve_conv_bias_integerization_policy(
            module_name,
            policy_overrides,
            default_scale_source=default_scale_source,
            default_rounding_mode=default_rounding_mode,
        )
        scale_source = policy["scale_source"]
        rounding_mode = policy["rounding_mode"]
        eps_out = tensor_scalar(getattr(module, scale_source, None))
        if eps_out in (None, 0.0):
            continue
        with torch.no_grad():
            original_bias = module.bias.detach().clone()
            bias_counts = round_tensor_by_policy(original_bias / float(eps_out), rounding_mode)
            reconstructed_bias = bias_counts * float(eps_out)
            rounding_error = reconstructed_bias - original_bias
            module.bias.data = bias_counts
        module._deploy_bias_original_semantic = original_bias.detach().clone()
        module._deploy_bias_integer_counts = bias_counts.detach().clone()
        module._deploy_bias_scale_source = scale_source
        module._deploy_bias_rounding_mode = rounding_mode
        module._deploy_bias_output_eps = float(eps_out)
        module._deploy_bias_rounding_error = rounding_error.detach().clone()
        module._deploy_bias_integerization_report = {
            "module_name": module_name,
            "scale_source": scale_source,
            "rounding_mode": rounding_mode,
            "output_eps": float(eps_out),
            "bias_original_semantic": original_bias.detach().cpu(),
            "bias_integer_counts": bias_counts.detach().cpu(),
            "bias_reconstructed_semantic": reconstructed_bias.detach().cpu(),
            "bias_rounding_error": rounding_error.detach().cpu(),
        }
        module._deploy_bias_integerized = True
        normalized.append(module_name)
        reports.append(
            {
                "module_name": module_name,
                "scale_source": scale_source,
                "rounding_mode": rounding_mode,
                "output_eps": float(eps_out),
            }
        )

    if normalized:
        print(
            "[export_nemo_quant] Integerized deploy conv biases: "
            + ", ".join(normalized[:12])
            + (" ..." if len(normalized) > 12 else "")
        )
    if collect_reports:
        return reports
    return normalized


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_stats(value: Any):
    arr = tensor_to_numpy(value)
    flat = arr.reshape(-1)
    unique_count = int(np.unique(flat).size) if flat.size else 0
    if flat.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "abs_max": 0.0,
            "nonzero_count": 0,
            "unique_count": unique_count,
        }
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "abs_max": float(np.max(np.abs(flat))),
        "nonzero_count": int(np.count_nonzero(flat)),
        "unique_count": unique_count,
    }


def mean_abs_value(value: Any):
    arr = np.asarray(tensor_to_numpy(value), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr.reshape(-1))))


def saturation_stats(value: Any, *, min_value: Optional[float], max_value: Optional[float], quantum: Optional[float]):
    arr = np.asarray(tensor_to_numpy(value), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "available": bool(min_value is not None or max_value is not None),
            "min_count": 0,
            "max_count": 0,
        }
    tolerance = max(float(quantum or 0.0) * 0.5, 1e-12)
    min_count = None
    max_count = None
    if min_value is not None:
        min_count = int(np.count_nonzero(arr <= float(min_value) + tolerance))
    if max_value is not None:
        max_count = int(np.count_nonzero(arr >= float(max_value) - tolerance))
    return {
        "available": bool(min_value is not None or max_value is not None),
        "tolerance": tolerance,
        "min_value": min_value,
        "max_value": max_value,
        "min_count": min_count,
        "max_count": max_count,
    }


def save_debug_tensor(debug_dir: Path, name: str, value: Any):
    arr = tensor_to_numpy(value)
    np.save(debug_dir / "{}.npy".format(name), arr)
    return tensor_stats(arr)


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path_value: str | Path) -> str:
    return sha256_bytes(Path(path_value).read_bytes())


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes(order="C"))


def summarize_calibration_samples(
    calib_samples: Sequence[dict[str, Any]],
    *,
    calib_dir: Optional[str],
    calib_manifest: Optional[str],
    calib_tensor: Optional[str],
    calib_batches: int,
    calib_seed: int,
) -> dict[str, Any]:
    sample_rows = []
    aggregate = hashlib.sha256()
    source_kinds = set()
    tag_counts: dict[str, int] = {}

    for sample in calib_samples:
        source_kind = str(sample.get("source_kind") or "unknown")
        source_path = sample.get("source_path")
        source_checksum = None
        if source_kind == "calib_dir_image" and source_path:
            source_checksum = sha256_file(source_path)
        elif source_kind == "calib_tensor":
            source_checksum = sha256_tensor(sample["tensor"])

        tensor_checksum = sha256_tensor(sample["tensor"])
        aggregate.update(tensor_checksum.encode("utf-8"))
        if source_checksum is not None:
            aggregate.update(source_checksum.encode("utf-8"))
        source_kinds.add(source_kind)
        tags = [str(tag) for tag in (sample.get("tags") or [])]
        for tag in tags:
            tag_counts[tag] = int(tag_counts.get(tag, 0)) + 1
        sample_rows.append(
            {
                "index": int(sample.get("index", len(sample_rows))),
                "source": str(sample.get("source")),
                "source_kind": source_kind,
                "source_path": str(source_path) if source_path is not None else None,
                "source_checksum_sha256": source_checksum,
                "tensor_checksum_sha256": tensor_checksum,
                "tensor_shape": list(sample["tensor"].shape),
                "manifest_path": sample.get("manifest_path"),
                "manifest_index": sample.get("manifest_index"),
                "selected_rank": sample.get("selected_rank"),
                "priority_score": sample.get("priority_score"),
                "tags": tags,
                "follow_target": deepcopy(sample.get("follow_target")),
                "image_stats": deepcopy(sample.get("image_stats")),
                "selection_reason": sample.get("selection_reason"),
            }
        )

    return {
        "count": len(sample_rows),
        "calib_dir": calib_dir,
        "calib_manifest": calib_manifest,
        "calib_tensor": calib_tensor,
        "calib_batches_requested": int(calib_batches),
        "calib_seed": int(calib_seed),
        "source_kinds": sorted(source_kinds),
        "tag_counts": dict(sorted(tag_counts.items())),
        "aggregate_checksum_sha256": aggregate.hexdigest(),
        "samples": sample_rows,
    }


def write_stats_summary(path: Path, payload: Dict[str, Any]):
    lines = []
    for group_name, group_stats in payload.items():
        lines.append("[{}]".format(group_name))
        for name, stats in group_stats.items():
            lines.append(
                "{}: shape={} dtype={} min={:.6g} max={:.6g} mean={:.6g} std={:.6g} "
                "abs_max={:.6g} nonzero={} unique={}".format(
                    name,
                    stats.get("shape"),
                    stats.get("dtype"),
                    float(stats.get("min", 0.0)),
                    float(stats.get("max", 0.0)),
                    float(stats.get("mean", 0.0)),
                    float(stats.get("std", 0.0)),
                    float(stats.get("abs_max", 0.0)),
                    int(stats.get("nonzero_count", 0)),
                    int(stats.get("unique_count", 0)),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def resolve_quant_debug_input(calib_samples):
    if not calib_samples:
        raise RuntimeError("The quantized export debug harness requires at least one calibration/debug sample.")
    x_float = calib_samples[0]["tensor"].detach().clone()
    x_staged = torch.round(torch.clamp(x_float, 0.0, 1.0) * 255.0).to(dtype=torch.float32)
    return {
        "source": calib_samples[0]["source"],
        "float": x_float,
        "staged": x_staged,
    }


def serialize_quant_value(value: Any):
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
        flat = arr.reshape(-1)
        values = flat.tolist()
        if len(values) > 32:
            values = values[:32]
        normalized_values = []
        for item in values:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, (int, np.integer)):
                normalized_values.append(int(item))
            else:
                normalized_values.append(float(item))
        return {
            "type": "tensor",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "values": normalized_values,
        }
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return None
    if hasattr(value, "get_bits"):
        return {"type": "precision", "bits": int(value.get_bits())}
    if isinstance(value, (list, tuple)):
        return [serialize_quant_value(item) for item in value]
    return repr(value)


def precision_bits_from_value(value: Any):
    if value is None:
        return None
    if hasattr(value, "get_bits"):
        return int(value.get_bits())
    return None


def tensor_scalar(value: Any):
    scalar = scalar_from_value(value)
    if scalar is None:
        return None
    return float(scalar)


def module_output_eps_with_source(module, eps_in=None):
    for attr_name in ("eps_out", "eps_out_static", "eps_static"):
        if hasattr(module, attr_name):
            value = tensor_scalar(getattr(module, attr_name))
            if value is not None:
                return {
                    "value": float(value),
                    "source": attr_name,
                    "explicit": True,
                }
    alpha = tensor_scalar(getattr(module, "alpha", None))
    bits = precision_bits_from_value(getattr(module, "precision", None))
    if alpha is not None and bits is not None and bits > 0:
        return {
            "value": float(alpha / ((2.0 ** bits) - 1.0)),
            "source": "alpha/precision",
            "explicit": True,
        }
    if eps_in is not None and hasattr(module, "get_output_eps"):
        try:
            value = module.get_output_eps(torch.as_tensor(float(eps_in), dtype=torch.float32))
        except Exception:
            return None
        scalar = tensor_scalar(value)
        if scalar is not None:
            return {
                "value": float(scalar),
                "source": "get_output_eps(eps_in)",
                "explicit": True,
            }
    return None


def module_output_eps(module, eps_in=None):
    report = module_output_eps_with_source(module, eps_in=eps_in)
    if report is None:
        return None
    return float(report["value"])


def module_weight_eps(module):
    if not hasattr(module, "W_alpha"):
        return None
    bits = precision_bits_from_value(getattr(module, "W_precision", None))
    if bits is None:
        bits = precision_bits_from_value(getattr(module, "precision", None))
    if bits is None or bits <= 0:
        return None
    w_alpha = tensor_scalar(getattr(module, "W_alpha", None))
    if w_alpha is None:
        return None
    return float((2.0 * w_alpha) / ((2.0 ** bits) - 1.0))


def bias_contribution_report(module, output_eps: Optional[float]):
    bias = getattr(module, "bias", None)
    if bias is None or output_eps in (None, 0.0):
        return None
    stored = getattr(module, "_deploy_bias_integerization_report", None)
    if isinstance(stored, dict):
        bias_semantic = np.asarray(
            tensor_to_numpy(stored.get("bias_original_semantic")),
            dtype=np.float64,
        ).reshape(-1)
        effective_counts = np.asarray(
            tensor_to_numpy(stored.get("bias_integer_counts")),
            dtype=np.float64,
        ).reshape(-1)
        reconstructed = np.asarray(
            tensor_to_numpy(stored.get("bias_reconstructed_semantic")),
            dtype=np.float64,
        ).reshape(-1)
        rounding_error = np.asarray(
            tensor_to_numpy(stored.get("bias_rounding_error")),
            dtype=np.float64,
        ).reshape(-1)
        return {
            "bias_semantic_stats": tensor_stats(bias_semantic),
            "bias_effective_output_counts_stats": tensor_stats(effective_counts),
            "bias_reconstructed_semantic_stats": tensor_stats(reconstructed),
            "bias_rounding_error_stats": tensor_stats(rounding_error),
            "scale_source": stored.get("scale_source"),
            "rounding_mode": stored.get("rounding_mode"),
            "output_eps": float(stored.get("output_eps") or output_eps),
        }

    bias_arr = np.asarray(tensor_to_numpy(bias), dtype=np.float64).reshape(-1)
    if getattr(module, "_deploy_bias_integerized", False):
        effective_counts = bias_arr.copy()
        bias_arr = bias_arr * float(output_eps)
    else:
        effective_counts = bias_arr / float(output_eps)
    return {
        "bias_semantic_stats": tensor_stats(bias_arr),
        "bias_effective_output_counts_stats": tensor_stats(effective_counts),
        "output_eps": float(output_eps),
    }


def stage4_1_path_quant_context(model):
    skip_module = resolve_dotted_module(model, "stage4.0.out_relu")
    conv1_module = resolve_dotted_module(model, "stage4.1.conv1")
    relu1_module = resolve_dotted_module(model, "stage4.1.relu1")
    conv2_module = resolve_dotted_module(model, "stage4.1.conv2")
    add_module = resolve_dotted_module(model, "stage4.1.add")
    add_module_class = add_module.__class__.__name__
    add_is_passthrough = add_module_class in {"PassthroughAdd", "Identity"}

    add_eps_in_list = [
        float(scalar_from_value(value))
        for value in getattr(add_module, "eps_in_list", [])
        if scalar_from_value(value) is not None
    ]

    skip_eps_report = module_output_eps_with_source(skip_module)
    skip_eps_out = None if skip_eps_report is None else float(skip_eps_report["value"])
    skip_eps_out_source = None if skip_eps_report is None else skip_eps_report["source"]
    if skip_eps_out is None and len(add_eps_in_list) >= 2:
        skip_eps_out = float(add_eps_in_list[1])
        skip_eps_out_source = "stage4.1.add.eps_in_list[1]_fallback"
    conv1_eps_in = skip_eps_out
    conv1_eps_out_report = module_output_eps_with_source(conv1_module, conv1_eps_in)
    conv1_eps_out = None if conv1_eps_out_report is None else float(conv1_eps_out_report["value"])
    relu1_eps_out_report = module_output_eps_with_source(relu1_module, conv1_eps_out)
    relu1_eps_out = None if relu1_eps_out_report is None else float(relu1_eps_out_report["value"])
    conv2_eps_in = relu1_eps_out
    conv2_eps_out_report = module_output_eps_with_source(conv2_module, conv2_eps_in)
    conv2_eps_out = None if conv2_eps_out_report is None else float(conv2_eps_out_report["value"])
    conv2_eps_out_source = None if conv2_eps_out_report is None else conv2_eps_out_report["source"]
    if conv2_eps_out is None and len(add_eps_in_list) >= 1:
        conv2_eps_out = float(add_eps_in_list[0])
        conv2_eps_out_source = "stage4.1.add.eps_in_list[0]_fallback"
    if not add_eps_in_list and add_is_passthrough:
        passthrough_eps = conv2_eps_out if conv2_eps_out is not None else relu1_eps_out
        if passthrough_eps is not None:
            add_eps_in_list = [float(passthrough_eps)]

    skip_bits = precision_bits_from_value(getattr(skip_module, "precision", None))
    skip_alpha = tensor_scalar(getattr(skip_module, "alpha", None))
    skip_alpha_out = tensor_scalar(getattr(skip_module, "alpha_out", None))
    skip_max_semantic = None
    if skip_alpha is not None:
        skip_max_semantic = float(skip_alpha)
    elif skip_alpha_out is not None and skip_eps_out is not None:
        skip_max_semantic = float(skip_alpha_out) * float(skip_eps_out)

    def conv_report(module, eps_in, eps_out):
        return {
            "module_class": module.__class__.__name__,
            "eps_in": eps_in,
            "eps_out": eps_out,
            "eps_in_source": "upstream_module_explicit_mapping",
            "eps_out_source": (
                None
                if module_output_eps_with_source(module, eps_in) is None
                else module_output_eps_with_source(module, eps_in)["source"]
            ),
            "weight_eps": module_weight_eps(module),
            "bias_report": bias_contribution_report(module, eps_out),
            "equivalent_requant": None,
        }

    add_eps_out = tensor_scalar(getattr(add_module, "eps_out", None))
    if add_eps_out is None and add_is_passthrough and add_eps_in_list:
        add_eps_out = float(add_eps_in_list[0])

    return {
        "stage4.1.conv1_input": {
            "module_class": skip_module.__class__.__name__,
            "eps_in": tensor_scalar(getattr(skip_module, "eps_in", None)),
            "eps_out": skip_eps_out,
            "eps_out_source": skip_eps_out_source,
            "precision_bits": skip_bits,
            "alpha": skip_alpha,
            "alpha_out": skip_alpha_out,
            "semantic_clip_bounds": {
                "min": 0.0,
                "max": skip_max_semantic,
            },
        },
        "skip_path_input": {
            "module_class": skip_module.__class__.__name__,
            "eps_in": tensor_scalar(getattr(skip_module, "eps_in", None)),
            "eps_out": skip_eps_out,
            "eps_out_source": skip_eps_out_source,
            "precision_bits": skip_bits,
            "alpha": skip_alpha,
            "alpha_out": skip_alpha_out,
            "semantic_clip_bounds": {
                "min": 0.0,
                "max": skip_max_semantic,
            },
        },
        "stage4.1.conv1": conv_report(conv1_module, conv1_eps_in, conv1_eps_out),
        "stage4.1.relu1": {
            "module_class": relu1_module.__class__.__name__,
            "eps_in": conv1_eps_out,
            "eps_out": relu1_eps_out,
            "eps_in_source": "stage4.1.conv1.eps_out",
            "eps_out_source": (
                None if relu1_eps_out_report is None else relu1_eps_out_report["source"]
            ),
            "precision_bits": precision_bits_from_value(getattr(relu1_module, "precision", None)),
            "alpha": tensor_scalar(getattr(relu1_module, "alpha", None)),
            "alpha_out": tensor_scalar(getattr(relu1_module, "alpha_out", None)),
            "D": tensor_scalar(getattr(relu1_module, "D", None)),
            "requantization_factor": getattr(relu1_module, "requantization_factor", None),
            "semantic_clip_bounds": {
                "min": 0.0,
                "max": tensor_scalar(getattr(relu1_module, "alpha", None)),
            },
        },
        "stage4.1.conv2_input": {
            "module_class": relu1_module.__class__.__name__,
            "eps_in": conv1_eps_out,
            "eps_out": relu1_eps_out,
            "eps_in_source": "stage4.1.conv1.eps_out",
            "eps_out_source": (
                None if relu1_eps_out_report is None else relu1_eps_out_report["source"]
            ),
            "precision_bits": precision_bits_from_value(getattr(relu1_module, "precision", None)),
            "alpha": tensor_scalar(getattr(relu1_module, "alpha", None)),
            "alpha_out": tensor_scalar(getattr(relu1_module, "alpha_out", None)),
            "D": tensor_scalar(getattr(relu1_module, "D", None)),
            "requantization_factor": getattr(relu1_module, "requantization_factor", None),
            "semantic_clip_bounds": {
                "min": 0.0,
                "max": tensor_scalar(getattr(relu1_module, "alpha", None)),
            },
        },
        "stage4.1.conv2": conv_report(conv2_module, conv2_eps_in, conv2_eps_out),
        "stage4.1.add": {
            "module_class": add_module_class,
            "eps_in_list": add_eps_in_list,
            "eps_out": add_eps_out,
            "eps_in_sources": {
                "main": "stage4.1.conv2.eps_out" if conv2_eps_out_report is not None else conv2_eps_out_source,
                "skip": (
                    "stage4.0.out_relu.eps_out" if skip_eps_report is not None else skip_eps_out_source
                ),
            },
        },
    }


def collect_module_quant_metadata(model, module_names):
    attr_names = (
        "eps_in",
        "eps_in_list",
        "eps_out",
        "eps_static",
        "alpha",
        "alpha_static",
        "alpha_out",
        "W_alpha",
        "D",
        "mul",
        "shift",
        "requantization_factor",
        "precision",
        "deployment",
        "integerized",
    )
    snapshots = {}
    for name in module_names:
        try:
            module = resolve_dotted_module(model, name)
        except (AttributeError, IndexError, KeyError):
            continue
        module_info = {"class_name": module.__class__.__name__}
        for attr_name in attr_names:
            if hasattr(module, attr_name):
                module_info[attr_name] = serialize_quant_value(getattr(module, attr_name))
        snapshots[name] = module_info
    return snapshots


def build_hybrid_follow_bn_calib_dict(model):
    calib_dict = {}
    if hasattr(model, "stem") and len(getattr(model, "stem")) >= 3:
        if hasattr(model.stem[1], "__class__") and hasattr(model.stem[2], "__class__"):
            calib_dict["stem.2"] = "stem.1"

    for stage_name in ("stage1", "stage2", "stage3", "stage4"):
        stage = getattr(model, stage_name, None)
        if stage is None:
            continue
        for block_idx in (0, 1):
            block = stage[block_idx]
            if hasattr(block, "bn1") and hasattr(block, "relu1"):
                calib_dict["{}.{}.relu1".format(stage_name, block_idx)] = "{}.{}.bn1".format(stage_name, block_idx)
            if hasattr(block, "bn2") and hasattr(block, "out_relu"):
                calib_dict["{}.{}.out_relu".format(stage_name, block_idx)] = "{}.{}.bn2".format(stage_name, block_idx)
            if getattr(block, "proj", None) is not None and len(block.proj) >= 2:
                calib_dict["{}.{}.out_relu".format(stage_name, block_idx)] = "{}.{}.proj.1".format(stage_name, block_idx)
    return calib_dict


def _first_existing_module_name(model, *candidate_names: str) -> str | None:
    for candidate_name in candidate_names:
        if not candidate_name:
            continue
        try:
            resolve_dotted_module(model, candidate_name)
            return candidate_name
        except (AttributeError, IndexError, KeyError):
            continue
    return None


def has_delayed_relu_stem(model) -> bool:
    stem = getattr(model, "stem", None)
    return isinstance(stem, DelayedActivationStem) or getattr(model, "stem_mode", None) == "delayed_relu"


def build_quant_native_follow_bn_overrides(model) -> tuple[OrderedDict[str, str], dict[str, str]]:
    bn_dict: OrderedDict[str, str] = OrderedDict()
    calib_dict: dict[str, str] = {}

    def register_linear_bn_pair(linear_name: str, bn_name: str, activation_name: str | None = None) -> None:
        if _first_existing_module_name(model, linear_name) is None:
            return
        if _first_existing_module_name(model, bn_name) is None:
            return
        bn_dict[linear_name] = bn_name
        resolved_activation = _first_existing_module_name(model, activation_name) if activation_name else None
        if resolved_activation is not None:
            calib_dict[bn_name] = resolved_activation

    stem = getattr(model, "stem", None)
    if isinstance(stem, DelayedActivationStem) or has_delayed_relu_stem(model):
        delayed_activation_name = _first_existing_module_name(
            model,
            "stem.post.relu",
            "stage1.downsample.relu",
            "stage1.main_conv1.relu",
        )
        register_linear_bn_pair("stem.pre.conv", "stem.pre.bn", delayed_activation_name)
        register_linear_bn_pair("stem.post.conv", "stem.post.bn", "stem.post.relu")
    else:
        register_linear_bn_pair("stem.conv", "stem.bn", "stem.relu")

    for stage_name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, stage_name, None)
        if isinstance(stage, StraightStage):
            register_linear_bn_pair(
                f"{stage_name}.downsample.conv",
                f"{stage_name}.downsample.bn",
                f"{stage_name}.downsample.relu",
            )
            register_linear_bn_pair(
                f"{stage_name}.refine.conv",
                f"{stage_name}.refine.bn",
                f"{stage_name}.refine.relu",
            )
        elif isinstance(stage, ResidualDownsampleStage):
            register_linear_bn_pair(
                f"{stage_name}.main_conv1.conv",
                f"{stage_name}.main_conv1.bn",
                f"{stage_name}.main_conv1.relu",
            )
            register_linear_bn_pair(
                f"{stage_name}.main_conv2.conv",
                f"{stage_name}.main_conv2.bn",
                f"{stage_name}.out_relu",
            )
            register_linear_bn_pair(
                f"{stage_name}.skip_proj.conv",
                f"{stage_name}.skip_proj.bn",
                f"{stage_name}.out_relu",
            )

    for refine_name in ("stage1_refine", "stage2_refine"):
        refine = getattr(model, refine_name, None)
        if isinstance(refine, ConvBNReLU):
            register_linear_bn_pair(
                f"{refine_name}.conv",
                f"{refine_name}.bn",
                f"{refine_name}.relu",
            )

    return bn_dict, calib_dict


def prepare_quant_native_follow_qd(model, calib_samples=None) -> dict[str, Any]:
    if not has_delayed_relu_stem(model):
        return {}
    if getattr(model, "_export_nemo_quant_delayed_stem_qd_ready", False):
        cached_kwargs = getattr(model, "_export_nemo_quant_delayed_stem_qd_kwargs", None)
        return dict(cached_kwargs or {})

    override_bn_dict, override_calib_dict = build_quant_native_follow_bn_overrides(model)
    if not override_bn_dict:
        return {}

    if hasattr(model, "prune_empty_bn"):
        original_prune_empty_bn = model.prune_empty_bn

        def patched_prune_empty_bn(self, bn_dict=None, threshold=None):
            active_bn_dict = bn_dict or override_bn_dict
            return original_prune_empty_bn(bn_dict=active_bn_dict, threshold=threshold)

        model.prune_empty_bn = MethodType(patched_prune_empty_bn, model)

    if hasattr(model, "calibrate_bn"):
        original_calibrate_bn = model.calibrate_bn

        def patched_calibrate_bn(self, *args, **kwargs):
            if not kwargs.get("calib_dict"):
                kwargs["calib_dict"] = override_calib_dict
            return original_calibrate_bn(*args, **kwargs)

        model.calibrate_bn = MethodType(patched_calibrate_bn, model)

    qd_kwargs: dict[str, Any] = {}
    if calib_samples is not None:
        def bn_calibration_fn():
            with torch.no_grad():
                for sample in calib_samples:
                    _ = model(sample["tensor"])

        qd_kwargs["bn_calibration_fn"] = bn_calibration_fn

    model._export_nemo_quant_delayed_stem_qd_ready = True
    model._export_nemo_quant_delayed_stem_bn_dict = override_bn_dict
    model._export_nemo_quant_delayed_stem_calib_dict = override_calib_dict
    model._export_nemo_quant_delayed_stem_qd_kwargs = dict(qd_kwargs)
    return dict(qd_kwargs)


def extract_numeric_list(serialized_value: Any):
    if serialized_value is None:
        return []
    if isinstance(serialized_value, dict) and serialized_value.get("type") == "tensor":
        values = []
        for item in serialized_value.get("values", []):
            if isinstance(item, (int, float)):
                values.append(float(item))
        return values
    if isinstance(serialized_value, list):
        values = []
        for item in serialized_value:
            values.extend(extract_numeric_list(item))
        return values
    if isinstance(serialized_value, (int, float)):
        return [float(serialized_value)]
    return []


def analyze_hybrid_follow_scale_metadata(metadata_snapshots):
    issues = []
    for stage_name, stage_snapshot in metadata_snapshots.items():
        add_metadata = stage_snapshot.get("stage4.1.add")
        if not add_metadata:
            continue

        d_values = extract_numeric_list(add_metadata.get("D"))
        if d_values:
            shift_values = []
            for value in d_values:
                if value <= 0:
                    continue
                if abs(value - round(value)) >= 1e-6:
                    continue
                rounded = int(round(value))
                if rounded <= 0:
                    continue
                if (rounded & (rounded - 1)) == 0:
                    shift_values.append(int(np.log2(rounded)))
            if max(abs(value) for value in d_values) >= 65536.0 or any(shift >= 16 for shift in shift_values):
                issues.append(
                    {
                        "stage": stage_name,
                        "module": "stage4.1.add",
                        "issue": "large_divisor",
                        "values": d_values,
                        "shift": shift_values,
                    }
                )

        eps_out_values = extract_numeric_list(add_metadata.get("eps_out"))
        eps_in_values = extract_numeric_list(add_metadata.get("eps_in_list")) or extract_numeric_list(add_metadata.get("eps_in"))
        if not eps_in_values or not eps_out_values:
            continue

        eps_in_min = max(min(eps_in_values), 1e-12)
        eps_ratio = max(eps_out_values) / eps_in_min
        if eps_ratio >= 2048.0:
            issues.append(
                {
                    "stage": stage_name,
                    "module": "stage4.1.add",
                    "issue": "eps_out_ratio_large",
                    "eps_in": eps_in_values,
                    "eps_out": eps_out_values,
                    "ratio": eps_ratio,
                }
            )

        branch_ratio = max(eps_in_values) / eps_in_min
        if branch_ratio >= 128.0:
            issues.append(
                {
                    "stage": stage_name,
                    "module": "stage4.1.add",
                    "issue": "branch_eps_mismatch",
                    "eps_in": eps_in_values,
                    "ratio": branch_ratio,
                }
            )
    return issues


def compare_arrays(left: Any, right: Any):
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.shape != right_arr.shape:
        return {
            "shape_mismatch": {
                "left": list(left_arr.shape),
                "right": list(right_arr.shape),
            }
        }
    diff = np.abs(left_arr - right_arr)
    return {
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "sum_abs_diff": float(np.sum(diff)) if diff.size else 0.0,
    }


def compare_arrays_rich(left: Any, right: Any):
    base = compare_arrays(left, right)
    if "shape_mismatch" in base:
        return base

    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    left_abs_mean = float(np.mean(np.abs(left_arr))) if left_arr.size else 0.0
    right_abs_mean = float(np.mean(np.abs(right_arr))) if right_arr.size else 0.0
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    cosine_similarity = None
    if denom > 0.0:
        cosine_similarity = float(np.dot(left_arr, right_arr) / denom)
    base.update(
        {
            "left_abs_mean": left_abs_mean,
            "right_abs_mean": right_abs_mean,
            "abs_mean_ratio": right_abs_mean / max(left_abs_mean, 1e-12),
            "cosine_similarity": cosine_similarity,
        }
    )
    return base


def hybrid_follow_output_to_decoded(output: Any, stage: str):
    semantic = semantic_output(output, stage)
    if semantic.size != 3:
        return None
    vis_logit = float(semantic[2])
    return {
        "x_offset": float(semantic[0]),
        "size_proxy": float(semantic[1]),
        "visibility_logit": vis_logit,
        "visibility_confidence": float(1.0 / (1.0 + np.exp(-vis_logit))),
    }


def compare_decoded_hybrid_follow_outputs(
    left_output: Any,
    left_stage: str,
    right_output: Any,
    right_stage: str,
):
    left_decoded = hybrid_follow_output_to_decoded(left_output, left_stage)
    right_decoded = hybrid_follow_output_to_decoded(right_output, right_stage)
    if left_decoded is None or right_decoded is None:
        return None
    return {
        "x_abs_diff": abs(left_decoded["x_offset"] - right_decoded["x_offset"]),
        "size_abs_diff": abs(left_decoded["size_proxy"] - right_decoded["size_proxy"]),
        "vis_logit_abs_diff": abs(
            left_decoded["visibility_logit"] - right_decoded["visibility_logit"]
        ),
        "vis_conf_abs_diff": abs(
            left_decoded["visibility_confidence"] - right_decoded["visibility_confidence"]
        ),
        "left": left_decoded,
        "right": right_decoded,
    }


def semantic_output(output, stage: str):
    arr = np.asarray(output, dtype=np.float64).reshape(-1)
    if stage == "id":
        return arr / 32768.0
    return arr


def summarize_variant_drift(fp_output, quant_output, onnx_output, exported_stage):
    fp_semantic = semantic_output(fp_output, "fp")
    quant_semantic = semantic_output(quant_output, exported_stage)
    onnx_semantic = semantic_output(onnx_output, exported_stage)
    return {
        "fp_vs_quantized": compare_arrays(fp_semantic, quant_semantic),
        "quantized_vs_onnx_raw": compare_arrays(quant_output, onnx_output),
        "quantized_vs_onnx_semantic": compare_arrays(quant_semantic, onnx_semantic),
    }


def callable_source_location(fn):
    try:
        source_path = inspect.getsourcefile(fn) or inspect.getfile(fn)
        source_lines, start_line = inspect.getsourcelines(fn)
        return {
            "path": str(Path(source_path).resolve()) if source_path else None,
            "start_line": int(start_line),
            "end_line": int(start_line + len(source_lines) - 1),
        }
    except Exception as exc:
        return {"error": "{}: {}".format(type(exc).__name__, exc)}


def pact_integer_add_code_paths():
    import nemo.quant.pact as pact_mod
    import nemo.transf.deploy as deploy_mod
    import nemo.transform as transform_mod

    return {
        "deploy_qd_stage": callable_source_location(deploy_mod._qd_stage),
        "deploy_set_eps_in": callable_source_location(deploy_mod._set_eps_in_pact),
        "integerize_pact": callable_source_location(transform_mod.integerize_pact),
        "hier_integerizer": callable_source_location(transform_mod._hier_integerizer),
        "pact_integer_add_get_output_eps": callable_source_location(pact_mod.PACT_IntegerAdd.get_output_eps),
        "pact_integer_add_forward": callable_source_location(pact_mod.PACT_IntegerAdd.forward),
        "pact_integer_requantize_add": callable_source_location(pact_mod.pact_integer_requantize_add),
    }


def scalar_from_value(value: Any):
    if value is None:
        return None
    arr = np.asarray(tensor_to_numpy(value)).reshape(-1)
    if arr.size == 0:
        return None
    item = arr[0]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, (int, np.integer)):
        return int(item)
    return float(item)


def integer_add_branch_names(branch_count: int):
    if branch_count <= 0:
        return []
    if branch_count == 1:
        return ["main"]
    labels = ["main", "skip"]
    while len(labels) < branch_count:
        labels.append("branch{}".format(len(labels)))
    return labels[:branch_count]


def make_json_ready(value: Any):
    if isinstance(value, dict):
        return {key: make_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def compute_integer_add_audit(module, inputs, output):
    branch_inputs = [np.asarray(tensor_to_numpy(value)) for value in inputs]
    branch_labels = integer_add_branch_names(len(branch_inputs))
    eps_in_list = [float(scalar_from_value(value)) for value in getattr(module, "eps_in_list", [])]
    eps_out = scalar_from_value(getattr(module, "eps_out", None))
    alpha_out = scalar_from_value(getattr(module, "alpha_out", None))
    divisor = scalar_from_value(getattr(module, "D", None))
    if divisor is not None:
        divisor = int(divisor)
    requantization_factor = getattr(module, "requantization_factor", None)
    if requantization_factor is not None:
        requantization_factor = int(requantization_factor)
    precision = getattr(module, "precision", None)
    precision_bits = precision.get_bits() if precision is not None and hasattr(precision, "get_bits") else None
    deployment = bool(getattr(module, "deployment", False))
    integerized = bool(getattr(module, "integerized", False))
    actual_output = np.asarray(tensor_to_numpy(output))

    branch_reports = {}
    semantic_tensors = {}
    raw_int_inputs = []
    for index, (label, tensor) in enumerate(zip(branch_labels, branch_inputs)):
        eps_in = eps_in_list[index] if index < len(eps_in_list) else None
        raw_int = np.rint(tensor).astype(np.int64)
        raw_int_inputs.append(raw_int)
        branch_reports[label] = {
            "eps_in": eps_in,
            "raw_stats": tensor_stats(tensor),
            "rounded_int_stats": tensor_stats(raw_int),
        }
        if eps_in is not None:
            semantic = raw_int.astype(np.float64) * eps_in
            semantic_tensors["{}_semantic".format(label)] = semantic
            branch_reports[label]["semantic_stats"] = tensor_stats(semantic)

    scale_report = {
        "precision_bits": precision_bits,
        "deployment": deployment,
        "integerized": integerized,
        "requantization_factor": requantization_factor,
        "eps_in_list": eps_in_list,
        "eps_out": eps_out,
        "alpha_out": alpha_out,
        "D": divisor,
        "shift": None,
        "forward_uses_requantization": bool(deployment and integerized),
        "forward_clamps_output": False,
        "policy_override": getattr(module, "_export_nemo_quant_scale_override", None),
    }

    predicted_pre_raw = None
    predicted_post_raw = None
    predicted_pre_semantic = None
    predicted_post_semantic = None

    if eps_out is not None and divisor not in (None, 0) and len(eps_in_list) >= len(raw_int_inputs):
        if divisor > 0 and (divisor & (divisor - 1)) == 0:
            scale_report["shift"] = int(np.log2(divisor))
        mul_ideal = [float(divisor * eps_in / eps_out) for eps_in in eps_in_list[: len(raw_int_inputs)]]
        mul = [int(round(value)) for value in mul_ideal]
        output_lsb_per_input_lsb = [
            (float(value) / float(divisor)) if divisor and value else 0.0
            for value in mul
        ]
        input_lsb_per_output_lsb = [
            (float(divisor) / float(value)) if value else None
            for value in mul
        ]
        predicted_pre_raw = np.zeros_like(raw_int_inputs[0], dtype=np.int64)
        for raw_int, value in zip(raw_int_inputs, mul):
            predicted_pre_raw = predicted_pre_raw + (raw_int.astype(np.int64) * np.int64(value))
        predicted_post_raw = np.floor(predicted_pre_raw.astype(np.float64) / float(divisor)).astype(np.int64)
        predicted_pre_semantic = predicted_pre_raw.astype(np.float64) * float(eps_out) / float(divisor)
        predicted_post_semantic = predicted_post_raw.astype(np.float64) * float(eps_out)
        semantic_tensors["pre_requant_semantic"] = predicted_pre_semantic
        semantic_tensors["post_requant_semantic"] = predicted_post_semantic
        scale_report.update(
            {
                "mul": mul,
                "mul_ideal": mul_ideal,
                "output_lsb_per_input_lsb": output_lsb_per_input_lsb,
                "input_lsb_per_output_lsb": input_lsb_per_output_lsb,
                "branch_eps_ratio": (
                    float(max(eps_in_list) / max(min(eps_in_list), 1e-12))
                    if eps_in_list
                    else None
                ),
                "output_dynamic_range_semantic": (
                    float(alpha_out) * float(eps_out)
                    if alpha_out is not None and eps_out is not None
                    else None
                ),
                "clip_bounds_semantic_equivalent": {
                    "min": 0.0,
                    "max": (
                        float(alpha_out) * float(eps_out)
                        if alpha_out is not None and eps_out is not None
                        else None
                    ),
                },
            }
        )

    actual_output_raw = np.rint(actual_output).astype(np.int64)
    actual_report = {
        "raw_stats": tensor_stats(actual_output),
        "rounded_int_stats": tensor_stats(actual_output_raw),
    }
    if eps_out is not None:
        actual_semantic = actual_output_raw.astype(np.float64) * float(eps_out)
        semantic_tensors["actual_output_semantic"] = actual_semantic
        actual_report["semantic_stats"] = tensor_stats(actual_semantic)
    if predicted_post_raw is not None:
        actual_report["predicted_post_requant_raw_stats"] = tensor_stats(predicted_post_raw)
        actual_report["predicted_post_requant_semantic_stats"] = tensor_stats(predicted_post_semantic)
        actual_report["actual_vs_predicted_post_requant_raw"] = compare_arrays(
            actual_output_raw,
            predicted_post_raw,
        )
        actual_report["matches_predicted_post_requant_raw"] = bool(
            np.array_equal(actual_output_raw, predicted_post_raw)
        )

    if predicted_post_semantic is not None and scale_report.get("output_dynamic_range_semantic") is not None:
        observed_post_abs_max = float(tensor_stats(predicted_post_semantic)["abs_max"])
        if observed_post_abs_max > 0.0:
            scale_report["dynamic_range_margin_factor"] = (
                float(scale_report["output_dynamic_range_semantic"]) / observed_post_abs_max
            )

    report = {
        "module_class": module.__class__.__name__,
        "scale_selection": make_json_ready(scale_report),
        "branch_inputs": make_json_ready(branch_reports),
        "pre_requant": (
            {
                "raw_stats": tensor_stats(predicted_pre_raw),
                "semantic_stats": tensor_stats(predicted_pre_semantic),
            }
            if predicted_pre_raw is not None and predicted_pre_semantic is not None
            else None
        ),
        "post_requant": (
            {
                "raw_stats": tensor_stats(predicted_post_raw),
                "semantic_stats": tensor_stats(predicted_post_semantic),
            }
            if predicted_post_raw is not None and predicted_post_semantic is not None
            else None
        ),
        "actual_output": make_json_ready(actual_report),
    }
    return {
        "report": report,
        "tensors": {
            **semantic_tensors,
            "actual_output_raw": actual_output_raw,
            "predicted_pre_raw": predicted_pre_raw,
            "predicted_post_raw": predicted_post_raw,
        },
    }


def run_hybrid_follow_integer_add_audit(model, input_tensor):
    captures = {}
    handles = []
    code_path = pact_integer_add_code_paths()

    def make_hook(module_name):
        def hook(module, inputs, output):
            payload = compute_integer_add_audit(module, inputs, output)
            payload["report"]["module_name"] = module_name
            payload["report"]["code_path"] = code_path
            captures[module_name] = payload
        return hook

    for module_name in ("stage4.0.add", "stage4.1.add"):
        try:
            module = resolve_dotted_module(model, module_name)
        except (AttributeError, IndexError, KeyError):
            continue
        handles.append(module.register_forward_hook(make_hook(module_name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for handle in handles:
        handle.remove()

    return {
        "reports": {name: payload["report"] for name, payload in captures.items()},
        "tensors": {name: payload["tensors"] for name, payload in captures.items()},
    }


def semantic_tensor_for_stage(stage_tag: str, tensor: Any):
    arr = np.asarray(tensor_to_numpy(tensor), dtype=np.float64)
    if stage_tag == "id":
        return arr / 32768.0
    return arr


def build_hybrid_follow_residual_focus_report(fp_probe, fq_probe, deploy_probe, id_add_audit, head_eps_in):
    stage4_1_add = id_add_audit["tensors"].get("stage4.1.add", {})
    if not stage4_1_add:
        return None

    id_global_pool_raw = np.asarray(deploy_probe["tensors"]["global_pool_post_requant"], dtype=np.float64)
    id_head_input_raw = np.asarray(deploy_probe["tensors"]["head_input"], dtype=np.float64)
    id_pool_semantic = id_global_pool_raw * float(head_eps_in)
    id_head_input_semantic = id_head_input_raw * float(head_eps_in)

    point_specs = [
        (
            "stage4.1.conv2 output",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["stage4_1_conv2"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["stage4_1_conv2"]),
            np.asarray(stage4_1_add["main_semantic"], dtype=np.float64),
        ),
        (
            "stage4.1 residual skip input",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["stage4_1_add_input1"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["stage4_1_add_input1"]),
            np.asarray(stage4_1_add["skip_semantic"], dtype=np.float64),
        ),
        (
            "stage4.1.add pre-requant",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["stage4_1_add"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["stage4_1_add"]),
            np.asarray(stage4_1_add["pre_requant_semantic"], dtype=np.float64),
        ),
        (
            "stage4.1.add post-requant",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["stage4_1_add"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["stage4_1_add"]),
            np.asarray(stage4_1_add["post_requant_semantic"], dtype=np.float64),
        ),
        (
            "global pool output",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["global_pool_post_requant"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["global_pool_post_requant"]),
            id_pool_semantic,
        ),
        (
            "head input",
            semantic_tensor_for_stage("fp", fp_probe["tensors"]["head_input"]),
            semantic_tensor_for_stage("fq", fq_probe["tensors"]["head_input"]),
            id_head_input_semantic,
        ),
    ]

    points = []
    for label, fp_tensor, fq_tensor, id_tensor in point_specs:
        points.append(
            {
                "point": label,
                "fp_stats": tensor_stats(fp_tensor),
                "fq_stats": tensor_stats(fq_tensor),
                "id_stats": tensor_stats(id_tensor),
                "fp_vs_fq": compare_arrays_rich(fp_tensor, fq_tensor),
                "fq_vs_id": compare_arrays_rich(fq_tensor, id_tensor),
                "fp_vs_id": compare_arrays_rich(fp_tensor, id_tensor),
            }
        )

    strongest_point = None
    strongest_metric = -1.0
    for point in points:
        metric = float(point["fq_vs_id"]["mean_abs_diff"])
        if metric > strongest_metric:
            strongest_metric = metric
            strongest_point = point

    pre_post_quantization_loss = compare_arrays_rich(
        np.asarray(stage4_1_add["pre_requant_semantic"], dtype=np.float64),
        np.asarray(stage4_1_add["post_requant_semantic"], dtype=np.float64),
    )
    id_scale = id_add_audit["reports"]["stage4.1.add"]["scale_selection"]
    diagnosis = (
        "Largest inspected FQ->ID drift is at '{point}' with mean_abs_diff={mean_abs_diff:.6f}. "
        "The stage4.1.add requant step alone changes the semantic tensor by "
        "mean_abs_diff={requant_mean_abs_diff:.6f}. "
        "Branch output LSB per input LSB is {branch_lsb} with input LSB per output LSB {input_lsb}."
    ).format(
        point=strongest_point["point"] if strongest_point is not None else "unknown",
        mean_abs_diff=strongest_metric if strongest_metric >= 0.0 else 0.0,
        requant_mean_abs_diff=float(pre_post_quantization_loss["mean_abs_diff"]),
        branch_lsb=id_scale.get("output_lsb_per_input_lsb"),
        input_lsb=id_scale.get("input_lsb_per_output_lsb"),
    )

    return {
        "points": points,
        "largest_fq_to_id_drift_point": strongest_point["point"] if strongest_point is not None else None,
        "largest_fq_to_id_drift_mean_abs_diff": strongest_metric if strongest_metric >= 0.0 else None,
        "stage4_1_add_pre_vs_post_requant": pre_post_quantization_loss,
        "stage4_1_add_scale_selection": id_scale,
        "stage4_1_add_code_path": id_add_audit["reports"]["stage4.1.add"].get("code_path"),
        "head_input_eps_in": float(head_eps_in),
        "diagnosis": diagnosis,
    }


def build_hybrid_follow_residual_upstream_report(
    fq_probe,
    deploy_probe,
    fq_quant_context,
    id_quant_context,
):
    point_specs = [
        ("stage4.1.conv1 input", "stage4_1_conv1_input", fq_quant_context.get("stage4.1.conv1_input"), id_quant_context.get("stage4.1.conv1_input")),
        ("stage4.1.conv1 output", "stage4_1_conv1", fq_quant_context.get("stage4.1.conv1"), id_quant_context.get("stage4.1.conv1")),
        ("stage4.1 activation between conv1 and conv2", "stage4_1_relu1", fq_quant_context.get("stage4.1.relu1"), id_quant_context.get("stage4.1.relu1")),
        ("stage4.1.conv2 input", "stage4_1_conv2_input", fq_quant_context.get("stage4.1.conv2_input"), id_quant_context.get("stage4.1.conv2_input")),
        ("stage4.1.conv2 output", "stage4_1_conv2", fq_quant_context.get("stage4.1.conv2"), id_quant_context.get("stage4.1.conv2")),
        ("stage4.1 residual skip input", "stage4_1_add_input1", fq_quant_context.get("skip_path_input"), id_quant_context.get("skip_path_input")),
    ]

    points = []
    for label, tensor_key, fq_ctx, id_ctx in point_specs:
        fq_raw = np.asarray(fq_probe["tensors"][tensor_key], dtype=np.float64)
        id_raw = np.asarray(deploy_probe["tensors"][tensor_key], dtype=np.float64)
        id_eps_out = None if id_ctx is None else id_ctx.get("eps_out")
        id_semantic = id_raw * float(id_eps_out) if id_eps_out not in (None, 0.0) else id_raw.copy()

        semantic_clip_bounds = None if id_ctx is None else id_ctx.get("semantic_clip_bounds")
        clip_min = None if semantic_clip_bounds is None else semantic_clip_bounds.get("min")
        clip_max = None if semantic_clip_bounds is None else semantic_clip_bounds.get("max")
        quantum = None if id_ctx is None else id_ctx.get("eps_out")

        id_abs_mean = mean_abs_value(id_semantic)
        fq_abs_mean = max(mean_abs_value(fq_raw), 1e-12)
        points.append(
            {
                "point": label,
                "fq_activation_stats": tensor_stats(fq_raw),
                "id_activation_stats_raw": tensor_stats(id_raw),
                "id_activation_stats_semantic": tensor_stats(id_semantic),
                "fq_vs_id": compare_arrays_rich(fq_raw, id_semantic),
                "id_to_fq_abs_mean_ratio": id_abs_mean / fq_abs_mean,
                "fq_quant_context": make_json_ready(fq_ctx),
                "id_quant_context": make_json_ready(id_ctx),
                "fq_saturation": {
                    "available": False,
                },
                "id_saturation": saturation_stats(
                    id_semantic,
                    min_value=clip_min,
                    max_value=clip_max,
                    quantum=quantum,
                ),
            }
        )

    point_by_name = {point["point"]: point for point in points}
    conv1_input_point = point_by_name.get("stage4.1.conv1 input")
    conv1_point = point_by_name.get("stage4.1.conv1 output")
    relu1_point = point_by_name.get("stage4.1 activation between conv1 and conv2")
    conv2_input_point = point_by_name.get("stage4.1.conv2 input")
    conv2_point = point_by_name.get("stage4.1.conv2 output")
    skip_point = point_by_name.get("stage4.1 residual skip input")

    diagnosis = (
        "Along the main branch before the residual add, stage4.1.conv1 input fq_vs_id mean_abs_diff={conv1_input_diff:.6f}, "
        "stage4.1.conv1 output fq_vs_id mean_abs_diff={conv1_diff:.6f} "
        "(abs-mean ratio={conv1_ratio:.6f}), stage4.1.conv2 fq_vs_id mean_abs_diff={conv2_diff:.6f} "
        "(abs-mean ratio={conv2_ratio:.6f}), and the skip path fq_vs_id mean_abs_diff={skip_diff:.6f}. "
        "That means the main branch is already {main_branch_state} before stage4.1.add."
    ).format(
        conv1_input_diff=float((conv1_input_point or {}).get("fq_vs_id", {}).get("mean_abs_diff") or 0.0),
        conv1_diff=float((conv1_point or {}).get("fq_vs_id", {}).get("mean_abs_diff") or 0.0),
        conv1_ratio=float((conv1_point or {}).get("id_to_fq_abs_mean_ratio") or 0.0),
        conv2_diff=float((conv2_point or {}).get("fq_vs_id", {}).get("mean_abs_diff") or 0.0),
        conv2_ratio=float((conv2_point or {}).get("id_to_fq_abs_mean_ratio") or 0.0),
        skip_diff=float((skip_point or {}).get("fq_vs_id", {}).get("mean_abs_diff") or 0.0),
        main_branch_state=(
            "attenuated and distorted"
            if float((conv2_point or {}).get("id_to_fq_abs_mean_ratio") or 1.0) < 0.75
            else "distorted more than the skip path"
        ),
    )

    return {
        "points": points,
        "diagnosis": diagnosis,
    }


def residual_audit_markdown(audit_payload):
    lines = [
        "# Hybrid Follow Integer Add Audit",
        "",
        "- Integer add scale policy: `{}`".format(HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY),
        "",
        "## Scale Selection",
    ]
    for stage_name in ("post_qd", "post_id"):
        stage_report = audit_payload.get(stage_name, {})
        lines.append("")
        lines.append("### {}".format(stage_name))
        for module_name in ("stage4.0.add", "stage4.1.add"):
            report = stage_report.get(module_name)
            if not report:
                continue
            scale = report["scale_selection"]
            code_path = report.get("code_path", {})
            lines.extend(
                [
                    "- {}: eps_in={} eps_out={} D={} shift={} mul={} requantization_factor={}".format(
                        module_name,
                        scale.get("eps_in_list"),
                        scale.get("eps_out"),
                        scale.get("D"),
                        scale.get("shift"),
                        scale.get("mul"),
                        scale.get("requantization_factor"),
                    ),
                    "  branch_eps_ratio={} output_lsb_per_input_lsb={} input_lsb_per_output_lsb={} forward_uses_requantization={}".format(
                        scale.get("branch_eps_ratio"),
                        scale.get("output_lsb_per_input_lsb"),
                        scale.get("input_lsb_per_output_lsb"),
                        scale.get("forward_uses_requantization"),
                    ),
                    "  get_output_eps={} forward={} requant={}".format(
                        code_path.get("pact_integer_add_get_output_eps"),
                        code_path.get("pact_integer_add_forward"),
                        code_path.get("pact_integer_requantize_add"),
                    ),
                ]
            )
    return "\n".join(lines) + "\n"


def residual_focus_markdown(focus_report):
    scale = focus_report.get("stage4_1_add_scale_selection") or {}
    lines = [
        "# Hybrid Follow Residual Drift Focus",
        "",
        "- Largest FQ->ID drift point: `{}`".format(
            focus_report.get("largest_fq_to_id_drift_point")
        ),
        "- Largest FQ->ID drift mean abs diff: `{:.6f}`".format(
            float(focus_report.get("largest_fq_to_id_drift_mean_abs_diff") or 0.0)
        ),
        "- stage4.1.add pre->post requant mean abs diff: `{:.6f}`".format(
            float(focus_report["stage4_1_add_pre_vs_post_requant"]["mean_abs_diff"])
        ),
        "- stage4.1.add eps_in={} eps_out={} D={} shift={} mul={}".format(
            scale.get("eps_in_list"),
            scale.get("eps_out"),
            scale.get("D"),
            scale.get("shift"),
            scale.get("mul"),
        ),
        "",
        focus_report["diagnosis"],
    ]
    return "\n".join(lines) + "\n"


def residual_upstream_markdown(upstream_report):
    lines = [
        "# Hybrid Follow Residual Upstream Report",
        "",
        upstream_report["diagnosis"],
        "",
    ]
    for point in upstream_report.get("points", []):
        fq_vs_id = point.get("fq_vs_id") or {}
        id_ctx = point.get("id_quant_context") or {}
        bias_report = id_ctx.get("bias_report") or {}
        lines.extend(
            [
                "## {}".format(point["point"]),
                "",
                "- fq_vs_id mean_abs_diff=`{:.6f}` max_abs_diff=`{:.6f}`".format(
                    float(fq_vs_id.get("mean_abs_diff") or 0.0),
                    float(fq_vs_id.get("max_abs_diff") or 0.0),
                ),
                "- id_to_fq_abs_mean_ratio=`{:.6f}`".format(
                    float(point.get("id_to_fq_abs_mean_ratio") or 0.0)
                ),
                "- id eps_in=`{}` eps_out=`{}`".format(
                    id_ctx.get("eps_in"),
                    id_ctx.get("eps_out"),
                ),
                "- fq eps_in=`{}` eps_out=`{}`".format(
                    (point.get("fq_quant_context") or {}).get("eps_in"),
                    (point.get("fq_quant_context") or {}).get("eps_out"),
                ),
                "- id scale context: weight_eps=`{}` D=`{}` requantization_factor=`{}`".format(
                    id_ctx.get("weight_eps"),
                    id_ctx.get("D"),
                    id_ctx.get("requantization_factor"),
                ),
                "- fq scale context: weight_eps=`{}` D=`{}` requantization_factor=`{}`".format(
                    (point.get("fq_quant_context") or {}).get("weight_eps"),
                    (point.get("fq_quant_context") or {}).get("D"),
                    (point.get("fq_quant_context") or {}).get("requantization_factor"),
                ),
                "- id activation semantic stats: `{}`".format(point.get("id_activation_stats_semantic")),
                "- fq activation stats: `{}`".format(point.get("fq_activation_stats")),
                "- id saturation: `{}`".format(point.get("id_saturation")),
            ]
        )
        if bias_report:
            lines.append("- bias effective output counts stats: `{}`".format(bias_report.get("bias_effective_output_counts_stats")))
        else:
            lines.append("- bias effective output counts stats: `None`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_hybrid_follow_head_input_eps(model):
    head_module_name = "head" if hasattr(model, "head") else "head_x"
    head_module = resolve_dotted_module(model, head_module_name)
    head_eps_in = scalar_from_value(getattr(head_module, "eps_in", None))
    if head_eps_in is not None:
        return float(head_eps_in)
    fallback_modules = (
        ("stage4.1.out_relu", ("eps_out", "eps_in")),
        ("stage4.1.add", ("eps_out", "eps_in")),
        ("stage4.1.conv2", ("eps_out", "eps_in")),
        ("global_pool", ("eps_out", "eps_in")),
    )
    for module_name, attr_names in fallback_modules:
        try:
            module = resolve_dotted_module(model, module_name)
        except (AttributeError, IndexError, KeyError):
            continue
        for attr_name in attr_names:
            value = scalar_from_value(getattr(module, attr_name, None))
            if value is not None:
                return float(value)
    try:
        inferred_context = stage4_1_path_quant_context(model)
    except Exception:
        inferred_context = {}
    for context_name, field_names in (
        ("stage4.1.out_relu", ("eps_out", "eps_in")),
        ("stage4.1.add", ("eps_out",)),
        ("stage4.1.conv2", ("eps_out", "eps_in")),
        ("global_pool", ("eps_out", "eps_in")),
    ):
        context = inferred_context.get(context_name) or {}
        for field_name in field_names:
            value = scalar_from_value(context.get(field_name))
            if value is not None:
                return float(value)
    return None


def build_integer_add_policy_trial(
    *,
    args,
    device,
    calib_samples,
    debug_input,
    policy,
):
    active_policy = getattr(args, "integer_add_scale_policy", HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY)

    model_fp = prepare_model_fp(args, device)
    fp_probe = run_hybrid_follow_pytorch_probe(model_fp, debug_input["float"])

    dummy_input = torch.randn(
        1,
        args.input_channels,
        args.height,
        args.width,
        device=device,
    )
    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
    model_q.to(device).eval()
    repair_hybrid_follow_fused_quant_graph(model_q)
    model_q.change_precision(bits=args.bits, scale_weights=True, scale_activations=True)
    run_activation_calibration(model_q, calib_samples)

    try:
        model_q.reset_alpha_weights()
    except Exception:
        pass

    fq_probe = run_hybrid_follow_pytorch_probe(model_q, debug_input["float"])
    with integer_add_scale_selection_scope(active_policy, {"stage4.1.add": policy}):
        model_q.qd_stage(eps_in=args.eps_in)
        repair_hybrid_follow_fused_quant_graph(model_q)
        qd_integer_add_audit = run_hybrid_follow_integer_add_audit(model_q, debug_input["staged"])

        model_q.id_stage()
        normalize_integer_requant_tensors(model_q)
        integerize_deploy_conv_biases(
            model_q,
            default_scale_source=getattr(args, "conv_bias_scale_source", HYBRID_FOLLOW_CONV_BIAS_SCALE_SOURCE),
            default_rounding_mode=getattr(args, "conv_bias_rounding", HYBRID_FOLLOW_CONV_BIAS_ROUNDING),
        )
        quant_probe = run_hybrid_follow_pytorch_probe(model_q, debug_input["staged"])
        id_integer_add_audit = run_hybrid_follow_integer_add_audit(model_q, debug_input["staged"])

    head_eps_in = resolve_hybrid_follow_head_input_eps(model_q)
    residual_focus = (
        build_hybrid_follow_residual_focus_report(
            fp_probe,
            fq_probe,
            quant_probe,
            id_integer_add_audit,
            head_eps_in,
        )
        if head_eps_in is not None
        else None
    )
    final_output_drift = compare_decoded_hybrid_follow_outputs(
        fq_probe["tensors"]["model_output"],
        "fq",
        quant_probe["tensors"]["model_output"],
        "id",
    )

    point_summary = {}
    if residual_focus is not None:
        for point in residual_focus["points"]:
            point_summary[point["point"]] = point["fq_vs_id"]

    stage4_1_scale = id_integer_add_audit["reports"]["stage4.1.add"]["scale_selection"]
    score = None
    if final_output_drift is not None:
        score = (
            float(final_output_drift["x_abs_diff"])
            + float(final_output_drift["size_abs_diff"])
            + float(final_output_drift["vis_conf_abs_diff"])
        )

    return {
        "policy": policy,
        "score_final_output": score,
        "final_output_drift": final_output_drift,
        "largest_fq_to_id_drift_point": (
            residual_focus["largest_fq_to_id_drift_point"] if residual_focus is not None else None
        ),
        "largest_fq_to_id_drift_mean_abs_diff": (
            residual_focus["largest_fq_to_id_drift_mean_abs_diff"] if residual_focus is not None else None
        ),
        "stage4_1_add_pre_vs_post_requant": (
            residual_focus["stage4_1_add_pre_vs_post_requant"]
            if residual_focus is not None
            else None
        ),
        "point_drift_summary": point_summary,
        "stage4_1_scale_selection": stage4_1_scale,
        "stage4_0_scale_selection": id_integer_add_audit["reports"]["stage4.0.add"]["scale_selection"],
        "qd_stage4_1_scale_selection": qd_integer_add_audit["reports"]["stage4.1.add"]["scale_selection"],
        "id_output_decoded": hybrid_follow_output_to_decoded(
            quant_probe["tensors"]["model_output"],
            "id",
        ),
    }


def run_integer_add_policy_sweep(
    *,
    args,
    device,
    calib_samples,
    debug_input,
    candidate_policies=HYBRID_FOLLOW_INTEGER_ADD_POLICY_CANDIDATES,
):
    active_policy = getattr(args, "integer_add_scale_policy", HYBRID_FOLLOW_INTEGER_ADD_SCALE_POLICY)
    policy_reports = {}
    candidate_order = []
    for policy in candidate_policies:
        candidate_order.append(policy)
        try:
            policy_reports[policy] = build_integer_add_policy_trial(
                args=args,
                device=device,
                calib_samples=calib_samples,
                debug_input=debug_input,
                policy=policy,
            )
        except Exception as exc:
            policy_reports[policy] = {
                "policy": policy,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    scored_reports = [
        report
        for report in policy_reports.values()
        if report.get("score_final_output") is not None
    ]
    selected_policy = active_policy
    if scored_reports:
        selected_policy = min(
            scored_reports,
            key=lambda report: float(report["score_final_output"]),
        )["policy"]

    active_report = policy_reports.get(active_policy)
    selected_report = policy_reports.get(selected_policy)
    return {
        "active_policy": active_policy,
        "candidate_order": candidate_order,
        "selected_policy": selected_policy,
        "policies": policy_reports,
        "selection_metric": "score_final_output",
        "selection_metric_description": (
            "x_abs_diff + size_abs_diff + vis_conf_abs_diff on the known sample"
        ),
        "active_policy_score": (
            active_report.get("score_final_output") if active_report is not None else None
        ),
        "selected_policy_score": (
            selected_report.get("score_final_output") if selected_report is not None else None
        ),
    }


def integer_add_policy_sweep_markdown(policy_sweep):
    lines = [
        "# Hybrid Follow Integer Add Policy Sweep",
        "",
        "- Active policy: `{}`".format(policy_sweep.get("active_policy")),
        "- Selected policy: `{}`".format(policy_sweep.get("selected_policy")),
        "- Selection metric: `score_final_output = x_abs_diff + size_abs_diff + vis_conf_abs_diff`",
        "",
    ]
    for policy in policy_sweep.get("candidate_order", []):
        report = policy_sweep["policies"].get(policy)
        if report is None:
            continue
        if report.get("error"):
            lines.extend(
                [
                    "## {}".format(policy),
                    "",
                    "- error: `{}`".format(report["error"]),
                    "",
                ]
            )
            continue
        final_drift = report.get("final_output_drift") or {}
        lines.extend(
            [
                "## {}".format(policy),
                "",
                "- final score: `{:.6f}`".format(float(report.get("score_final_output") or 0.0)),
                "- final drift: x=`{:.6f}` size=`{:.6f}` vis_conf=`{:.6f}`".format(
                    float(final_drift.get("x_abs_diff") or 0.0),
                    float(final_drift.get("size_abs_diff") or 0.0),
                    float(final_drift.get("vis_conf_abs_diff") or 0.0),
                ),
                "- largest FQ->ID drift point: `{}` (`{:.6f}`)".format(
                    report.get("largest_fq_to_id_drift_point"),
                    float(report.get("largest_fq_to_id_drift_mean_abs_diff") or 0.0),
                ),
                "- stage4.1.add: eps_out=`{}` D=`{}` mul=`{}`".format(
                    report["stage4_1_scale_selection"].get("eps_out"),
                    report["stage4_1_scale_selection"].get("D"),
                    report["stage4_1_scale_selection"].get("mul"),
                ),
            ]
        )
        point_summary = report.get("point_drift_summary") or {}
        for point_name in (
            "stage4.1.conv2 output",
            "stage4.1 residual skip input",
            "stage4.1.add pre-requant",
            "stage4.1.add post-requant",
            "global pool output",
            "head input",
        ):
            point_report = point_summary.get(point_name)
            if point_report is None:
                continue
            lines.append(
                "- {}: fq_vs_id mean_abs_diff=`{:.6f}` max_abs_diff=`{:.6f}`".format(
                    point_name,
                    float(point_report.get("mean_abs_diff") or 0.0),
                    float(point_report.get("max_abs_diff") or 0.0),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_hybrid_follow_onnx_probe_model(onnx_path: Path, probe_path: Path, selected_output_names):
    import onnx

    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)
    value_infos = {}
    for value_info in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        value_infos[value_info.name] = value_info

    existing_outputs = {output.name for output in model.graph.output}
    for output_name in selected_output_names:
        if output_name in existing_outputs:
            continue
        if output_name in value_infos:
            model.graph.output.append(deepcopy(value_infos[output_name]))
        else:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    output_name,
                    onnx.TensorProto.FLOAT,
                    None,
                )
            )
    onnx.save(model, str(probe_path))


def run_hybrid_follow_pytorch_probe(model, input_tensor):
    captures = {}
    handles = []

    def capture_output(alias):
        def hook(_module, _inputs, output):
            captures[alias] = output.detach().cpu()
        return hook

    def capture_input(alias):
        def hook(_module, inputs):
            if inputs and alias not in captures:
                captures[alias] = inputs[0].detach().cpu()
        return hook

    def capture_add(alias):
        def hook(_module, inputs, output):
            if len(inputs) >= 1:
                captures["{}_input0".format(alias)] = inputs[0].detach().cpu()
            if len(inputs) >= 2:
                captures["{}_input1".format(alias)] = inputs[1].detach().cpu()
            captures[alias] = output.detach().cpu()
        return hook

    hook_specs = [
        ("stage4.0.add", "stage4_0_add", "add"),
        ("stage4.0.out_relu", "stage4_0_out_relu", "output"),
        ("stage4.1.conv1", "stage4_1_conv1_input", "input"),
        ("stage4.1.conv1", "stage4_1_conv1", "output"),
        ("stage4.1.relu1", "stage4_1_relu1", "output"),
        ("stage4.1.conv2", "stage4_1_conv2_input", "input"),
        ("stage4.1.conv2", "stage4_1_conv2", "output"),
        ("stage4.1.add", "stage4_1_add", "add"),
        ("stage4.1.out_relu", "stage4_1_out_relu", "output"),
        ("global_pool", "global_pool_post_requant", "output"),
    ]
    head_input_module = "head" if hasattr(model, "head") else "head_x"
    hook_specs.append((head_input_module, "head_input", "input"))

    for module_name, alias, mode in hook_specs:
        try:
            module = resolve_dotted_module(model, module_name)
        except (AttributeError, IndexError, KeyError):
            continue
        if mode == "output":
            handles.append(module.register_forward_hook(capture_output(alias)))
        elif mode == "add":
            handles.append(module.register_forward_hook(capture_add(alias)))
        else:
            handles.append(module.register_forward_pre_hook(capture_input(alias)))

    with torch.no_grad():
        output = model(input_tensor)

    for handle in handles:
        handle.remove()

    captures["model_output"] = output.detach().cpu()
    return {
        "tensors": {name: tensor_to_numpy(value) for name, value in captures.items()},
        "stats": {name: tensor_stats(value) for name, value in captures.items()},
    }


def run_hybrid_follow_onnx_probe(onnx_path: Path, input_tensor, probe_dir: Path):
    import onnx
    import onnxruntime as ort

    probe_dir.mkdir(parents=True, exist_ok=True)
    selected_outputs = {
        "stage4_0_out_relu": "/stage4/stage4.0/out_relu/Clip_output_0",
        "stage4_1_conv2": "/stage4/stage4.1/conv2/Conv_output_0",
        "stage4_1_add_pre_requant": "/stage4/stage4.1/add/Add_output_0",
        "stage4_1_add_post_requant": "/stage4/stage4.1/add/Floor_output_0",
        "stage4_1_out_relu": "/stage4/stage4.1/out_relu/Clip_output_0",
        "global_pool_pre_requant": "/global_pool/AveragePool_output_0",
        "global_pool_post_requant": "/global_pool/Floor_output_0",
        "head_input": "/Flatten_output_0",
    }

    model = onnx.load(str(onnx_path))
    available_outputs = {
        output_name
        for node in model.graph.node
        for output_name in node.output
    }
    selected_outputs = {
        alias: output_name
        for alias, output_name in selected_outputs.items()
        if output_name in available_outputs
    }

    probe_path = probe_dir / "{}_probe.onnx".format(onnx_path.stem)
    build_hybrid_follow_onnx_probe_model(onnx_path, probe_path, selected_outputs.values())

    session = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(
        output_names,
        {session.get_inputs()[0].name: tensor_to_numpy(input_tensor)},
    )
    output_map = dict(zip(output_names, output_values))

    captures = {}
    for alias, output_name in selected_outputs.items():
        captures[alias] = np.asarray(output_map[output_name])
    captures["model_output"] = np.asarray(output_values[0])

    return {
        "probe_model_path": str(probe_path),
        "tensors": captures,
        "stats": {name: tensor_stats(value) for name, value in captures.items()},
    }


def detect_zero_collapse(stats_map: Dict[str, Any], ordered_names):
    previous_name = None
    previous_stats = None
    for name in ordered_names:
        stats = stats_map.get(name)
        if stats is None:
            continue
        if previous_stats is not None:
            if previous_stats["nonzero_count"] > 0 and stats["nonzero_count"] == 0:
                return {
                    "location": name,
                    "previous_location": previous_name,
                }
        previous_name = name
        previous_stats = stats
    return None


def detect_pre_requant_collapse(stats_map: Dict[str, Any], prefix: str):
    input0 = stats_map.get("{}_input0".format(prefix))
    input1 = stats_map.get("{}_input1".format(prefix))
    output = stats_map.get(prefix)
    if output is None:
        return None

    input_nonzero = 0
    if input0 is not None:
        input_nonzero += int(input0["nonzero_count"])
    if input1 is not None:
        input_nonzero += int(input1["nonzero_count"])

    if input_nonzero > 0 and output["nonzero_count"] == 0:
        return {
            "location": prefix,
            "previous_location": "{}_inputs".format(prefix),
        }
    return None


def run_hybrid_follow_variant_debug(
    args,
    device,
    calib_samples,
    debug_input,
    variant_name,
    *,
    fuse,
    collapse,
    output_dir: Path,
):
    variant_report = {
        "variant": variant_name,
        "fusion_enabled": fuse,
        "head_collapse_enabled": collapse,
        "status": "ok",
    }

    try:
        model_fp = prepare_model_fp(
            args,
            device,
            fuse_hybrid_follow=fuse,
            collapse_hybrid_follow_heads=collapse,
        )
        with torch.no_grad():
            fp_output = model_fp(debug_input["float"]).detach().cpu().numpy()

        dummy_input = torch.randn(
            1,
            args.input_channels,
            args.height,
            args.width,
            device=device,
        )
        model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
        model_q.to(device).eval()
        repair_hybrid_follow_fused_quant_graph(model_q)
        model_q.change_precision(bits=args.bits, scale_weights=True, scale_activations=True)
        run_activation_calibration(model_q, calib_samples)

        if args.stage in {"qd", "id"}:
            try:
                model_q.reset_alpha_weights()
            except Exception:
                pass

        exported_stage = args.stage
        if args.stage in {"qd", "id"}:
            qd_kwargs = {}
            if not fuse:
                qd_kwargs["calib_dict"] = build_hybrid_follow_bn_calib_dict(model_fp)
            model_q.qd_stage(eps_in=args.eps_in, **qd_kwargs)
            repair_hybrid_follow_fused_quant_graph(model_q)
            exported_stage = "qd"
        if args.stage == "id":
            model_q.id_stage()
            normalize_integer_requant_tensors(model_q)
            exported_stage = "id"

        deploy_input = debug_input["staged"] if exported_stage in {"qd", "id"} else debug_input["float"]
        quant_probe = run_hybrid_follow_pytorch_probe(model_q, deploy_input)

        model_q.eval()
        for param in model_q.parameters():
            param.requires_grad_(False)

        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "{}.onnx".format(variant_name)
        nemo.utils.export_onnx(
            str(onnx_path),
            model_q,
            model_q,
            (args.input_channels, args.height, args.width),
            round_params=args.round_export_params,
            batch_size=1,
        )

        onnx_probe = run_hybrid_follow_onnx_probe(onnx_path, deploy_input, output_dir / "probe")
        variant_report["exported_stage"] = exported_stage
        variant_report["drift"] = summarize_variant_drift(
            fp_output=fp_output,
            quant_output=quant_probe["tensors"]["model_output"],
            onnx_output=onnx_probe["tensors"]["model_output"],
            exported_stage=exported_stage,
        )
        variant_report["first_bad_location"] = detect_zero_collapse(
            onnx_probe["stats"],
            HYBRID_FOLLOW_COLLAPSE_ORDER,
        )
        variant_report["quantized_stats"] = quant_probe["stats"]
        variant_report["onnx_stats"] = onnx_probe["stats"]
        variant_report["onnx_path"] = str(onnx_path)
        return variant_report
    except Exception as exc:
        variant_report["status"] = "error"
        variant_report["error"] = "{}: {}".format(type(exc).__name__, exc)
        return variant_report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export SSD or hybrid_follow models to ONNX using FP or NEMO FQ/QD/ID stages."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hybrid_follow",
        choices=["ssd", "hybrid_follow", "plain_follow", "plain_follow_v2", "plain_follow_tiny", "dronet_lite_follow"],
    )
    parser.add_argument("--ckpt", type=str, default="training/person_ssd_pytorch/ssd_mbv2_raw.pth")
    parser.add_argument("--out", type=str, default="export/ssd_mbv2_nemo_id.onnx")

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument(
        "--follow-head-type",
        type=str,
        default=None,
        help="Output contract for quant-native follow models. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--stage4-variant", type=str, default=None,
                        help="Optional hybrid_follow late-stage variant override. Defaults to checkpoint metadata.")
    parser.add_argument("--opset-version", type=int, default=13)

    parser.add_argument("--bits", type=int, default=8, help="Quantization bits (like Q in notebook)")
    parser.add_argument(
        "--eps-in",
        type=float,
        default=1.0 / 255.0,
        help="Input quantum eps_in. For images in [0,1], use 1/255.",
    )

    parser.add_argument("--stage", choices=["fp", "fq", "qd", "id"], default="fp",
                        help="Which stage to export (fp/fq/qd/id).")
    parser.add_argument("--strict-stage", action="store_true",
                        help="Retained for compatibility; requested quantized stages now fail if conversion errors out.")
    parser.add_argument("--stage-report", type=str, default=None,
                        help="Optional path to write the final exported stage (fq/qd/id).")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory of calibration images (jpg/png).")
    parser.add_argument("--calib-manifest", type=str, default=None,
                        help="Optional JSON calibration manifest with an ordered subset of deployment-matched images.")
    parser.add_argument("--calib-tensor", type=str, default=None,
                        help="Path to a .pt tensor file shaped [N,C,H,W] for calibration.")
    parser.add_argument("--calib-batches", type=int, default=64,
                        help="How many samples to use for activation calibration.")
    parser.add_argument("--calib-seed", type=int, default=0,
                        help="Shuffle seed used when sampling calibration images from a directory.")
    parser.add_argument("--mean", type=str, default=None,
                        help="Optional normalization mean, e.g. '0.5' (C=1) or '0.5,0.5,0.5' (C=3)")
    parser.add_argument("--std", type=str, default=None,
                        help="Optional normalization std, e.g. '0.5' (C=1) or '0.5,0.5,0.5' (C=3)")
    parser.add_argument("--disable-conv-bn-fusion", action="store_true",
                        help="Skip Conv-BN fusion for hybrid_follow export/debug comparisons.")
    parser.add_argument("--disable-hybrid-follow-head-collapse", action="store_true",
                        help="Keep the original three hybrid_follow heads instead of collapsing to one FC layer.")
    parser.add_argument("--debug-quant-drift-dir", type=str, default=None,
                        help="Optional directory for strict hybrid_follow quantized export drift debugging artifacts.")
    parser.add_argument("--clamp-dory-weights", action="store_true",
                        help="Clamp Conv/Gemm/MatMul ONNX weight initializers into signed int8 range after export.")
    parser.add_argument("--round-export-params", action="store_true",
                        help="Opt in to NEMO export-time parameter rounding. Disabled by default because it can collapse hybrid_follow ID exports before deployment.")
    parser.add_argument(
        "--hybrid-follow-export-preset",
        type=str,
        default=HYBRID_FOLLOW_EXPORT_PRESET,
        choices=list(HYBRID_FOLLOW_EXPORT_PRESET_CANDIDATES),
        help=(
            "Named hybrid_follow export preset. "
            "Use baseline for current defaults or microblock_add_only for the promoted "
            "stage4.1.add activation/scale patch."
        ),
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = ExportRequest.from_namespace(parser.parse_args(argv))

    if is_follow_model_type(args.model_type) and args.input_channels != 1:
        raise ValueError(f"{args.model_type} export requires --input-channels 1.")

    device = (
        torch.device("cpu")
        if args.force_cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[export_nemo_quant] Using device: {device}")
    print(
        f"[export_nemo_quant] Input tensor config: "
        f"C={args.input_channels}, H={args.height}, W={args.width}"
    )

    image_size = (args.height, args.width)
    if args.stage != "fp" and patch_model_to_graph_compat():
        print(
            "[export_nemo_quant] Applied _model_to_graph compatibility shim for older torch versions."
        )

    debug_dir = None
    if args.debug_quant_drift_dir:
        debug_dir = Path(args.debug_quant_drift_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    calib_samples = []
    calibration_artifact = None
    if args.stage != "fp" or debug_dir is not None:
        calib_samples = collect_calib_samples(args, image_size, device)
        calibration_artifact = summarize_calibration_samples(
            calib_samples,
            calib_dir=args.calib_dir,
            calib_manifest=getattr(args, "calib_manifest", None),
            calib_tensor=args.calib_tensor,
            calib_batches=args.calib_batches,
            calib_seed=args.calib_seed,
        )

    debug_input = None
    debug_report = {
        "requested_stage": args.stage,
        "exported_stage": None,
        "selected_input": None,
        "calibration": calibration_artifact,
        "first_bad_location": None,
        "tensor_stats": {},
        "suspect_scale_metadata": [],
        "fusion_head_collapse_effect": None,
        "clamp_changed_result": None,
        "round_export_params": bool(args.round_export_params),
        "integer_add_scale_policy": args.integer_add_scale_policy,
        "hybrid_follow_export_preset": args.hybrid_follow_export_preset,
    }

    if debug_dir is not None and args.model_type == "hybrid_follow":
        debug_input = resolve_quant_debug_input(calib_samples)
        debug_report["selected_input"] = {
            "source": debug_input["source"],
            "float_stats": save_debug_tensor(debug_dir, "selected_input_float", debug_input["float"]),
            "staged_stats": save_debug_tensor(debug_dir, "selected_input_staged_0_255", debug_input["staged"]),
        }
        write_json(debug_dir / "debug_export_config.json", vars(args))

    model_fp = prepare_model_fp(args, device)
    dummy_input = torch.randn(1, args.input_channels, args.height, args.width, device=device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_artifact is not None:
        calibration_artifact["artifact_path"] = str(
            out_path.with_name(f"{out_path.stem}_calibration.json")
        )
        write_json(Path(calibration_artifact["artifact_path"]), calibration_artifact)

    if args.stage == "fp":
        print(
            f"[export_nemo_quant] Exporting FP model to ONNX:\n"
            f"  -> {out_path}\n"
            f"  input_shape=(1,{args.input_channels},{args.height},{args.width})"
        )
        export_fp_onnx(
            model=model_fp,
            dummy_input=dummy_input,
            out_path=out_path,
            opset_version=args.opset_version,
        )

        if debug_dir is not None and args.model_type == "hybrid_follow" and debug_input is not None:
            fp_probe = run_hybrid_follow_pytorch_probe(model_fp, debug_input["float"])
            fp_probe_dir = debug_dir / "pytorch_fp"
            fp_probe_dir.mkdir(parents=True, exist_ok=True)
            fp_stats = {}
            for name, value in fp_probe["tensors"].items():
                fp_stats[name] = save_debug_tensor(fp_probe_dir, name, value)
            debug_report["tensor_stats"]["pytorch_fp"] = fp_stats
            write_stats_summary(debug_dir / "tensor_stats.txt", debug_report["tensor_stats"])
            write_json(debug_dir / "debug_report.json", debug_report)

        if args.stage_report:
            stage_path = Path(args.stage_report)
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_path.write_text("fp\n", encoding="utf-8")
        print("[export_nemo_quant] Final exported stage: FP")
        print("[export_nemo_quant] Done.")
        return

    if args.model_type == "hybrid_follow":
        print("[export_nemo_quant] hybrid_follow quantized export requested.")
        print(
            "[export_nemo_quant] hybrid_follow export preset: "
            f"{args.hybrid_follow_export_preset}"
        )

    print("[export_nemo_quant] Building FakeQuantized (FQ) model via quantize_pact...")
    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
    model_q.to(device).eval()
    if args.model_type == "hybrid_follow":
        repair_hybrid_follow_fused_quant_graph(model_q)

    print(f"[export_nemo_quant] Setting precision to {args.bits} bits...")
    model_q.change_precision(bits=args.bits, scale_weights=True, scale_activations=True)

    print("[export_nemo_quant] Calibrating activations with statistics_act() ...")
    run_activation_calibration(model_q, calib_samples)
    preset_config = None
    preset_report = None
    if args.model_type == "hybrid_follow":
        preset_config = derive_hybrid_follow_export_preset_config(
            model_q,
            calib_samples,
            args.hybrid_follow_export_preset,
        )
        preset_report = apply_hybrid_follow_export_preset_config(model_q, preset_config)
    if args.stage in {"qd", "id"}:
        try:
            model_q.reset_alpha_weights()
        except Exception:
            pass

    model_deploy = model_q
    exported_stage = args.stage
    metadata_snapshots = {}
    qd_integer_add_audit = None
    id_integer_add_audit = None
    fq_stage4_1_quant_context = None
    id_stage4_1_quant_context = None
    integerized_conv_biases = []

    fp_probe = None
    fq_probe = None
    if debug_dir is not None and args.model_type == "hybrid_follow" and debug_input is not None:
        fp_probe = run_hybrid_follow_pytorch_probe(model_fp, debug_input["float"])
        fp_probe_dir = debug_dir / "pytorch_fp"
        fp_probe_dir.mkdir(parents=True, exist_ok=True)
        debug_report["tensor_stats"]["pytorch_fp"] = {}
        for name, value in fp_probe["tensors"].items():
            debug_report["tensor_stats"]["pytorch_fp"][name] = save_debug_tensor(fp_probe_dir, name, value)
        metadata_snapshots["pre_qd"] = collect_module_quant_metadata(model_deploy, HYBRID_FOLLOW_METADATA_MODULES)
        fq_stage4_1_quant_context = stage4_1_path_quant_context(model_deploy)
        fq_probe = run_hybrid_follow_pytorch_probe(model_q, debug_input["float"])
        fq_probe_dir = debug_dir / "pytorch_fq"
        fq_probe_dir.mkdir(parents=True, exist_ok=True)
        debug_report["tensor_stats"]["pytorch_fq"] = {}
        for name, value in fq_probe["tensors"].items():
            debug_report["tensor_stats"]["pytorch_fq"][name] = save_debug_tensor(fq_probe_dir, name, value)
        debug_report["hybrid_follow_export_preset_report"] = make_json_ready(
            getattr(model_q, "_export_nemo_quant_preset_report", preset_report)
        )
    integer_add_scope_overrides = (
        (preset_report or {}).get("integer_add_operator_overrides")
        if args.model_type == "hybrid_follow"
        else None
    )
    with integer_add_scale_selection_scope(
        args.integer_add_scale_policy,
        integer_add_scope_overrides,
    ):
        if args.stage in {"qd", "id"}:
            qd_kwargs = {}
            if args.model_type == "hybrid_follow" and args.disable_conv_bn_fusion:
                qd_kwargs["calib_dict"] = build_hybrid_follow_bn_calib_dict(model_fp)
            elif is_quant_native_follow_model_type(args.model_type):
                qd_kwargs.update(prepare_quant_native_follow_qd(model_deploy, calib_samples=calib_samples))

            print(f"[export_nemo_quant] Entering QuantizedDeployable (QD) via qd_stage(eps_in={args.eps_in}) ...")
            if args.model_type == "hybrid_follow":
                repair_hybrid_follow_fused_quant_graph(model_deploy)
                model_deploy.qd_stage(eps_in=args.eps_in, **qd_kwargs)
                repair_hybrid_follow_fused_quant_graph(model_deploy)
            else:
                model_deploy.qd_stage(eps_in=args.eps_in, **qd_kwargs)

            if args.stage == "qd":
                exported_stage = "qd"

            if debug_dir is not None and args.model_type == "hybrid_follow":
                metadata_snapshots["post_qd"] = collect_module_quant_metadata(model_deploy, HYBRID_FOLLOW_METADATA_MODULES)
                qd_integer_add_audit = run_hybrid_follow_integer_add_audit(
                    model_deploy,
                    debug_input["staged"],
                )

        if args.stage == "id":
            print("[export_nemo_quant] Entering IntegerDeployable (ID) via id_stage() ...")
            model_deploy.id_stage()
            normalize_integer_requant_tensors(model_deploy)
            integerized_conv_biases = integerize_deploy_conv_biases(
                model_deploy,
                default_scale_source=args.conv_bias_scale_source,
                default_rounding_mode=args.conv_bias_rounding,
            )
            exported_stage = "id"

            if debug_dir is not None and args.model_type == "hybrid_follow":
                metadata_snapshots["post_id"] = collect_module_quant_metadata(model_deploy, HYBRID_FOLLOW_METADATA_MODULES)
                id_stage4_1_quant_context = stage4_1_path_quant_context(model_deploy)
                id_integer_add_audit = run_hybrid_follow_integer_add_audit(
                    model_deploy,
                    debug_input["staged"],
                )

        model_deploy.eval()
        for param in model_deploy.parameters():
            param.requires_grad_(False)

        quant_probe = None
        suspect_scale_metadata = []
        policy_sweep = None
        if debug_dir is not None and args.model_type == "hybrid_follow" and debug_input is not None:
            deploy_input = debug_input["staged"] if exported_stage in {"qd", "id"} else debug_input["float"]
            quant_probe = run_hybrid_follow_pytorch_probe(model_deploy, deploy_input)
            quant_probe_dir = debug_dir / "pytorch_quantized"
            quant_probe_dir.mkdir(parents=True, exist_ok=True)
            debug_report["tensor_stats"]["pytorch_quantized"] = {}
            for name, value in quant_probe["tensors"].items():
                debug_report["tensor_stats"]["pytorch_quantized"][name] = save_debug_tensor(quant_probe_dir, name, value)
            suspect_scale_metadata = analyze_hybrid_follow_scale_metadata(metadata_snapshots)
            debug_report["suspect_scale_metadata"] = suspect_scale_metadata

        input_shape = (args.input_channels, args.height, args.width)
        print(
            f"[export_nemo_quant] Exporting {exported_stage.upper()} model to ONNX:\n"
            f"  -> {out_path}\n"
            f"  input_shape=(1,{input_shape[0]},{input_shape[1]},{input_shape[2]})"
        )

        nemo.utils.export_onnx(
            str(out_path),
            model_deploy,
            model_deploy,
            input_shape,
            round_params=args.round_export_params,
            batch_size=1,
        )
    unclamped_weight_audit = report_dory_weight_initializer_ranges(out_path)
    clamped_weight_audit = None
    unclamped_debug_copy = None
    clamped_debug_copy = None

    if debug_dir is not None:
        unclamped_debug_copy = debug_dir / "{}_unclamped.onnx".format(out_path.stem)
        unclamped_debug_copy.write_bytes(out_path.read_bytes())

    if args.clamp_dory_weights:
        print("[export_nemo_quant] Clamping Conv/Gemm/MatMul initializers into signed int8 range ...")
        clamped_weight_audit = clamp_dory_weight_initializers_to_int8(out_path)
        report_dory_weight_initializer_ranges(out_path)
        if debug_dir is not None:
            clamped_debug_copy = debug_dir / "{}_clamped.onnx".format(out_path.stem)
            clamped_debug_copy.write_bytes(out_path.read_bytes())

    if args.stage_report:
        stage_path = Path(args.stage_report)
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(f"{exported_stage}\n", encoding="utf-8")

    debug_failure_reasons = []
    if debug_dir is not None and args.model_type == "hybrid_follow" and debug_input is not None:
        deploy_input = debug_input["staged"] if exported_stage in {"qd", "id"} else debug_input["float"]
        onnx_probe = run_hybrid_follow_onnx_probe(out_path, deploy_input, debug_dir / "onnx_probe")
        onnx_probe_dir = debug_dir / "onnx_probe_outputs"
        onnx_probe_dir.mkdir(parents=True, exist_ok=True)
        debug_report["tensor_stats"]["onnx"] = {}
        for name, value in onnx_probe["tensors"].items():
            debug_report["tensor_stats"]["onnx"][name] = save_debug_tensor(onnx_probe_dir, name, value)

        pytorch_collapse = None
        if quant_probe is not None:
            pytorch_collapse = detect_pre_requant_collapse(quant_probe["stats"], "stage4_1_add")
            if pytorch_collapse is None:
                pytorch_collapse = detect_zero_collapse(
                    quant_probe["stats"],
                    HYBRID_FOLLOW_PYTORCH_COLLAPSE_ORDER,
                )
        onnx_collapse = detect_zero_collapse(
            onnx_probe["stats"],
            HYBRID_FOLLOW_COLLAPSE_ORDER,
        )

        if pytorch_collapse is not None:
            debug_report["first_bad_location"] = "pytorch.{}".format(pytorch_collapse["location"])
            debug_failure_reasons.append(
                "PyTorch quantized activations collapse at {} after {}.".format(
                    pytorch_collapse["location"],
                    pytorch_collapse["previous_location"],
                )
            )
        elif onnx_collapse is not None:
            debug_report["first_bad_location"] = "onnx.{}".format(onnx_collapse["location"])
            debug_failure_reasons.append(
                "ONNX activations collapse at {} after {}.".format(
                    onnx_collapse["location"],
                    onnx_collapse["previous_location"],
                )
            )

        output_drift = {}
        if fp_probe is not None and fq_probe is not None:
            output_drift["fp_vs_fq"] = compare_decoded_hybrid_follow_outputs(
                fp_probe["tensors"]["model_output"],
                "fp",
                fq_probe["tensors"]["model_output"],
                "fq",
            )
        if fq_probe is not None and quant_probe is not None:
            output_drift["fq_vs_deploy"] = compare_decoded_hybrid_follow_outputs(
                fq_probe["tensors"]["model_output"],
                "fq",
                quant_probe["tensors"]["model_output"],
                exported_stage,
            )
        if fp_probe is not None and quant_probe is not None:
            output_drift["fp_vs_deploy"] = compare_decoded_hybrid_follow_outputs(
                fp_probe["tensors"]["model_output"],
                "fp",
                quant_probe["tensors"]["model_output"],
                exported_stage,
            )
        if quant_probe is not None:
            output_drift["deploy_vs_onnx_raw"] = compare_arrays(
                quant_probe["tensors"]["model_output"],
                onnx_probe["tensors"]["model_output"],
            )
            output_drift["deploy_vs_onnx_semantic"] = compare_arrays(
                semantic_output(quant_probe["tensors"]["model_output"], exported_stage),
                semantic_output(onnx_probe["tensors"]["model_output"], exported_stage),
            )
        debug_report["output_drift"] = output_drift

        integer_add_audit_reports = {}
        if qd_integer_add_audit is not None:
            integer_add_audit_reports["post_qd"] = qd_integer_add_audit["reports"]
        if id_integer_add_audit is not None:
            integer_add_audit_reports["post_id"] = id_integer_add_audit["reports"]
        debug_report["integer_add_audit"] = integer_add_audit_reports
        if integer_add_audit_reports:
            write_json(debug_dir / "integer_add_audit.json", integer_add_audit_reports)
            (debug_dir / "integer_add_audit.md").write_text(
                residual_audit_markdown(integer_add_audit_reports),
                encoding="utf-8",
            )

        residual_focus_report = None
        residual_upstream_report = None
        if (
            fp_probe is not None
            and fq_probe is not None
            and quant_probe is not None
            and id_integer_add_audit is not None
        ):
            head_eps_in = resolve_hybrid_follow_head_input_eps(model_deploy)
            if head_eps_in is not None:
                residual_focus_report = build_hybrid_follow_residual_focus_report(
                    fp_probe,
                    fq_probe,
                    quant_probe,
                    id_integer_add_audit,
                    head_eps_in,
                )
        if (
            fq_probe is not None
            and quant_probe is not None
            and fq_stage4_1_quant_context is not None
            and id_stage4_1_quant_context is not None
        ):
            residual_upstream_report = build_hybrid_follow_residual_upstream_report(
                fq_probe,
                quant_probe,
                fq_stage4_1_quant_context,
                id_stage4_1_quant_context,
            )
        debug_report["residual_focus_report"] = residual_focus_report
        if residual_focus_report is not None:
            write_json(debug_dir / "residual_focus_report.json", residual_focus_report)
            (debug_dir / "residual_focus_report.md").write_text(
                residual_focus_markdown(residual_focus_report),
                encoding="utf-8",
            )
        debug_report["residual_upstream_report"] = residual_upstream_report
        if residual_upstream_report is not None:
            write_json(debug_dir / "residual_upstream_report.json", residual_upstream_report)
            (debug_dir / "residual_upstream_report.md").write_text(
                residual_upstream_markdown(residual_upstream_report),
                encoding="utf-8",
            )

        if exported_stage == "id":
            policy_sweep = run_integer_add_policy_sweep(
                args=args,
                device=device,
                calib_samples=calib_samples,
                debug_input=debug_input,
            )
        debug_report["integer_add_policy_sweep"] = policy_sweep
        if policy_sweep is not None:
            write_json(debug_dir / "integer_add_policy_sweep.json", policy_sweep)
            (debug_dir / "integer_add_policy_sweep.md").write_text(
                integer_add_policy_sweep_markdown(policy_sweep),
                encoding="utf-8",
            )

        residual_scale_warning = None
        if suspect_scale_metadata:
            fatal_scale_issues = [
                item
                for item in suspect_scale_metadata
                if item.get("issue") in {"large_divisor", "eps_out_ratio_large"}
            ]
            if fatal_scale_issues:
                residual_scale_warning = {
                    "issues": fatal_scale_issues,
                    "fq_vs_deploy": output_drift.get("fq_vs_deploy"),
                    "fp_vs_deploy": output_drift.get("fp_vs_deploy"),
                    "deploy_vs_onnx_semantic": output_drift.get("deploy_vs_onnx_semantic"),
                }
                if pytorch_collapse is not None or onnx_collapse is not None:
                    residual_scale_warning["severity"] = "error"
                    debug_failure_reasons.append(
                        "Residual scale metadata around stage4.1.add coincides with activation collapse."
                    )
                else:
                    residual_scale_warning["severity"] = "warning"
                    residual_scale_warning["diagnosis"] = (
                        "NEMO deploy-stage drift is present before export, but ONNX still matches the "
                        "in-memory deploy graph."
                    )
        debug_report["residual_scale_warning"] = residual_scale_warning

        clamp_changed_result = None
        if args.clamp_dory_weights and unclamped_debug_copy is not None:
            unclamped_probe = run_hybrid_follow_onnx_probe(
                unclamped_debug_copy,
                deploy_input,
                debug_dir / "onnx_probe_unclamped",
            )
            clamp_diff = compare_arrays(
                unclamped_probe["tensors"]["model_output"],
                onnx_probe["tensors"]["model_output"],
            )
            clamp_changed_result = clamp_diff.get("max_abs_diff", 0.0) > 0.0
        debug_report["clamp_changed_result"] = clamp_changed_result

        variant_reports = {}
        if args.stage in {"qd", "id"}:
            variant_specs = (
                ("variant_a_current_fused_single_head", True, True),
                ("variant_b_unfused_single_head", False, True),
                ("variant_c_fused_three_heads", True, False),
            )
            for variant_name, fuse_enabled, collapse_enabled in variant_specs:
                variant_reports[variant_name] = run_hybrid_follow_variant_debug(
                    args,
                    device,
                    calib_samples,
                    debug_input,
                    variant_name,
                    fuse=fuse_enabled,
                    collapse=collapse_enabled,
                    output_dir=debug_dir / "variants" / variant_name,
                )

            bad_variants = [
                name
                for name, report in variant_reports.items()
                if report.get("status") == "ok" and report.get("first_bad_location") is not None
            ]
            error_variants = [
                name
                for name, report in variant_reports.items()
                if report.get("status") != "ok"
            ]
            if not bad_variants and not error_variants:
                debug_report["fusion_head_collapse_effect"] = "none_after_fix"
            elif not bad_variants and error_variants:
                debug_report["fusion_head_collapse_effect"] = (
                    "no_collapse_in_completed_variants;errors=" + ",".join(error_variants)
                )
            elif len(bad_variants) == len(variant_reports):
                debug_report["fusion_head_collapse_effect"] = "regardless"
            else:
                debug_report["fusion_head_collapse_effect"] = ",".join(bad_variants)
        else:
            variant_reports = {}

        debug_report["requested_stage"] = args.stage
        debug_report["exported_stage"] = exported_stage
        debug_report["metadata_snapshots"] = metadata_snapshots
        debug_report["variant_results"] = variant_reports
        debug_report["weight_initializer_audit"] = {
            "unclamped": unclamped_weight_audit,
            "clamped": clamped_weight_audit,
        }
        debug_report["integerized_conv_biases"] = integerized_conv_biases
        if onnx_collapse is not None:
            debug_report["diagnosis"] = "export_or_onnx_collapse"
        elif pytorch_collapse is not None:
            debug_report["diagnosis"] = "in_memory_quantized_collapse_before_export"
        elif (
            residual_scale_warning is not None
            and residual_scale_warning.get("severity") == "warning"
            and output_drift.get("deploy_vs_onnx_raw", {}).get("max_abs_diff", 1.0) == 0.0
        ):
            debug_report["diagnosis"] = "deploy_stage_residual_scale_warning_but_export_matches_deploy"
        else:
            debug_report["diagnosis"] = "no_collapse_detected"

        write_json(debug_dir / "debug_report.json", debug_report)
        write_stats_summary(debug_dir / "tensor_stats.txt", debug_report["tensor_stats"])
        summary_lines = [
            "# Hybrid Follow Quant Drift Debug Report",
            "",
            "- Requested stage: `{}`".format(args.stage),
            "- Exported stage: `{}`".format(exported_stage),
            "- Selected input: `{}`".format(debug_input["source"]),
            "- First bad location: `{}`".format(debug_report["first_bad_location"]),
            "- Suspect scale metadata count: `{}`".format(len(suspect_scale_metadata)),
            "- Fusion/head-collapse effect: `{}`".format(debug_report["fusion_head_collapse_effect"]),
            "- Clamp changed result: `{}`".format(debug_report["clamp_changed_result"]),
            "- Round export params: `{}`".format(debug_report["round_export_params"]),
            "- Integer add scale policy: `{}`".format(debug_report["integer_add_scale_policy"]),
            "- Diagnosis: `{}`".format(debug_report["diagnosis"]),
        ]
        fq_vs_deploy = output_drift.get("fq_vs_deploy")
        if fq_vs_deploy is not None:
            summary_lines.extend(
                [
                    "- FQ->deploy x abs diff: `{:.6f}`".format(fq_vs_deploy["x_abs_diff"]),
                    "- FQ->deploy size abs diff: `{:.6f}`".format(fq_vs_deploy["size_abs_diff"]),
                    "- FQ->deploy vis conf abs diff: `{:.6f}`".format(fq_vs_deploy["vis_conf_abs_diff"]),
                ]
            )
        deploy_vs_onnx = output_drift.get("deploy_vs_onnx_raw")
        if deploy_vs_onnx is not None:
            summary_lines.append(
                "- Deploy->ONNX raw max abs diff: `{:.6f}`".format(
                    deploy_vs_onnx["max_abs_diff"]
                )
            )
        if residual_scale_warning is not None:
            summary_lines.append(
                "- Residual scale warning severity: `{}`".format(
                    residual_scale_warning["severity"]
                )
            )
        if residual_focus_report is not None:
            summary_lines.extend(
                [
                    "- Largest FQ->ID drift point: `{}`".format(
                        residual_focus_report["largest_fq_to_id_drift_point"]
                    ),
                    "- stage4.1.add pre->post requant mean abs diff: `{:.6f}`".format(
                        float(
                            residual_focus_report["stage4_1_add_pre_vs_post_requant"][
                                "mean_abs_diff"
                            ]
                        )
                    ),
                ]
            )
        if residual_upstream_report is not None:
            point_by_name = {
                point["point"]: point
                for point in residual_upstream_report.get("points", [])
            }
            summary_lines.extend(
                [
                    "- stage4.1.conv1 fq->id mean abs diff: `{:.6f}`".format(
                        float(
                            point_by_name.get("stage4.1.conv1 output", {})
                            .get("fq_vs_id", {})
                            .get("mean_abs_diff", 0.0)
                        )
                    ),
                    "- stage4.1.conv2 fq->id mean abs diff: `{:.6f}`".format(
                        float(
                            point_by_name.get("stage4.1.conv2 output", {})
                            .get("fq_vs_id", {})
                            .get("mean_abs_diff", 0.0)
                        )
                    ),
                    "- stage4.1 skip fq->id mean abs diff: `{:.6f}`".format(
                        float(
                            point_by_name.get("stage4.1 residual skip input", {})
                            .get("fq_vs_id", {})
                            .get("mean_abs_diff", 0.0)
                        )
                    ),
                ]
            )
        if policy_sweep is not None:
            summary_lines.extend(
                [
                    "- Integerized deploy conv biases: `{}`".format(len(integerized_conv_biases)),
                    "- Integer add selected policy on known sample: `{}`".format(
                        policy_sweep["selected_policy"]
                    ),
                    "- Integer add active/selected score: `{}` -> `{}`".format(
                        policy_sweep.get("active_policy_score"),
                        policy_sweep.get("selected_policy_score"),
                    ),
                ]
            )
        (debug_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[export_nemo_quant] Final exported stage: {exported_stage.upper()} (requested: {args.stage.upper()})")
    if debug_failure_reasons:
        raise RuntimeError(" ; ".join(debug_failure_reasons))
    print("[export_nemo_quant] Done.")


if __name__ == "__main__":
    main()

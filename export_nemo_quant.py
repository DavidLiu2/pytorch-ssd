#!/usr/bin/env python3
from __future__ import annotations

import argparse
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import nemo  # pytorch-nemo (pulp-platform)
from PIL import Image

from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw


def build_model(num_classes: int, width_mult: float, image_size, input_channels: int):
    return SSDMobileNetV2Raw(
        num_classes=num_classes,
        width_mult=width_mult,
        image_size=image_size,
        input_channels=input_channels,
    )


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
    state = torch.load(ckpt_path, map_location=device)
    # common wrappers
    if isinstance(state, dict):
        for k in ["model", "state_dict", "net", "module"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
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


def debug_backbone_feature_paths(model, context: str):
    max_list_items = 12
    path_tokens = ("backbone.features", "backbone.stage")
    module_names = sorted(
        n for n, _ in model.named_modules() if any(tok in n for tok in path_tokens)
    )
    module_set = set(module_names)
    targets = [
        "ssd.backbone.features.0.1",
        "backbone.features.0.1",
        "ssd.backbone.stage0.0.1",
        "backbone.stage0.0.1",
    ]

    print(
        f"[export_nemo_quant][debug] {context}: "
        f"{len(module_names)} module names contain backbone feature/stage paths"
    )
    for name in module_names[:max_list_items]:
        print(f"[export_nemo_quant][debug] module: {name}")
    if len(module_names) > max_list_items:
        print(
            "[export_nemo_quant][debug] module list truncated: "
            f"showing {max_list_items}/{len(module_names)}"
        )
    for target in targets:
        print(
            f"[export_nemo_quant][debug] module exists '{target}': "
            f"{target in module_set}"
        )

    graph = getattr(model, "graph", None)
    if graph is None:
        print(f"[export_nemo_quant][debug] {context}: model.graph is None")
        return

    graph_names = sorted({
        name for name in graph.non_unique_names_dict.values()
        if any(tok in name for tok in path_tokens)
    })
    graph_set = set(graph_names)
    print(
        f"[export_nemo_quant][debug] {context}: "
        f"{len(graph_names)} graph scope names contain backbone feature/stage paths"
    )
    for name in graph_names[:max_list_items]:
        print(f"[export_nemo_quant][debug] graph: {name}")
    if len(graph_names) > max_list_items:
        print(
            "[export_nemo_quant][debug] graph list truncated: "
            f"showing {max_list_items}/{len(graph_names)}"
        )
    for target in targets:
        print(
            f"[export_nemo_quant][debug] graph scope exists '{target}': "
            f"{target in graph_set}"
        )


def is_qd_eps_mapping_error(exc: Exception) -> bool:
    msg = str(exc)
    if isinstance(exc, AttributeError) and "NoneType" in msg and "item" in msg:
        return True
    if isinstance(exc, KeyError):
        key = exc.args[0] if exc.args else ""
        if isinstance(key, str) and ("backbone." in key or "ssd.backbone" in key):
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
            if isinstance(old_list, list) and len(old_list) > 0:
                new_list = []
                for old_eps in old_list:
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


def image_to_tensor(
    path: Path,
    hw: tuple[int, int],
    device,
    input_channels: int,
    mean=None,
    std=None,
):
    mode = "L" if input_channels == 1 else "RGB"
    im = Image.open(path).convert(mode).resize((hw[1], hw[0]), resample=Image.BILINEAR)
    x_np = np.asarray(im, dtype=np.uint8)

    if input_channels == 1:
        x = torch.from_numpy(x_np).unsqueeze(0).contiguous().unsqueeze(0).to(device=device)
    else:
        x = torch.from_numpy(x_np).permute(2, 0, 1).contiguous().unsqueeze(0).to(device=device)
    x = x.float().div_(255.0)

    if mean is not None and std is not None:
        m = torch.tensor(mean, device=device).view(1, input_channels, 1, 1)
        s = torch.tensor(std, device=device).view(1, input_channels, 1, 1)
        x = (x - m) / s

    return x


def iter_calib_batches(args, image_size, device):
    hw = (image_size[0], image_size[1])

    # Option A: tensor file
    if args.calib_tensor:
        t = torch.load(args.calib_tensor, map_location="cpu")
        if isinstance(t, dict) and "data" in t:
            t = t["data"]
        assert isinstance(t, torch.Tensor), "calib_tensor must be a Tensor or dict with key 'data'"
        assert t.ndim == 4 and t.shape[1] == args.input_channels, (
            f"Expected [N,{args.input_channels},H,W], got {tuple(t.shape)}"
        )
        # If tensor resolution differs, user should pre-resize; we won't interpolate silently.
        for i in range(min(args.calib_batches, t.shape[0])):
            yield t[i:i+1].to(device=device, dtype=torch.float32)
        return

    # Option B: image directory
    if args.calib_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [p for p in Path(args.calib_dir).rglob("*") if p.suffix.lower() in exts]
        if not paths:
            raise RuntimeError(f"No images found under calib-dir={args.calib_dir}")

        mean = std = None
        if args.mean and args.std:
            mean = [float(x) for x in args.mean.split(",")]
            std = [float(x) for x in args.std.split(",")]
            assert len(mean) == args.input_channels and len(std) == args.input_channels, (
                f"mean/std must have {args.input_channels} comma-separated values"
            )

        n = min(args.calib_batches, len(paths))
        for i in range(n):
            yield image_to_tensor(
                paths[i],
                hw=hw,
                device=device,
                input_channels=args.input_channels,
                mean=mean,
                std=std,
            )
        return

    # Fallback: dummy calibration (not recommended)
    for _ in range(args.calib_batches):
        yield torch.rand(1, args.input_channels, hw[0], hw[1], device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Export SSD-MobileNetV2 to ONNX using NEMO FQ/QD/ID stages (ETH tutorial flow)"
    )
    parser.add_argument("--ckpt", type=str, default="training/person_ssd_pytorch/ssd_mbv2_raw.pth")
    parser.add_argument("--out", type=str, default="export/ssd_mbv2_nemo_id.onnx")

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1, 3])

    parser.add_argument("--bits", type=int, default=8, help="Quantization bits (like Q in notebook)")
    parser.add_argument(
        "--eps-in",
        type=float,
        default=1.0 / 255.0,
        help="Input quantum eps_in. For images in [0,1], use 1/255.",
    )

    parser.add_argument("--stage", choices=["fq", "qd", "id"], default="id",
                        help="Which stage to export (fq/q d/id).")
    parser.add_argument("--strict-stage", action="store_true",
                        help="Fail instead of falling back when qd/id conversion errors out.")
    parser.add_argument("--stage-report", type=str, default=None,
                        help="Optional path to write the final exported stage (fq/qd/id).")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--debug-forward-raw", action="store_true",
                        help="Print when forward_raw() is called to verify tracing/calibration path.")

    # Calibration inputs (matches notebook's statistics_act() idea)
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory of calibration images (jpg/png).")
    parser.add_argument("--calib-tensor", type=str, default=None,
                        help="Path to a .pt tensor file shaped [N,C,H,W] for calibration.")
    parser.add_argument("--calib-batches", type=int, default=64,
                        help="How many samples to use for activation calibration.")
    parser.add_argument("--mean", type=str, default=None,
                        help="Optional normalization mean, e.g. '0.5' (C=1) or '0.5,0.5,0.5' (C=3)")
    parser.add_argument("--std", type=str, default=None,
                        help="Optional normalization std, e.g. '0.5' (C=1) or '0.5,0.5,0.5' (C=3)")

    args = parser.parse_args()

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

    # 1) Build FP model + load weights
    model_fp = build_model(args.num_classes, args.width_mult, image_size, args.input_channels)
    model_fp = load_checkpoint(model_fp, args.ckpt, device)
    if hasattr(model_fp, "enable_forward_raw_debug"):
        model_fp.enable_forward_raw_debug(args.debug_forward_raw)
    model_fp.to(device).eval()

    # NEMO expects a dummy_input for graph tracing
    dummy_input = torch.randn(1, args.input_channels, args.height, args.width, device=device)
    if patch_model_to_graph_compat():
        print(
            "[export_nemo_quant] Applied _model_to_graph compatibility shim "
            "for older torch versions."
        )

    # 2) FQ: quantize_pact + set bitwidth + activation calibration (ETH notebook)
    print("[export_nemo_quant] Building FakeQuantized (FQ) model via quantize_pact...")
    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
    if hasattr(model_q, "enable_forward_raw_debug"):
        model_q.enable_forward_raw_debug(args.debug_forward_raw)
    model_q.to(device).eval()
    if bind_safe_set_eps_in(model_q):
        print(
            "[export_nemo_quant] Applied safe set_eps_in patch "
            "(dict-first + graph fallback) for QD/ID."
        )

    print(f"[export_nemo_quant] Setting precision to {args.bits} bits...")
    # notebook: model_q.change_precision(bits=Q, scale_weights=True, scale_activations=True)
    model_q.change_precision(bits=args.bits, scale_weights=True, scale_activations=True)

    if not args.calib_dir and not args.calib_tensor:
        print("[export_nemo_quant] WARNING: No calib data provided; using random tensors for statistics_act(). "
              "Provide --calib-dir or --calib-tensor for real calibration.")

    print("[export_nemo_quant] Calibrating activations with statistics_act() ...")
    with torch.no_grad():
        with model_q.statistics_act():
            for x in iter_calib_batches(args, image_size, device):
                _ = model_q(x)
    model_q.reset_alpha_act()
    # optional but often helpful (the notebook sometimes does this before QD):
    try:
        model_q.reset_alpha_weights()
    except Exception:
        pass

    model_deploy = model_q
    exported_stage = args.stage

    # 3) QD / ID using the notebook API (qd_stage / id_stage)
    if args.stage in ["qd", "id"]:
        debug_backbone_feature_paths(model_deploy, context="before qd_stage")
        print(f"[export_nemo_quant] Entering QuantizedDeployable (QD) via qd_stage(eps_in={args.eps_in}) ...")
        try:
            model_deploy.qd_stage(eps_in=args.eps_in)
        except Exception as e:
            if is_qd_eps_mapping_error(e):
                print(f"[export_nemo_quant] WARNING: qd_stage eps-mapping failed ({type(e).__name__}: {e}).")
                print("[export_nemo_quant] Applying named_modules() eps fallback and continuing QD conversion.")
                try:
                    run_best_effort_qd_steps(model_deploy, eps_in=args.eps_in)
                except Exception as fallback_error:
                    if args.strict_stage:
                        raise fallback_error
                    print(
                        "[export_nemo_quant] WARNING: QD fallback failed "
                        f"({type(fallback_error).__name__}: {fallback_error})."
                    )
                    print("[export_nemo_quant] Falling back to FQ export.")
                    exported_stage = "fq"
                    model_deploy = model_q
                else:
                    exported_stage = "qd"
            else:
                if args.strict_stage:
                    raise
                print(f"[export_nemo_quant] WARNING: qd_stage failed ({type(e).__name__}: {e}).")
                print("[export_nemo_quant] Falling back to FQ export.")
                exported_stage = "fq"
                model_deploy = model_q
        else:
            if args.stage == "qd":
                exported_stage = "qd"

    if args.stage == "id" and exported_stage != "fq":
        print("[export_nemo_quant] Entering IntegerDeployable (ID) via id_stage(eps_in=dict) ...")
        try:
            bind_safe_set_eps_in(model_deploy)
            eps_dict = build_eps_dict_from_modules(model_deploy)
            if not eps_dict:
                raise RuntimeError("eps_dict is empty; no modules expose eps_in. QD fallback likely failed.")
            seed_bn_eps_for_id(model_deploy)
            model_deploy.id_stage(eps_in=eps_dict)
        except Exception as e:
            if args.strict_stage:
                raise
            print(f"[export_nemo_quant] WARNING: id_stage failed ({type(e).__name__}: {e}).")
            print("[export_nemo_quant] Falling back to QD export.")
            exported_stage = "qd"
        else:
            exported_stage = "id"

    model_deploy.eval()
    for param in model_deploy.parameters():
        param.requires_grad_(False)

    # 4) Export ONNX
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape = (args.input_channels, args.height, args.width)  # NEMO export_onnx expects (C,H,W)
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
        round_params=True,
        batch_size=1,
    )

    if args.stage_report:
        stage_path = Path(args.stage_report)
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(f"{exported_stage}\n", encoding="utf-8")

    print(f"[export_nemo_quant] Final exported stage: {exported_stage.upper()} (requested: {args.stage.upper()})")
    print("[export_nemo_quant] Done.")


if __name__ == "__main__":
    main()

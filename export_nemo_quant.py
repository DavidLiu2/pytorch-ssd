#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

import nemo  # pytorch-nemo (pulp-platform)
from PIL import Image

from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw


def build_model(num_classes: int, width_mult: float, image_size):
    return SSDMobileNetV2Raw(
        num_classes=num_classes,
        width_mult=width_mult,
        image_size=image_size,
    )


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

    best = None
    for name, cand in candidates:
        missing, unexpected = model.load_state_dict(cand, strict=False)
        score = len(missing) + len(unexpected)
        if best is None or score < best["score"]:
            best = {
                "name": name,
                "state": cand,
                "missing": missing,
                "unexpected": unexpected,
                "score": score,
            }

    # Reload best candidate so final model state matches chosen mapping.
    missing, unexpected = model.load_state_dict(best["state"], strict=False)
    print(
        f"[export_nemo_quant] Checkpoint mapping: {best['name']} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[export_nemo_quant] Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[export_nemo_quant] Unexpected keys (first 10): {unexpected[:10]}")
    return model


def image_to_tensor(path: Path, hw: tuple[int, int], device, mean=None, std=None):
    # output: float32 [1,3,H,W] in [0,1] (optionally normalized)
    im = Image.open(path).convert("RGB").resize((hw[1], hw[0]), resample=Image.BILINEAR)
    x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
                          .view(im.size[1], im.size[0], 3)
                          .numpy())).to(torch.uint8)

    # (H,W,3) uint8 -> (1,3,H,W) float32 in [0,1]
    x = x.permute(2, 0, 1).contiguous().unsqueeze(0).to(device=device)
    x = x.float().div_(255.0)

    if mean is not None and std is not None:
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
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
        assert t.ndim == 4 and t.shape[1] == 3, f"Expected [N,3,H,W], got {tuple(t.shape)}"
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
            assert len(mean) == 3 and len(std) == 3, "mean/std must be 3 comma-separated values"

        n = min(args.calib_batches, len(paths))
        for i in range(n):
            yield image_to_tensor(paths[i], hw=hw, device=device, mean=mean, std=std)
        return

    # Fallback: dummy calibration (not recommended)
    for _ in range(args.calib_batches):
        yield torch.rand(1, 3, hw[0], hw[1], device=device)


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

    # Calibration inputs (matches notebook's statistics_act() idea)
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory of calibration images (jpg/png).")
    parser.add_argument("--calib-tensor", type=str, default=None,
                        help="Path to a .pt tensor file shaped [N,3,H,W] for calibration.")
    parser.add_argument("--calib-batches", type=int, default=64,
                        help="How many samples to use for activation calibration.")
    parser.add_argument("--mean", type=str, default=None,
                        help="Optional normalization mean, e.g. '0.5,0.5,0.5'")
    parser.add_argument("--std", type=str, default=None,
                        help="Optional normalization std, e.g. '0.5,0.5,0.5'")

    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if args.force_cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[export_nemo_quant] Using device: {device}")

    image_size = (args.height, args.width)

    # 1) Build FP model + load weights
    model_fp = build_model(args.num_classes, args.width_mult, image_size)
    model_fp = load_checkpoint(model_fp, args.ckpt, device)
    model_fp.to(device).eval()

    # NEMO expects a dummy_input for graph tracing
    dummy_input = torch.randn(1, 3, args.height, args.width, device=device)

    # 2) FQ: quantize_pact + set bitwidth + activation calibration (ETH notebook)
    print("[export_nemo_quant] Building FakeQuantized (FQ) model via quantize_pact...")
    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input)
    model_q.to(device).eval()

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
        print(f"[export_nemo_quant] Entering QuantizedDeployable (QD) via qd_stage(eps_in={args.eps_in}) ...")
        try:
            model_deploy.qd_stage(eps_in=args.eps_in)
        except Exception as e:
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
        print("[export_nemo_quant] Entering IntegerDeployable (ID) via id_stage() ...")
        try:
            model_deploy.id_stage()
        except Exception as e:
            if args.strict_stage:
                raise
            print(f"[export_nemo_quant] WARNING: id_stage failed ({type(e).__name__}: {e}).")
            print("[export_nemo_quant] Falling back to QD export.")
            exported_stage = "qd"
        else:
            exported_stage = "id"

    model_deploy.eval()

    # 4) Export ONNX
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape = (3, args.height, args.width)  # NEMO export_onnx expects (C,H,W)
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

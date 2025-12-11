#!/usr/bin/env python3
import argparse
import torch

import nemo  # this is pytorch-nemo (pulp-platform), not NVIDIA NeMo

from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw


def build_model(num_classes: int, width_mult: float, image_size):
    """
    Create the same raw SSD-MobileNetV2 backbone you used for training.
    """
    model = SSDMobileNetV2Raw(
        num_classes=num_classes,
        width_mult=width_mult,
        image_size=image_size,
    )
    return model


def load_checkpoint(model, ckpt_path, device):
    """
    Load your .pth weights with strict=False so we can ignore any extra keys.
    """
    print(f"[export_nemo_quant] Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[export_nemo_quant] Missing keys: {missing}")
    print(f"[export_nemo_quant] Unexpected keys: {unexpected}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export SSD-MobileNetV2 (NEMO-quantized) to ONNX for DORY"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="training/person_ssd_pytorch/ssd_mbv2_raw.pth",
        help="Path to trained .pth checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="export/ssd_mbv2_quant.onnx",
        help="Output ONNX file path (e.g. export/ssd_mnv2_nemo_id.onnx)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes including background (your training setting)",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=0.1,
        help="Width multiplier for MobileNetV2 (must match training)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=160,
        help="Input height (must match training)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Input width (must match training)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force export on CPU even if CUDA is available",
    )

    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not args.force_cpu
        else torch.device("cpu")
    )
    print(f"[export_nemo_quant] Using device: {device}")

    image_size = (args.height, args.width)

    # 1. Build float model and load weights
    model = build_model(
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        image_size=image_size,
    )
    model = load_checkpoint(model, args.ckpt, device)
    model.to(device)
    model.eval()

    # 2. Prepare dummy input for NEMO & ONNX export
    dummy_input = torch.randn(
        1, 3, image_size[0], image_size[1], device=device
    )

    # 3. Create NEMO fake-quantized (PACT) model
    print("[export_nemo_quant] Applying NEMO PACT fake quantization...")
    # This converts ReLUs to PACT_Act + quantized weights (FQ representation)
    model_fq = nemo.transform.quantize_pact(
        model,
        dummy_input=dummy_input,
    )
    model_fq.to(device)
    model_fq.eval()

    # 4. (Optional) Convert to Quantized/Integer Deployable representations
    #    depending on what your installed NEMO exposes. We try a couple of
    #    known transforms but fall back gracefully if they don't exist.
    model_deploy = model_fq
    try:
        # Quantized Deployable (QD)
        print("[export_nemo_quant] Trying nemo.transform.quantize_qd...")
        model_qd = nemo.transform.quantize_qd(
            model_fq,
            dummy_input=dummy_input,
        )
        model_qd.to(device)
        model_qd.eval()
        model_deploy = model_qd
        print("[export_nemo_quant] -> quantize_qd succeeded")
    except (AttributeError, TypeError):
        print(
            "[export_nemo_quant] quantize_qd not available in this NEMO "
            "version; staying at PACT (FQ) level."
        )

    # If IntegerDeployable is available, try to use it as final export model.
    try:
        print("[export_nemo_quant] Trying nemo.transform.quantize_id...")
        model_id = nemo.transform.quantize_id(
            model_deploy,
            dummy_input=dummy_input,
        )
        model_id.to(device)
        model_id.eval()
        model_deploy = model_id
        print("[export_nemo_quant] -> quantize_id succeeded (IntegerDeployable)")
    except (AttributeError, TypeError):
        print(
            "[export_nemo_quant] quantize_id not available; "
            "exporting current deploy model instead."
        )

    # 5. Export to ONNX with NEMO helper
    # NEMO's export_onnx expects input_shape = (C, H, W)
    input_shape = (3, image_size[0], image_size[1])
    print(
        f"[export_nemo_quant] Exporting NEMO-quantized model to ONNX:\n"
        f"  -> {args.out}\n"
        f"  input_shape=(1, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})"
    )

    nemo.utils.export_onnx(
        args.out,
        model_deploy,
        model_deploy,
        input_shape,
        round_params=True,
        batch_size=1,  # explicit, though it's the default
    )

    print("[export_nemo_quant] Done. ONNX saved at:", args.out)


if __name__ == "__main__":
    main()

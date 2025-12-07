# pytorch_ssd/export_float_raw.py
import argparse
from pathlib import Path

import torch

from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default="training/person_ssd_pytorch/ssd_mbv2_epoch_030.pth",
        help="Path to the checkpoint you liked from training",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="training/person_ssd_pytorch/ssd_mbv2_raw.pth",
        help="Where to save the raw-heads wrapper weights",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build wrapper
    model = SSDMobileNetV2Raw(
        num_classes=2,
        width_mult=0.5,
        image_size=(320, 320),
    ).to(device)

    # 2) Load training checkpoint into the internal SSD
    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.ssd.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # 3) Quick sanity check
    model.eval()
    x = torch.randn(1, 3, 320, 320, device=device)
    with torch.no_grad():
        locs, cls_logits = model(x)
    print("locs shape:", locs.shape)           # [1, N_boxes, 4]
    print("cls_logits shape:", cls_logits.shape)  # [1, N_boxes, 2]

    # 4) Save wrapper state_dict
    out_path = Path(args.out)
    torch.save(model.state_dict(), out_path)
    print(f"Saved raw-heads model to {out_path}")


if __name__ == "__main__":
    main()

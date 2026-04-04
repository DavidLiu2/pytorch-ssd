import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image

from models.follow_model_factory import (
    build_follow_model_from_checkpoint,
    load_checkpoint_payload,
    load_follow_checkpoint,
)
from utils.follow_task import follow_runtime_decode_summary


def _load_checkpoint_payload(ckpt_path: Path, device: torch.device):
    return load_checkpoint_payload(ckpt_path, device)


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    return build_follow_model_from_checkpoint(ckpt_path, device)


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    *,
    checkpoint=None,
) -> None:
    load_follow_checkpoint(
        model,
        ckpt_path,
        device,
        checkpoint=checkpoint,
        strict=True,
    )


def preprocess_image(image_path: Path, image_size=(128, 128)) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    width, height = img.size
    crop_size = min(width, height)
    crop_left = (width - crop_size) // 2
    crop_top = (height - crop_size) // 2
    img = F.crop(img, crop_top, crop_left, crop_size, crop_size)
    img = F.resize(img, list(image_size))
    img = F.to_tensor(img)
    return img.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="training/hybrid_follow/hybrid_follow_epoch_030.pth",
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(Path(args.ckpt), device)
    model.eval()

    x = preprocess_image(Path(args.image)).to(device)

    with torch.no_grad():
        raw = model(x)[0]

    metadata = _load_checkpoint_payload(Path(args.ckpt), device)
    model_type = str((metadata or {}).get("model_type", "hybrid_follow"))
    follow_head_type = (metadata or {}).get("follow_head_type")
    decoded = follow_runtime_decode_summary(
        raw,
        head_type=follow_head_type,
        model_type=model_type,
        vis_thresh=args.vis_thresh,
    )

    print(f"model_type={model_type}")
    if follow_head_type is not None:
        print(f"follow_head_type={follow_head_type}")
    print("raw_output={}".format(",".join(f"{float(value):.6f}" for value in raw.detach().cpu().tolist())))
    for key, value in decoded.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()

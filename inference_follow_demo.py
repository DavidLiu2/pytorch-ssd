import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image

from models.hybrid_follow_net import HybridFollowNet


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint payload is not a state_dict-like dict.")
    model.load_state_dict(state, strict=True)


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
    model = HybridFollowNet(input_channels=1, image_size=(128, 128)).to(device)
    load_checkpoint(model, Path(args.ckpt), device)
    model.eval()

    x = preprocess_image(Path(args.image)).to(device)

    with torch.no_grad():
        raw = model(x)[0]

    x_offset_raw = float(raw[0].cpu().item())
    size_proxy_raw = float(raw[1].cpu().item())
    visibility_logit = float(raw[2].cpu().item())

    x_offset = max(-1.0, min(1.0, x_offset_raw))
    size_proxy = max(0.0, min(1.0, size_proxy_raw))
    visibility_confidence = float(torch.sigmoid(raw[2]).cpu().item())
    target_visible = visibility_confidence >= args.vis_thresh

    print(f"raw_x_offset={x_offset_raw:.6f}")
    print(f"raw_size_proxy={size_proxy_raw:.6f}")
    print(f"raw_visibility_logit={visibility_logit:.6f}")
    print(f"x_offset={x_offset:.6f}")
    print(f"size_proxy={size_proxy:.6f}")
    print(f"visibility_confidence={visibility_confidence:.6f}")
    print(f"target_visible={int(target_visible)}")


if __name__ == "__main__":
    main()

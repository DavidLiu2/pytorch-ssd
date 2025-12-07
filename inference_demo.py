import argparse
from pathlib import Path

import torch
import torchvision
import cv2
import numpy as np
import torchvision.transforms.functional as F

from models.ssd_mobilenet_v2 import create_ssd_mobilenet_v2


def load_model(ckpt_path: Path, num_classes=2, width_mult=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_ssd_mobilenet_v2(
        num_classes=num_classes,
        width_mult=width_mult,
        image_size=(320, 320),
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device



def preprocess_image(img_bgr):
    # img_bgr: H x W x 3, uint8 (OpenCV BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = torchvision.transforms.ToPILImage()(img_rgb)

    # grayscale → tensor → repeat to 3 channels
    img_gray = pil.convert("L")
    img_t = F.to_tensor(img_gray)      # [1,H,W]
    img_t = img_t.repeat(3, 1, 1)      # [3,H,W], same as training
    return img_t



def draw_boxes(img_bgr, boxes, scores, score_thresh=0.5):
    h, w, _ = img_bgr.shape
    for box, score in zip(boxes, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(
            img_bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_bgr,
            f"{score:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="training/person_ssd_pytorch/ssd_mbv2_epoch_030.pth",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a test image (person present)",
    )
    parser.add_argument("--score_thresh", type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    model, device = load_model(ckpt_path)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image {args.image}")

    img_t = preprocess_image(img_bgr)
    img_t = img_t.to(device)

    with torch.no_grad():
        # torchvision detection API: model expects list[Tensor]
        outputs = model([img_t])[0]  # single image → first output dict

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    img_out = draw_boxes(img_bgr.copy(), boxes, scores, score_thresh=args.score_thresh)
    out_path = ckpt_path.parent / "demo_output.jpg"
    cv2.imwrite(str(out_path), img_out)
    print(f"Saved detection result to {out_path}")


if __name__ == "__main__":
    main()

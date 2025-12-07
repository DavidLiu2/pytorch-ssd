import random
from typing import Dict, Any

import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
            # ensure target boxes remain tensor if present
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() < self.p:
            img = F.hflip(img)
            w, _ = img.size
            boxes = target["boxes"]
            # boxes: [N, 4] in (x1, y1, x2, y2)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return img, target


class ToTensorGray3:
    """
    Convert RGB PIL image -> grayscale -> 3-channel tensor [3,H,W] in [0,1].

    We leave resizing and normalization to torchvision's SSDTransform.
    """

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        # Convert to grayscale (single channel)
        img_gray = img.convert("L")          # PIL, 1-channel
        img_t = F.to_tensor(img_gray)        # [1,H,W] float in [0,1]
        img_t = img_t.repeat(3, 1, 1)        # [3,H,W], identical channels
        return img_t, target


def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(0.5),
        ToTensorGray3(),
    ])


def get_val_transforms():
    return Compose([
        ToTensorGray3(),
    ])

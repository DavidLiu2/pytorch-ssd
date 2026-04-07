import random
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


def _empty_boxes_like(boxes: torch.Tensor) -> torch.Tensor:
    return boxes.new_zeros((0, 4))


def _boxes_from_target(target: Dict[str, Any]) -> Optional[torch.Tensor]:
    boxes = target.get("boxes")
    if boxes is None:
        return None
    if boxes.numel() == 0:
        return _empty_boxes_like(boxes)
    return boxes


def _filter_target_rows(target: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
    for key in ("boxes", "labels", "area", "iscrowd"):
        value = target.get(key)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == keep.shape[0]:
            target[key] = value[keep]
    return target


def _recompute_area(target: Dict[str, Any]) -> Dict[str, Any]:
    boxes = _boxes_from_target(target)
    if boxes is None:
        return target
    if boxes.numel() == 0:
        target["boxes"] = _empty_boxes_like(boxes)
        target["area"] = boxes.new_zeros((0,))
        return target

    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    target["area"] = widths * heights
    return target


def _clamp_and_filter_boxes(
    target: Dict[str, Any],
    width: int,
    height: int,
) -> Dict[str, Any]:
    boxes = _boxes_from_target(target)
    if boxes is None:
        return target
    if boxes.numel() == 0:
        target["boxes"] = _empty_boxes_like(boxes)
        target["area"] = boxes.new_zeros((0,))
        return target

    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(width))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(height))
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    target["boxes"] = boxes
    target = _filter_target_rows(target, keep)
    return _recompute_area(target)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class CenterCropSquare:
    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        width, height = img.size
        crop_size = min(width, height)
        crop_left = (width - crop_size) // 2
        crop_top = (height - crop_size) // 2

        img = F.crop(img, crop_top, crop_left, crop_size, crop_size)

        boxes = _boxes_from_target(target)
        if boxes is not None and boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] -= float(crop_left)
            boxes[:, [1, 3]] -= float(crop_top)
            target["boxes"] = boxes
            target = _clamp_and_filter_boxes(target, crop_size, crop_size)

        return img, target


class ResizeImage:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        old_width, old_height = img.size
        new_height, new_width = self.size
        img = F.resize(img, [new_height, new_width])

        boxes = _boxes_from_target(target)
        if boxes is not None and boxes.numel() > 0:
            scale_x = float(new_width) / float(old_width)
            scale_y = float(new_height) / float(old_height)
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
            target = _recompute_area(target)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        if random.random() < self.p:
            img = F.hflip(img)
            width, _ = img.size
            boxes = _boxes_from_target(target)
            if boxes is not None and boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = float(width) - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


class ToTensorGray:
    """
    Convert a PIL image to a grayscale tensor.

    By default returns true single-channel tensors shaped [1, H, W].
    """

    def __init__(self, output_channels: int = 1):
        if output_channels not in (1, 3):
            raise ValueError("output_channels must be 1 or 3")
        self.output_channels = output_channels

    def __call__(self, img: Image.Image, target: Dict[str, Any]):
        img_gray = img.convert("L")
        img_t = F.to_tensor(img_gray)
        if self.output_channels == 3:
            img_t = img_t.repeat(3, 1, 1)
        return img_t, target


def get_train_transforms(
    model_type: str = "ssd",
    input_channels: int = 1,
    image_size: Tuple[int, int] = (128, 128),
):
    if model_type in {"hybrid_follow", "plain_follow", "plain_follow_bin", "plain_follow_v2", "plain_follow_tiny", "dronet_lite_follow"}:
        if input_channels != 1:
            raise ValueError(f"{model_type} path requires input_channels=1.")
        flip_prob = 0.5
        if model_type == "dronet_lite_follow":
            # Keep dronet-lite augmentation milder so the visibility gate settles
            # before the residual path starts chasing harder x offsets.
            flip_prob = 0.25
        return Compose(
            [
                CenterCropSquare(),
                ResizeImage(image_size),
                RandomHorizontalFlip(flip_prob),
                ToTensorGray(output_channels=1),
            ]
        )

    return Compose(
        [
            RandomHorizontalFlip(0.5),
            ToTensorGray(output_channels=input_channels),
        ]
    )


def get_val_transforms(
    model_type: str = "ssd",
    input_channels: int = 1,
    image_size: Tuple[int, int] = (128, 128),
):
    if model_type in {"hybrid_follow", "plain_follow", "plain_follow_bin", "plain_follow_v2", "plain_follow_tiny", "dronet_lite_follow"}:
        if input_channels != 1:
            raise ValueError(f"{model_type} path requires input_channels=1.")
        return Compose(
            [
                CenterCropSquare(),
                ResizeImage(image_size),
                ToTensorGray(output_channels=1),
            ]
        )

    return Compose(
        [
            ToTensorGray(output_channels=input_channels),
        ]
    )

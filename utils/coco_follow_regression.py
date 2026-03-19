from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def _image_hw(img) -> Tuple[int, int]:
    if isinstance(img, torch.Tensor):
        return int(img.shape[-2]), int(img.shape[-1])
    width, height = img.size
    return int(height), int(width)


def _largest_valid_box(boxes: torch.Tensor) -> Optional[torch.Tensor]:
    if boxes.numel() == 0:
        return None

    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    areas = widths * heights
    valid = areas > 0.0
    if not torch.any(valid):
        return None

    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    best_local = int(torch.argmax(areas[valid]).item())
    best_index = int(valid_indices[best_local].item())
    return boxes[best_index]


def compute_follow_target(
    boxes: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Target formulas after crop/resize geometry:

      x_offset  = clamp(2 * ((box_center_x / image_width) - 0.5), -1, 1)
      size_proxy = clamp(box_height / image_height, 0, 1)
      visibility_confidence = 1 if a valid box exists else 0
    """

    best_box = _largest_valid_box(boxes)
    if best_box is None:
        return (
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        )

    x1, y1, x2, y2 = best_box.tolist()
    box_center_x = 0.5 * (x1 + x2)
    box_height = max(y2 - y1, 0.0)

    x_offset = (2.0 * (box_center_x / float(image_width))) - 1.0
    size_proxy = box_height / float(image_height)

    follow_target = torch.tensor(
        [
            float(max(-1.0, min(1.0, x_offset))),
            float(max(0.0, min(1.0, size_proxy))),
            1.0,
        ],
        dtype=torch.float32,
    )
    return follow_target, best_box.to(dtype=torch.float32)


class COCOFollowRegressionDataset(Dataset):
    def __init__(self, root: str, ann_file: str, transforms=None, image_mode: str = "L"):
        super().__init__()
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.transforms = transforms
        if image_mode not in {"RGB", "L"}:
            raise ValueError("image_mode must be 'RGB' or 'L'")
        self.image_mode = image_mode

        self.coco = COCO(ann_file)
        self.person_cat_ids = self.coco.getCatIds(catNms=["person"])
        if not self.person_cat_ids:
            self.person_cat_ids = self.coco.getCatIds()
        self.img_ids = self.coco.getImgIds()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = self.root / img_info["file_name"]
        img = Image.open(img_path).convert(self.image_mode)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            if ann.get("category_id") not in self.person_cat_ids:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            areas.append(float(w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        true_no_person = 1 if len(boxes) == 0 else 0

        target = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "true_no_person": torch.tensor([true_no_person], dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        image_height, image_width = _image_hw(img)
        follow_target, best_box = compute_follow_target(
            target["boxes"],
            image_height=image_height,
            image_width=image_width,
        )

        return img, {
            "follow_target": follow_target,
            "largest_box": best_box,
            "image_id": target["image_id"],
            "true_no_person": target["true_no_person"],
        }

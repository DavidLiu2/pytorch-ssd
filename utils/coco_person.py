import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class COCOPersonDataset(Dataset):
    """
    COCO person-only dataset compatible with torchvision detection models.

    Root layout (relative to repo root):
      data/coco/images/train2017
      data/coco/images/val2017
      data/coco/annotations/train_person.json
      data/coco/annotations/val_person.json
    """

    def __init__(self, root: str, ann_file: str, transforms=None):
        super().__init__()
        self.root = Path(root)
        self.transforms = transforms

        self.coco = COCO(ann_file)
        # person-only JSON already filtered, but we’ll still pull category IDs:
        self.cat_ids = self.coco.getCatIds()
        # COCO image ids list:
        self.img_ids = self.coco.getImgIds()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = self.root / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox is [x, y, w, h]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            # Only one foreground class: person → label 1 (0 is background)
            labels.append(1)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target: Dict[str, Any] = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# Simple detection-style collate function for DataLoader
def detection_collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

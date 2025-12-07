import os
from pathlib import Path
import argparse
import math
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.coco_person import COCOPersonDataset, detection_collate_fn
from models.ssd_mobilenet_v2 import create_ssd_mobilenet_v2
from utils.transforms import get_train_transforms, get_val_transforms


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="pytorch_ssd/data/coco/images/train2017",
                    help="Path to train images dir (include repo root)")
    ap.add_argument("--train_ann", type=str,
                    default="pytorch_ssd/data/coco/annotations/train_person.json")
    ap.add_argument("--val_root", type=str, default="pytorch_ssd/data/coco/images/val2017")
    ap.add_argument("--val_ann", type=str,
                    default="pytorch_ssd/data/coco/annotations/val_person.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=str, default="pytorch_ssd/training/person_ssd_pytorch")
    return ap.parse_args()

def reduce_losses(loss_out):
    """
    Robustly reduce SSD loss outputs to a single scalar tensor.

    Handles:
      - dict[str, Tensor]          (torchvision >= 0.9 typical)
      - list[dict[str, Tensor]]    (some SSD variants, one dict per image)
      - plain Tensor
    """

    def _reduce_dict(d):
        # Sum *scalarized* losses in a dict
        total = 0.0
        for v in d.values():
            if not torch.is_tensor(v):
                continue
            # Make sure it's a scalar: sum over all dims
            total = total + v.sum()
        # return as tensor
        if not torch.is_tensor(total):
            total = torch.as_tensor(total)
        return total

    if isinstance(loss_out, dict):
        return _reduce_dict(loss_out)

    if isinstance(loss_out, list):
        # list of dicts or tensors
        totals = []
        for item in loss_out:
            if isinstance(item, dict):
                totals.append(_reduce_dict(item))
            elif torch.is_tensor(item):
                totals.append(item.sum())
        if len(totals) == 0:
            return torch.tensor(0.0)
        # average over batch
        return torch.stack(totals).mean()

    if torch.is_tensor(loss_out):
        return loss_out.sum()

    # fallback
    return torch.as_tensor(loss_out)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets
    train_ds = COCOPersonDataset(
        root=str(repo_root / args.data_root),
        ann_file=str(repo_root / args.train_ann),
        transforms=get_train_transforms(),
    )
    val_ds = COCOPersonDataset(
        root=str(repo_root / args.val_root),
        ann_file=str(repo_root / args.val_ann),
        transforms=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    # model
    model = create_ssd_mobilenet_v2(num_classes=2, width_mult=0.25, image_size=(320, 320))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc="Train", ncols=80)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = reduce_losses(loss_dict)
            loss_value = float(losses.detach().cpu().item())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += loss_value
            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Train loss: {avg_loss:.4f}")

        # quick val loss (optional; SSD val metrics really need mAP with COCO API)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = reduce_losses(loss_dict)
                val_loss += float(losses.detach().cpu().item())


        val_loss /= max(len(val_loader), 1)
        print(f"Val loss (approx): {val_loss:.4f}")

        ckpt_path = output_dir / f"ssd_mbv2_epoch_{epoch+1:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

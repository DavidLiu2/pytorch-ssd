import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inference_follow_demo import load_checkpoint, preprocess_image
from models.hybrid_follow_net import HybridFollowNet
from utils.coco_follow_regression import COCOFollowRegressionDataset
from utils.transforms import get_val_transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="training/hybrid_follow/hybrid_follow_epoch_030.pth",
    )
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vis-thresh", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--diagnostic-balanced",
        type=int,
        default=0,
        help="If set, print N strong positives and N negatives.",
    )
    parser.add_argument("--save-top-fp-dir", type=str, default=None)
    parser.add_argument("--save-top-fn-dir", type=str, default=None)
    parser.add_argument("--top-k-errors", type=int, default=20)
    return parser.parse_args()


def _new_model(device: torch.device, ckpt_path: Path) -> HybridFollowNet:
    model = HybridFollowNet(input_channels=1, image_size=(128, 128)).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    return model


def _row_source_path(row: Dict[str, object], image_root: Path) -> Path:
    path = row.get("path")
    if isinstance(path, str) and path:
        return Path(path)
    return image_root / str(row["file_name"])


def _save_rows_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_ranked_images(
    rows: List[Dict[str, object]],
    image_root: Path,
    target_visible: int,
    pred_visible: int,
    sort_desc: bool,
    out_dir: Optional[Path],
    top_k: int,
) -> None:
    if out_dir is None:
        return

    selected = [
        row
        for row in rows
        if row.get("target_visible") == target_visible and row.get("pred_visible") == pred_visible
    ]
    selected = sorted(
        selected,
        key=lambda row: float(row["visibility_confidence"]),
        reverse=sort_desc,
    )[:top_k]

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(selected, start=1):
        src = _row_source_path(row, image_root)
        dst = out_dir / (
            f"{rank:02d}_p{float(row['visibility_confidence']):.4f}_"
            f"{Path(str(row['file_name'])).name}"
        )
        shutil.copy2(src, dst)


def _print_row(row: Dict[str, object], vis_thresh: float) -> None:
    target = row.get("target_visible")
    if target is None:
        target_str = "?"
    else:
        target_str = str(int(target))

    visible_flag = int(float(row["visibility_confidence"]) >= vis_thresh)
    print(
        f"{row['file_name']}, "
        f"target={target_str}, "
        f"logit={float(row['raw_visibility_logit']):.6f}, "
        f"prob={float(row['visibility_confidence']):.6f}, "
        f"pred={visible_flag}"
    )


def _print_summary(rows: List[Dict[str, object]], vis_thresh: float) -> None:
    if not rows or rows[0].get("target_visible") is None:
        return

    positive_rows = [row for row in rows if int(row["target_visible"]) == 1]
    negative_rows = [row for row in rows if int(row["target_visible"]) == 0]
    pred_visible_frac = sum(int(row["pred_visible"]) for row in rows) / len(rows)
    gt_visible_frac = sum(int(row["target_visible"]) for row in rows) / len(rows)
    accuracy = (
        sum(int(row["pred_visible"]) == int(row["target_visible"]) for row in rows) / len(rows)
    )

    print("SUMMARY")
    print(f"num_total={len(rows)}")
    print(f"num_pos={len(positive_rows)}")
    print(f"num_neg={len(negative_rows)}")
    print(f"pred_visible_frac@{vis_thresh:.2f}={pred_visible_frac:.6f}")
    print(f"gt_visible_frac={gt_visible_frac:.6f}")
    print(f"accuracy@{vis_thresh:.2f}={accuracy:.6f}")

    if positive_rows:
        pos_logits = [float(row["raw_visibility_logit"]) for row in positive_rows]
        print(f"pos_mean_logit={sum(pos_logits) / len(pos_logits):.6f}")
        print(f"pos_min_logit={min(pos_logits):.6f}")
        print(f"pos_max_logit={max(pos_logits):.6f}")
    if negative_rows:
        neg_logits = [float(row["raw_visibility_logit"]) for row in negative_rows]
        print(f"neg_mean_logit={sum(neg_logits) / len(neg_logits):.6f}")
        print(f"neg_min_logit={min(neg_logits):.6f}")
        print(f"neg_max_logit={max(neg_logits):.6f}")


def _print_balanced_diagnostic(rows: List[Dict[str, object]], count: int, vis_thresh: float) -> None:
    if count <= 0 or not rows or rows[0].get("target_visible") is None:
        return

    positives = [
        row
        for row in rows
        if int(row["target_visible"]) == 1
    ]
    positives = sorted(positives, key=lambda row: float(row.get("target_size_proxy", 0.0)), reverse=True)

    no_person_negatives = [
        row
        for row in rows
        if int(row["target_visible"]) == 0 and int(row.get("true_no_person", 0)) == 1
    ]
    other_negatives = [
        row
        for row in rows
        if int(row["target_visible"]) == 0 and int(row.get("true_no_person", 0)) == 0
    ]

    print("DIAGNOSTIC_POSITIVES")
    for row in positives[:count]:
        _print_row(row, vis_thresh)

    print("DIAGNOSTIC_NEGATIVES")
    selected_negatives = no_person_negatives[:count]
    if len(selected_negatives) < count:
        selected_negatives.extend(other_negatives[: count - len(selected_negatives)])
    for row in selected_negatives:
        _print_row(row, vis_thresh)


def evaluate_coco(
    model: HybridFollowNet,
    device: torch.device,
    image_root: Path,
    ann_file: Path,
    batch_size: int,
    vis_thresh: float,
) -> List[Dict[str, object]]:
    dataset = COCOFollowRegressionDataset(
        root=str(image_root),
        ann_file=str(ann_file),
        transforms=get_val_transforms(
            model_type="hybrid_follow",
            input_channels=1,
            image_size=(128, 128),
        ),
        image_mode="L",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows: List[Dict[str, object]] = []
    vis_loss_sum = 0.0
    x_loss_sum = 0.0
    size_loss_sum = 0.0
    sample_count = 0
    visible_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            follow_targets = targets["follow_target"].to(device)
            raw = model(images)
            probs = torch.sigmoid(raw[:, 2])

            vis_loss_sum += F.binary_cross_entropy_with_logits(
                raw[:, 2],
                follow_targets[:, 2],
                reduction="sum",
            ).item()
            visible_mask = follow_targets[:, 2] > 0.5
            visible_count += int(visible_mask.sum().item())
            if torch.any(visible_mask):
                x_loss_sum += F.smooth_l1_loss(
                    raw[visible_mask, 0],
                    follow_targets[visible_mask, 0],
                    reduction="sum",
                ).item()
                size_loss_sum += F.smooth_l1_loss(
                    raw[visible_mask, 1],
                    follow_targets[visible_mask, 1],
                    reduction="sum",
                ).item()
            sample_count += int(follow_targets.shape[0])

            image_ids = targets["image_id"][:, 0].tolist()
            for batch_index, img_id in enumerate(image_ids):
                anns = dataset.coco.loadAnns(
                    dataset.coco.getAnnIds(
                        imgIds=img_id,
                        catIds=dataset.person_cat_ids,
                        iscrowd=None,
                    )
                )
                img_info = dataset.coco.loadImgs(img_id)[0]
                rows.append(
                    {
                        "image_id": img_id,
                        "file_name": img_info["file_name"],
                        "target_visible": int(follow_targets[batch_index, 2].item() > 0.5),
                        "target_size_proxy": float(follow_targets[batch_index, 1].item()),
                        "true_no_person": int(len(anns) == 0),
                        "raw_visibility_logit": float(raw[batch_index, 2].item()),
                        "visibility_confidence": float(probs[batch_index].item()),
                        "pred_visible": int(probs[batch_index].item() >= vis_thresh),
                    }
                )

    if sample_count > 0:
        avg_vis_loss = vis_loss_sum / sample_count
        avg_x_loss = x_loss_sum / max(visible_count, 1)
        avg_size_loss = size_loss_sum / max(visible_count, 1)
        print("LOSS")
        print(f"avg_visibility_bce={avg_vis_loss:.6f}")
        print(f"avg_x_offset_smooth_l1={avg_x_loss:.6f}")
        print(f"avg_size_proxy_smooth_l1={avg_size_loss:.6f}")
        print(f"visible_samples={visible_count}")
        print(f"all_samples={sample_count}")

    return rows


def evaluate_folder(
    model: HybridFollowNet,
    device: torch.device,
    image_root: Path,
    batch_size: int,
    vis_thresh: float,
) -> List[Dict[str, object]]:
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_root}")

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = torch.cat([preprocess_image(path) for path in batch_paths], dim=0).to(device)
            raw = model(images)
            probs = torch.sigmoid(raw[:, 2])
            for batch_index, path in enumerate(batch_paths):
                rows.append(
                    {
                        "path": str(path),
                        "file_name": path.name,
                        "target_visible": None,
                        "true_no_person": None,
                        "raw_visibility_logit": float(raw[batch_index, 2].item()),
                        "visibility_confidence": float(probs[batch_index].item()),
                        "pred_visible": int(probs[batch_index].item() >= vis_thresh),
                    }
                )
    return rows


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _new_model(device, Path(args.ckpt))
    image_root = Path(args.image_root)

    if args.ann_file:
        rows = evaluate_coco(
            model=model,
            device=device,
            image_root=image_root,
            ann_file=Path(args.ann_file),
            batch_size=args.batch_size,
            vis_thresh=args.vis_thresh,
        )
    else:
        rows = evaluate_folder(
            model=model,
            device=device,
            image_root=image_root,
            batch_size=args.batch_size,
            vis_thresh=args.vis_thresh,
        )

    _print_balanced_diagnostic(rows, args.diagnostic_balanced, args.vis_thresh)
    _print_summary(rows, args.vis_thresh)

    if args.csv:
        _save_rows_csv(rows, Path(args.csv))
        print(f"CSV={args.csv}")

    if args.ann_file:
        _copy_ranked_images(
            rows,
            image_root=image_root,
            target_visible=0,
            pred_visible=1,
            sort_desc=True,
            out_dir=Path(args.save_top_fp_dir) if args.save_top_fp_dir else None,
            top_k=args.top_k_errors,
        )
        _copy_ranked_images(
            rows,
            image_root=image_root,
            target_visible=1,
            pred_visible=0,
            sort_desc=False,
            out_dir=Path(args.save_top_fn_dir) if args.save_top_fn_dir else None,
            top_k=args.top_k_errors,
        )

    if not args.ann_file:
        for row in rows:
            _print_row(row, args.vis_thresh)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.hybrid_follow_net import HybridFollowNet
from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw
from utils.coco_follow_regression import COCOFollowRegressionDataset
from utils.coco_person import COCOPersonDataset, detection_collate_fn
from utils.transforms import get_train_transforms, get_val_transforms


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="pytorch_ssd/data/coco/images/train2017",
        help="Path to train images dir (include repo root)",
    )
    ap.add_argument(
        "--train_ann",
        type=str,
        default=None,
    )
    ap.add_argument("--val_root", type=str, default="pytorch_ssd/data/coco/images/val2017")
    ap.add_argument(
        "--val_ann",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default="hybrid_follow",
        choices=["ssd", "hybrid_follow"],
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--width-mult", type=float, default=0.1)
    ap.add_argument(
        "--allow-person-only-follow-ann",
        action="store_true",
        help="Allow hybrid_follow training on *_person.json annotations. "
        "This is discouraged because it removes true no-person negatives.",
    )
    ap.add_argument(
        "--vis-thresh",
        type=float,
        default=0.5,
        help="Visibility threshold used for reporting val accuracy/F1 and no-person FP rate.",
    )
    return ap.parse_args()


def reduce_losses(loss_out):
    """
    Robustly reduce SSD loss outputs to a single scalar tensor.
    """

    def _reduce_dict(d):
        total = 0.0
        for value in d.values():
            if torch.is_tensor(value):
                total = total + value.sum()
        if not torch.is_tensor(total):
            total = torch.as_tensor(total)
        return total

    if isinstance(loss_out, dict):
        return _reduce_dict(loss_out)

    if isinstance(loss_out, list):
        totals = []
        for item in loss_out:
            if isinstance(item, dict):
                totals.append(_reduce_dict(item))
            elif torch.is_tensor(item):
                totals.append(item.sum())
        if not totals:
            return torch.tensor(0.0)
        return torch.stack(totals).mean()

    if torch.is_tensor(loss_out):
        return loss_out.sum()

    return torch.as_tensor(loss_out)


def compute_hybrid_follow_loss(predictions: torch.Tensor, follow_targets: torch.Tensor):
    x_target = follow_targets[:, 0]
    size_target = follow_targets[:, 1]
    visibility_target = follow_targets[:, 2]

    visibility_loss = F.binary_cross_entropy_with_logits(
        predictions[:, 2],
        visibility_target,
    )

    visible_mask = visibility_target > 0.5
    zero = predictions.new_zeros(())
    if torch.any(visible_mask):
        x_loss = F.smooth_l1_loss(
            predictions[visible_mask, 0],
            x_target[visible_mask],
            reduction="mean",
        )
        size_loss = F.smooth_l1_loss(
            predictions[visible_mask, 1],
            size_target[visible_mask],
            reduction="mean",
        )
    else:
        x_loss = zero
        size_loss = zero

    total_loss = x_loss + size_loss + visibility_loss
    return {
        "total": total_loss,
        "x_offset": x_loss.detach(),
        "size_proxy": size_loss.detach(),
        "visibility": visibility_loss.detach(),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _safe_metric_or_inf(total: float, count: int) -> float:
    if count <= 0:
        return float("inf")
    return total / float(count)


def save_checkpoint(path: Path, model, args, epoch: int, extra_state=None) -> None:
    state = {
        "epoch": epoch,
        "model_type": args.model_type,
        "height": args.height,
        "width": args.width,
        "input_channels": args.input_channels,
        "state_dict": model.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def build_datasets(args, repo_root: Path):
    image_size = (args.height, args.width)

    if args.model_type == "hybrid_follow":
        if args.input_channels != 1:
            raise ValueError("hybrid_follow requires --input-channels 1.")

        train_ds = COCOFollowRegressionDataset(
            root=str(repo_root / args.data_root),
            ann_file=str(repo_root / args.train_ann),
            transforms=get_train_transforms(
                model_type="hybrid_follow",
                input_channels=1,
                image_size=image_size,
            ),
            image_mode="L",
        )
        val_ds = COCOFollowRegressionDataset(
            root=str(repo_root / args.val_root),
            ann_file=str(repo_root / args.val_ann),
            transforms=get_val_transforms(
                model_type="hybrid_follow",
                input_channels=1,
                image_size=image_size,
            ),
            image_mode="L",
        )
        return train_ds, val_ds, None

    image_mode = "L" if args.input_channels == 1 else "RGB"
    train_ds = COCOPersonDataset(
        root=str(repo_root / args.data_root),
        ann_file=str(repo_root / args.train_ann),
        transforms=get_train_transforms(
            model_type="ssd",
            input_channels=args.input_channels,
            image_size=image_size,
        ),
        image_mode=image_mode,
    )
    val_ds = COCOPersonDataset(
        root=str(repo_root / args.val_root),
        ann_file=str(repo_root / args.val_ann),
        transforms=get_val_transforms(
            model_type="ssd",
            input_channels=args.input_channels,
            image_size=image_size,
        ),
        image_mode=image_mode,
    )
    return train_ds, val_ds, detection_collate_fn


def _default_annotation_path(model_type: str, split: str) -> str:
    if model_type == "hybrid_follow":
        return f"pytorch_ssd/data/coco/annotations/instances_{split}2017.json"
    return f"pytorch_ssd/data/coco/annotations/{split}_person.json"


def resolve_annotation_paths(args):
    train_ann = args.train_ann or _default_annotation_path(args.model_type, "train")
    val_ann = args.val_ann or _default_annotation_path(args.model_type, "val")

    if args.model_type == "hybrid_follow" and not args.allow_person_only_follow_ann:
        for ann_path in (train_ann, val_ann):
            if Path(ann_path).name.endswith("_person.json"):
                raise ValueError(
                    "hybrid_follow must train on full COCO instances annotations so "
                    "true no-person negatives are present. "
                    "Pass --allow-person-only-follow-ann to override."
                )

    return train_ann, val_ann


def build_model(args):
    image_size = (args.height, args.width)

    if args.model_type == "hybrid_follow":
        return HybridFollowNet(
            input_channels=1,
            image_size=image_size,
        )

    return SSDMobileNetV2Raw(
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        image_size=image_size,
        input_channels=args.input_channels,
    )


def resolve_output_dir(args, repo_root: Path) -> Path:
    if args.output_dir:
        return repo_root / args.output_dir

    if args.model_type == "hybrid_follow":
        return repo_root / "pytorch_ssd/training/hybrid_follow"
    return repo_root / "pytorch_ssd/training/person_ssd_pytorch"


def checkpoint_name(model_type: str, epoch: int) -> str:
    if model_type == "hybrid_follow":
        return f"hybrid_follow_epoch_{epoch:03d}.pth"
    return f"ssd_mbv2_epoch_{epoch:03d}.pth"


def train_or_eval_epoch(
    model,
    loader,
    device,
    model_type: str,
    optimizer=None,
    track_follow_metrics: bool = False,
    vis_thresh: float = 0.5,
):
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    running_x = 0.0
    running_size = 0.0
    running_visibility = 0.0
    visibility_bce_sum = 0.0
    follow_sample_count = 0
    x_abs_error_sum = 0.0
    size_abs_error_sum = 0.0
    visible_target_count = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    no_person_fp = 0
    no_person_total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Val", ncols=100)
        for batch in pbar:
            if model_type == "hybrid_follow":
                images, targets = batch
                images = images.to(device)
                follow_targets = targets["follow_target"].to(device)

                predictions = model(images)
                loss_items = compute_hybrid_follow_loss(predictions, follow_targets)
                losses = loss_items["total"]
                loss_value = float(losses.detach().cpu().item())

                if is_train:
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                running_loss += loss_value
                running_x += float(loss_items["x_offset"].cpu().item())
                running_size += float(loss_items["size_proxy"].cpu().item())
                running_visibility += float(loss_items["visibility"].cpu().item())
                if track_follow_metrics:
                    visibility_target = follow_targets[:, 2]
                    pred_visible = torch.sigmoid(predictions[:, 2]) >= vis_thresh
                    target_visible = visibility_target > 0.5
                    batch_size = int(visibility_target.shape[0])

                    visibility_bce_sum += float(
                        F.binary_cross_entropy_with_logits(
                            predictions[:, 2],
                            visibility_target,
                            reduction="sum",
                        ).detach().cpu().item()
                    )
                    follow_sample_count += batch_size
                    tp += int((pred_visible & target_visible).sum().item())
                    fp += int((pred_visible & ~target_visible).sum().item())
                    tn += int((~pred_visible & ~target_visible).sum().item())
                    fn += int((~pred_visible & target_visible).sum().item())

                    # Visibility is still a useful diagnostic, but no-person frames do not
                    # have a meaningful x/size control target. We exclude them from the
                    # follow metrics so checkpoint selection matches the controller's job.
                    if torch.any(target_visible):
                        batch_x_abs_error = torch.abs(
                            predictions[target_visible, 0] - follow_targets[target_visible, 0]
                        )
                        batch_size_abs_error = torch.abs(
                            predictions[target_visible, 1] - follow_targets[target_visible, 1]
                        )
                        visible_count = int(target_visible.sum().item())
                        x_abs_error_sum += float(batch_x_abs_error.sum().detach().cpu().item())
                        size_abs_error_sum += float(
                            batch_size_abs_error.sum().detach().cpu().item()
                        )
                        visible_target_count += visible_count

                    true_no_person = targets.get("true_no_person")
                    if true_no_person is not None:
                        no_person_mask = true_no_person.to(device).view(-1) > 0
                        no_person_total += int(no_person_mask.sum().item())
                        no_person_fp += int((pred_visible & no_person_mask).sum().item())

                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "x": f"{float(loss_items['x_offset'].cpu().item()):.4f}",
                        "size": f"{float(loss_items['size_proxy'].cpu().item()):.4f}",
                        "vis": f"{float(loss_items['visibility'].cpu().item()):.4f}",
                    }
                )
                continue

            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = reduce_losses(loss_dict)
            loss_value = float(losses.detach().cpu().item())

            if is_train:
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            running_loss += loss_value
            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    num_batches = max(len(loader), 1)
    stats = {
        "loss": running_loss / num_batches,
    }
    if model_type == "hybrid_follow":
        stats["x_offset"] = running_x / num_batches
        stats["size_proxy"] = running_size / num_batches
        stats["visibility"] = running_visibility / num_batches
        if track_follow_metrics:
            precision = _safe_div(float(tp), float(tp + fp))
            recall = _safe_div(float(tp), float(tp + fn))
            stats["visibility_bce"] = _safe_div(visibility_bce_sum, float(follow_sample_count))
            stats["accuracy"] = _safe_div(float(tp + tn), float(follow_sample_count))
            stats["precision"] = precision
            stats["recall"] = recall
            stats["f1"] = _safe_div(2.0 * precision * recall, precision + recall)
            stats["no_person_fp_rate"] = _safe_div(float(no_person_fp), float(no_person_total))
            stats["no_person_fp"] = float(no_person_fp)
            stats["no_person_total"] = float(no_person_total)
            stats["visible_target_count"] = float(visible_target_count)
            stats["x_mae"] = _safe_metric_or_inf(x_abs_error_sum, visible_target_count)
            stats["size_mae"] = _safe_metric_or_inf(size_abs_error_sum, visible_target_count)
            stats["follow_score"] = (
                stats["x_mae"] + (0.3 * stats["size_mae"])
                if visible_target_count > 0
                else float("inf")
            )
            # x_offset is the primary control signal downstream, so the dedicated
            # control score is the visible-only x error and ignores no-person negatives.
            stats["control_score"] = stats["x_mae"]
    return stats


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.train_ann, args.val_ann = resolve_annotation_paths(args)

    if args.model_type == "hybrid_follow":
        print(f"Hybrid-follow train annotations: {args.train_ann}")
        print(f"Hybrid-follow val annotations: {args.val_ann}")

    train_ds, val_ds, collate_fn = build_datasets(args, repo_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(args).to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = resolve_output_dir(args, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_x_mae = float("inf")
    best_x_epoch = 0
    best_follow_score = float("inf")
    best_follow_score_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_or_eval_epoch(
            model=model,
            loader=train_loader,
            device=device,
            model_type=args.model_type,
            optimizer=optimizer,
        )
        scheduler.step()

        val_stats = train_or_eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            model_type=args.model_type,
            optimizer=None,
            track_follow_metrics=args.model_type == "hybrid_follow",
            vis_thresh=args.vis_thresh,
        )

        if args.model_type == "hybrid_follow":
            print(
                "Train loss: {loss:.4f} (x={x_offset:.4f}, size={size_proxy:.4f}, vis={visibility:.4f})".format(
                    **train_stats
                )
            )
            print(
                "Val loss: {loss:.4f} (x={x_offset:.4f}, size={size_proxy:.4f}, vis={visibility:.4f})".format(
                    **val_stats
                )
            )
            print("Val follow metrics:")
            print(f"  x_mae={val_stats['x_mae']:.4f}")
            print(f"  size_mae={val_stats['size_mae']:.4f}")
            print(f"  follow_score={val_stats['follow_score']:.4f}")
            print(f"  control_score={val_stats['control_score']:.4f}")
            print(
                "Val visibility @ {thresh:.2f}: bce={bce:.4f}, acc={acc:.4f}, "
                "precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, "
                "no_person_fp_rate={fp_rate:.4f} ({fp_count:.0f}/{fp_total:.0f})".format(
                    thresh=args.vis_thresh,
                    bce=val_stats["visibility_bce"],
                    acc=val_stats["accuracy"],
                    precision=val_stats["precision"],
                    recall=val_stats["recall"],
                    f1=val_stats["f1"],
                    fp_rate=val_stats["no_person_fp_rate"],
                    fp_count=val_stats["no_person_fp"],
                    fp_total=val_stats["no_person_total"],
                )
            )
        else:
            print(f"Train loss: {train_stats['loss']:.4f}")
            print(f"Val loss (approx): {val_stats['loss']:.4f}")

        ckpt_path = output_dir / checkpoint_name(args.model_type, epoch)
        save_checkpoint(
            ckpt_path,
            model,
            args,
            epoch,
            extra_state={
                "train_stats": train_stats,
                "val_stats": val_stats,
            },
        )
        print(f"Saved checkpoint to {ckpt_path}")

        if args.model_type == "hybrid_follow":
            # High visibility confidence does not guarantee the drone can steer well.
            # For person-follow control, horizontal centering error is the primary
            # selection target, with size acting only as a secondary tie-breaker.
            current_x_mae = val_stats["x_mae"]
            if current_x_mae < best_x_mae:
                best_x_mae = current_x_mae
                best_x_epoch = epoch
                best_x_path = output_dir / "hybrid_follow_best_x.pth"
                save_checkpoint(
                    best_x_path,
                    model,
                    args,
                    epoch,
                    extra_state={
                        "train_stats": train_stats,
                        "val_stats": val_stats,
                        "best_metric": "x_mae",
                        "best_metric_value": current_x_mae,
                        "selection_metrics": {
                            "x_mae": val_stats["x_mae"],
                            "size_mae": val_stats["size_mae"],
                            "follow_score": val_stats["follow_score"],
                            "control_score": val_stats["control_score"],
                        },
                    },
                )
                print(
                    "Updated best x-error checkpoint: "
                    f"epoch={epoch}, val_x_mae={current_x_mae:.4f}, "
                    f"path={best_x_path}"
                )

            current_follow_score = val_stats["follow_score"]
            if current_follow_score < best_follow_score:
                best_follow_score = current_follow_score
                best_follow_score_epoch = epoch
                best_follow_score_path = output_dir / "hybrid_follow_best_follow_score.pth"
                save_checkpoint(
                    best_follow_score_path,
                    model,
                    args,
                    epoch,
                    extra_state={
                        "train_stats": train_stats,
                        "val_stats": val_stats,
                        "best_metric": "follow_score",
                        "best_metric_value": current_follow_score,
                        "selection_metrics": {
                            "x_mae": val_stats["x_mae"],
                            "size_mae": val_stats["size_mae"],
                            "follow_score": val_stats["follow_score"],
                            "control_score": val_stats["control_score"],
                        },
                    },
                )
                print(
                    "Updated best follow-score checkpoint: "
                    f"epoch={epoch}, val_follow_score={current_follow_score:.4f}, "
                    f"path={best_follow_score_path}"
                )

            print(
                "Best x-error so far: "
                f"epoch={best_x_epoch}, val_x_mae={best_x_mae:.4f}"
            )
            print(
                "Best follow score so far: "
                f"epoch={best_follow_score_epoch}, val_follow_score={best_follow_score:.4f}"
            )


if __name__ == "__main__":
    main()

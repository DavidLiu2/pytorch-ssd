from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.follow_model_factory import (
    build_follow_model,
    checkpoint_state_dict,
    follow_checkpoint_metadata,
)
from models.hybrid_follow_net import (
    HYBRID_FOLLOW_STAGE4_VARIANTS,
    HybridFollowNet,
    adapt_hybrid_follow_state_dict_to_model,
    checkpoint_stage4_ablation_value,
    normalize_stage4_variant,
)
from models.ssd_mobilenet_v2_raw import SSDMobileNetV2Raw
from models.quant_native_follow_net import (
    DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    QUANT_NATIVE_FOLLOW_STEM_MODES,
    normalize_quant_native_follow_stem_mode,
)
from utils.coco_follow_regression import COCOFollowRegressionDataset
from utils.coco_person import COCOPersonDataset, detection_collate_fn
from utils.follow_task import (
    FOLLOW_HEAD_TYPES,
    compute_follow_metrics,
    compute_follow_task_loss,
    follow_model_default_stage_channels,
    is_follow_model_type,
    is_quant_native_follow_model_type,
    resolve_follow_head_type,
    visibility_confidence_from_outputs,
)
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
    ap.add_argument(
        "--train-sample-manifest",
        type=str,
        default=None,
        help="Optional calibration-style manifest used to restrict the training set to a ranked image subset.",
    )
    ap.add_argument("--val_root", type=str, default="pytorch_ssd/data/coco/images/val2017")
    ap.add_argument(
        "--val_ann",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--val-sample-manifest",
        type=str,
        default=None,
        help="Optional calibration-style manifest used to restrict the validation set to a ranked image subset.",
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default="hybrid_follow",
        choices=["ssd", "hybrid_follow", "plain_follow", "plain_follow_bin", "plain_follow_v2", "plain_follow_tiny", "dronet_lite_follow"],
    )
    ap.add_argument(
        "--follow-head-type",
        type=str,
        default=None,
        choices=list(FOLLOW_HEAD_TYPES),
        help="Output contract for quant-native follow models.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--width-mult", type=float, default=0.1)
    ap.add_argument("--stem-channels", type=int, default=16)
    ap.add_argument(
        "--stem-mode",
        type=str,
        default=DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
        choices=list(QUANT_NATIVE_FOLLOW_STEM_MODES),
        help="Stem architecture for quant-native follow models.",
    )
    ap.add_argument(
        "--stage-channels",
        type=str,
        default=None,
        help="Comma-separated stage channel sizes. Use 4 values for hybrid_follow and 3 for quant-native follow models.",
    )
    ap.add_argument(
        "--stage4-variant",
        type=str,
        default=None,
        choices=list(HYBRID_FOLLOW_STAGE4_VARIANTS),
        help="Late-stage hybrid_follow architecture variant.",
    )
    ap.add_argument(
        "--stage4-1-ablation",
        type=str,
        default=None,
        choices=[
            "none",
            "single_conv",
            "plain_non_residual",
            "single_conv_non_residual",
            "narrow_stage4",
        ],
        help="Legacy alias for --stage4-variant.",
    )
    ap.add_argument("--stage4-heads-only", action="store_true")
    ap.add_argument("--stem-heads-only", action="store_true")
    ap.add_argument(
        "--trainable-module-prefixes",
        type=str,
        default=None,
        help="Comma-separated parameter prefixes to keep trainable while freezing everything else.",
    )
    ap.add_argument("--quant-aware-finetune", action="store_true")
    ap.add_argument("--qat-bits", type=int, default=8)
    ap.add_argument("--qat-calib-batches", type=int, default=16)
    ap.add_argument(
        "--qat-train-activation-modules",
        type=str,
        default=None,
        help="Comma-separated PACT activation module names to fine-tune exclusively during QAT.",
    )
    ap.add_argument("--activation-range-regularization", action="store_true")
    ap.add_argument("--activation-range-reg-weight", type=float, default=1e-3)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
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
    ap.add_argument(
        "--vis-threshold-sweep",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated visibility thresholds evaluated on validation for follow models.",
    )
    ap.add_argument(
        "--follow-visible-fraction",
        type=float,
        default=None,
        help="Target visible fraction for balanced follow batches. Defaults are model-specific.",
    )
    ap.add_argument(
        "--hard-negative-start-epoch",
        type=int,
        default=None,
        help="Epoch to start hard-negative mining for no-person samples. Defaults are model-specific.",
    )
    ap.add_argument(
        "--hard-negative-boost",
        type=float,
        default=None,
        help="How strongly hard negatives are oversampled/reweighted. Defaults are model-specific.",
    )
    ap.add_argument(
        "--hard-negative-ema",
        type=float,
        default=None,
        help="EMA factor for tracked hard-negative visibility confidence. Defaults are model-specific.",
    )
    ap.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Optional override for the dronet_lite_follow gating-focused warmup phase.",
    )
    ap.add_argument(
        "--disable-dronet-two-phase",
        action="store_true",
        help="Disable the default two-phase dronet_lite_follow loss schedule.",
    )
    args = ap.parse_args()
    if args.qat_train_activation_modules:
        args.qat_train_activation_modules = tuple(
            module_name.strip()
            for module_name in args.qat_train_activation_modules.split(",")
            if module_name.strip()
        )
    else:
        args.qat_train_activation_modules = ()
    if args.trainable_module_prefixes:
        args.trainable_module_prefixes = tuple(
            prefix.strip()
            for prefix in args.trainable_module_prefixes.split(",")
            if prefix.strip()
        )
    else:
        args.trainable_module_prefixes = ()
    if args.model_type == "hybrid_follow":
        args.stage4_variant = normalize_stage4_variant(
            stage4_variant=args.stage4_variant,
            stage4_1_ablation=args.stage4_1_ablation,
        )
        args.stage4_1_ablation = checkpoint_stage4_ablation_value(args.stage4_variant)
    else:
        args.stage4_variant = None
        args.stage4_1_ablation = "none"
    if args.stage_channels:
        parsed_stage_channels = tuple(
            int(value.strip())
            for value in str(args.stage_channels).split(",")
            if value.strip()
        )
    else:
        parsed_stage_channels = None
    if args.model_type == "hybrid_follow":
        default_stage_channels = tuple(HYBRID_FOLLOW_BASE_STAGE_CHANNELS)
        expected_stage_channel_count = len(default_stage_channels)
    elif is_quant_native_follow_model_type(args.model_type):
        default_stage_channels = follow_model_default_stage_channels(args.model_type)
        expected_stage_channel_count = len(default_stage_channels)
    else:
        default_stage_channels = None
        expected_stage_channel_count = None
    if expected_stage_channel_count is not None:
        if parsed_stage_channels is not None and len(parsed_stage_channels) != expected_stage_channel_count:
            raise ValueError(
                f"{args.model_type} expects {expected_stage_channel_count} stage channels, got {parsed_stage_channels}"
            )
        args.stage_channels = parsed_stage_channels or default_stage_channels
    else:
        args.stage_channels = None
    args.follow_head_type = resolve_follow_head_type(
        args.follow_head_type,
        model_type=args.model_type,
    )
    if is_quant_native_follow_model_type(args.model_type):
        args.stem_mode = normalize_quant_native_follow_stem_mode(args.stem_mode)
    else:
        args.stem_mode = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE
    raw_threshold_sweep = tuple(
        float(value.strip())
        for value in str(args.vis_threshold_sweep).split(",")
        if value.strip()
    )
    args.vis_threshold_sweep = tuple(sorted({float(value) for value in raw_threshold_sweep}))
    return args


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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _safe_metric_or_inf(total: float, count: int) -> float:
    if count <= 0:
        return float("inf")
    return total / float(count)


def _clamp_unit_interval(value: float, *, default: float) -> float:
    if value is None:
        return float(default)
    return min(max(float(value), 0.05), 0.95)


def _copy_loss_weights(weights: dict[str, float] | None) -> dict[str, float] | None:
    if weights is None:
        return None
    return {key: float(value) for key, value in weights.items()}


def build_follow_training_policy(args) -> dict[str, Any]:
    if args.model_type in {"plain_follow", "plain_follow_bin", "plain_follow_v2", "plain_follow_tiny"}:
        visible_fraction = _clamp_unit_interval(
            args.follow_visible_fraction,
            default=0.60,
        )
        hard_negative_start_epoch = int(
            args.hard_negative_start_epoch
            if args.hard_negative_start_epoch is not None
            else max(4, args.epochs // 5)
        )
        hard_negative_boost = float(
            args.hard_negative_boost if args.hard_negative_boost is not None else 1.5
        )
        hard_negative_ema = float(
            args.hard_negative_ema if args.hard_negative_ema is not None else 0.70
        )
        return {
            "sampler_enabled": True,
            "visible_fraction": visible_fraction,
            "phase1_epochs": 0,
            "hard_negative_start_epoch": hard_negative_start_epoch,
            "hard_negative_boost": hard_negative_boost,
            "hard_negative_ema": hard_negative_ema,
            "base_loss_weights": {
                "visibility": 2.0,
                "x": 1.0,
                "size": 0.3,
                "residual": 0.5,
            },
            "phase1_loss_weights": None,
        }

    if args.model_type == "dronet_lite_follow":
        phase1_epochs = 0
        if not args.disable_dronet_two_phase:
            phase1_epochs = int(
                args.phase1_epochs if args.phase1_epochs is not None else max(4, args.epochs // 5)
            )
        visible_fraction = _clamp_unit_interval(
            args.follow_visible_fraction,
            default=0.50,
        )
        hard_negative_start_epoch = int(
            args.hard_negative_start_epoch
            if args.hard_negative_start_epoch is not None
            else max(phase1_epochs + 1, 4)
        )
        hard_negative_boost = float(
            args.hard_negative_boost if args.hard_negative_boost is not None else 2.0
        )
        hard_negative_ema = float(
            args.hard_negative_ema if args.hard_negative_ema is not None else 0.75
        )
        return {
            "sampler_enabled": True,
            "visible_fraction": visible_fraction,
            "phase1_epochs": phase1_epochs,
            "hard_negative_start_epoch": hard_negative_start_epoch,
            "hard_negative_boost": hard_negative_boost,
            "hard_negative_ema": hard_negative_ema,
            "base_loss_weights": {
                "visibility": 2.5,
                "x": 1.0,
                "size": 0.3,
                "residual": 0.5,
            },
            "phase1_loss_weights": {
                "visibility": 3.5,
                "x": 0.75,
                "size": 0.2,
                "residual": 0.25,
            },
        }

    return {
        "sampler_enabled": False,
        "visible_fraction": None,
        "phase1_epochs": 0,
        "hard_negative_start_epoch": args.epochs + 1,
        "hard_negative_boost": 0.0,
        "hard_negative_ema": 0.70,
        "base_loss_weights": None,
        "phase1_loss_weights": None,
    }


def resolve_follow_epoch_policy(policy: dict[str, Any], epoch: int) -> dict[str, Any]:
    phase1_epochs = int(policy.get("phase1_epochs", 0) or 0)
    if phase1_epochs > 0 and int(epoch) <= phase1_epochs and policy.get("phase1_loss_weights"):
        return {
            "phase_name": "phase1_visibility_focus",
            "loss_weights": _copy_loss_weights(policy["phase1_loss_weights"]),
            "hard_negative_active": int(epoch) >= int(policy["hard_negative_start_epoch"]),
        }
    return {
        "phase_name": "phase2_full_multitask" if phase1_epochs > 0 else "single_phase",
        "loss_weights": _copy_loss_weights(policy.get("base_loss_weights")),
        "hard_negative_active": int(epoch) >= int(policy["hard_negative_start_epoch"]),
    }


def _center_crop_person_visible_after_preprocess(
    anns,
    *,
    person_cat_ids: set[int],
    image_width: int,
    image_height: int,
) -> tuple[bool, bool]:
    crop_size = min(int(image_width), int(image_height))
    crop_left = (int(image_width) - crop_size) // 2
    crop_top = (int(image_height) - crop_size) // 2

    true_no_person = True
    visible_after_crop = False
    for ann in anns:
        if ann.get("category_id") not in person_cat_ids:
            continue
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if float(w) <= 0.0 or float(h) <= 0.0:
            continue
        true_no_person = False
        x1 = float(x) - float(crop_left)
        y1 = float(y) - float(crop_top)
        x2 = x1 + float(w)
        y2 = y1 + float(h)
        x1 = max(0.0, min(float(crop_size), x1))
        y1 = max(0.0, min(float(crop_size), y1))
        x2 = max(0.0, min(float(crop_size), x2))
        y2 = max(0.0, min(float(crop_size), y2))
        if x2 > x1 and y2 > y1:
            visible_after_crop = True
            break
    return visible_after_crop, true_no_person


def build_follow_sampling_metadata(dataset: COCOFollowRegressionDataset) -> dict[str, Any]:
    person_cat_ids = {int(cat_id) for cat_id in dataset.person_cat_ids}
    image_ids: list[int] = []
    visible_flags: list[bool] = []
    true_no_person_flags: list[bool] = []

    for img_id in dataset.img_ids:
        image_info = dataset.coco.imgs[int(img_id)]
        anns = dataset.coco.imgToAnns.get(int(img_id), [])
        visible_after_crop, true_no_person = _center_crop_person_visible_after_preprocess(
            anns,
            person_cat_ids=person_cat_ids,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
        )
        image_ids.append(int(img_id))
        visible_flags.append(bool(visible_after_crop))
        true_no_person_flags.append(bool(true_no_person))

    visible_tensor = torch.as_tensor(visible_flags, dtype=torch.bool)
    true_no_person_tensor = torch.as_tensor(true_no_person_flags, dtype=torch.bool)
    return {
        "image_ids": image_ids,
        "visible_flags": visible_tensor,
        "true_no_person_flags": true_no_person_tensor,
        "visible_count": int(visible_tensor.sum().item()),
        "not_visible_count": int((~visible_tensor).sum().item()),
        "true_no_person_count": int(true_no_person_tensor.sum().item()),
        "crop_negative_count": int((~visible_tensor & ~true_no_person_tensor).sum().item()),
    }


def build_follow_epoch_sampler(
    *,
    dataset: COCOFollowRegressionDataset,
    sampling_metadata: dict[str, Any],
    visible_fraction: float,
    epoch: int,
    hard_negative_start_epoch: int,
    hard_negative_boost: float,
    hard_negative_image_scores: dict[int, float] | None,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    num_samples = len(dataset)
    weights = torch.ones(num_samples, dtype=torch.double)
    visible_flags = sampling_metadata["visible_flags"]
    not_visible_flags = ~visible_flags
    true_no_person_flags = sampling_metadata["true_no_person_flags"]

    visible_count = int(visible_flags.sum().item())
    not_visible_count = int(not_visible_flags.sum().item())
    target_visible_fraction = _clamp_unit_interval(visible_fraction, default=0.50)

    if visible_count > 0 and not_visible_count > 0:
        weights[visible_flags] = target_visible_fraction / float(visible_count)
        weights[not_visible_flags] = (1.0 - target_visible_fraction) / float(not_visible_count)
    elif visible_count > 0:
        weights.fill_(1.0 / float(visible_count))
    else:
        weights.fill_(1.0 / float(max(not_visible_count, 1)))

    hard_negative_active = int(epoch) >= int(hard_negative_start_epoch) and float(hard_negative_boost) > 0.0
    boosted_negative_count = 0
    boosted_score_total = 0.0
    if hard_negative_active and hard_negative_image_scores:
        for sample_index, img_id in enumerate(sampling_metadata["image_ids"]):
            if not bool(true_no_person_flags[sample_index].item()):
                continue
            score = hard_negative_image_scores.get(int(img_id))
            if score is None:
                continue
            clamped_score = min(max(float(score), 0.0), 1.0)
            weights[sample_index] *= 1.0 + float(hard_negative_boost) * clamped_score
            boosted_negative_count += 1
            boosted_score_total += clamped_score

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
    total_weight = float(weights.sum().item())
    expected_visible_fraction = 0.0
    if total_weight > 0.0 and visible_count > 0:
        expected_visible_fraction = float(weights[visible_flags].sum().item()) / total_weight
    expected_true_no_person_fraction = 0.0
    if total_weight > 0.0 and int(true_no_person_flags.sum().item()) > 0:
        expected_true_no_person_fraction = float(weights[true_no_person_flags].sum().item()) / total_weight

    summary = {
        "target_visible_fraction": float(target_visible_fraction),
        "expected_visible_fraction": float(expected_visible_fraction),
        "expected_true_no_person_fraction": float(expected_true_no_person_fraction),
        "visible_count": visible_count,
        "not_visible_count": not_visible_count,
        "true_no_person_count": int(true_no_person_flags.sum().item()),
        "hard_negative_active": hard_negative_active,
        "boosted_true_no_person_count": int(boosted_negative_count),
        "boosted_true_no_person_mean_score": _safe_div(
            boosted_score_total,
            float(boosted_negative_count),
        ),
    }
    return sampler, summary


def _threshold_selection_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics["f1"])
        + (0.25 * float(metrics["precision"]))
        - (0.75 * float(metrics["no_person_fp_rate"]))
    )


def save_checkpoint(path: Path, model, args, epoch: int, extra_state=None) -> None:
    state = {
        "epoch": epoch,
        "model_type": args.model_type,
        "state_dict": model.state_dict(),
    }
    if is_follow_model_type(args.model_type):
        state.update(
            follow_checkpoint_metadata(
                model_type=args.model_type,
                model=model,
                input_channels=args.input_channels,
                image_size=(args.height, args.width),
            )
        )
    else:
        state.update(
            {
                "height": args.height,
                "width": args.width,
                "input_channels": args.input_channels,
            }
        )
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def _manifest_selected_image_ids(payload: dict[str, Any]) -> list[int]:
    ordered = list(payload.get("ordered_samples") or [])
    target_count = int(payload.get("target_count") or 0)
    selected = [
        row
        for row in ordered
        if row.get("selected_rank") is not None
        and (target_count <= 0 or int(row["selected_rank"]) < target_count)
    ]
    if not selected and target_count > 0:
        selected = ordered[:target_count]
    elif not selected:
        selected = ordered

    image_ids: list[int] = []
    seen: set[int] = set()
    for row in selected:
        raw_image_id = row.get("image_id")
        if raw_image_id is None:
            continue
        image_id = int(raw_image_id)
        if image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)
    return image_ids


def apply_follow_dataset_manifest_filter(
    dataset: COCOFollowRegressionDataset,
    manifest_path: Path,
    *,
    split_name: str,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = _manifest_selected_image_ids(payload)
    if not manifest_ids:
        raise ValueError(
            f"{split_name} sample manifest did not contain any selected image_ids: {manifest_path}"
        )

    dataset_ids = {int(image_id) for image_id in dataset.img_ids}
    filtered_ids = [image_id for image_id in manifest_ids if image_id in dataset_ids]
    if not filtered_ids:
        raise ValueError(
            f"{split_name} sample manifest image_ids do not overlap the dataset: {manifest_path}"
        )

    previous_count = len(dataset.img_ids)
    dataset.img_ids = filtered_ids
    summary = {
        "split": split_name,
        "manifest_path": str(manifest_path),
        "previous_count": int(previous_count),
        "selected_count": int(len(filtered_ids)),
        "dropped_count": int(previous_count - len(filtered_ids)),
        "manifest_target_count": int(payload.get("target_count") or len(filtered_ids)),
    }
    print(
        f"Applied {split_name} sample manifest: path={manifest_path}, "
        f"selected={summary['selected_count']}/{summary['previous_count']}"
    )
    return summary


def build_datasets(args, repo_root: Path):
    image_size = (args.height, args.width)

    if is_follow_model_type(args.model_type):
        if args.input_channels != 1:
            raise ValueError(f"{args.model_type} requires --input-channels 1.")

        train_ds = COCOFollowRegressionDataset(
            root=str(repo_root / args.data_root),
            ann_file=str(repo_root / args.train_ann),
            transforms=get_train_transforms(
                model_type=args.model_type,
                input_channels=1,
                image_size=image_size,
            ),
            image_mode="L",
        )
        val_ds = COCOFollowRegressionDataset(
            root=str(repo_root / args.val_root),
            ann_file=str(repo_root / args.val_ann),
            transforms=get_val_transforms(
                model_type=args.model_type,
                input_channels=1,
                image_size=image_size,
            ),
            image_mode="L",
        )
        if args.train_sample_manifest:
            apply_follow_dataset_manifest_filter(
                train_ds,
                (repo_root / args.train_sample_manifest).resolve(),
                split_name="train",
            )
        if args.val_sample_manifest:
            apply_follow_dataset_manifest_filter(
                val_ds,
                (repo_root / args.val_sample_manifest).resolve(),
                split_name="val",
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
    if is_follow_model_type(model_type):
        return f"pytorch_ssd/data/coco/annotations/instances_{split}2017.json"
    return f"pytorch_ssd/data/coco/annotations/{split}_person.json"


def resolve_annotation_paths(args):
    train_ann = args.train_ann or _default_annotation_path(args.model_type, "train")
    val_ann = args.val_ann or _default_annotation_path(args.model_type, "val")

    if is_follow_model_type(args.model_type) and not args.allow_person_only_follow_ann:
        for ann_path in (train_ann, val_ann):
            if Path(ann_path).name.endswith("_person.json"):
                raise ValueError(
                    f"{args.model_type} must train on full COCO instances annotations so "
                    "true no-person negatives are present. "
                    "Pass --allow-person-only-follow-ann to override."
                )

    return train_ann, val_ann


def build_model(args):
    image_size = (args.height, args.width)

    if is_follow_model_type(args.model_type):
        return build_follow_model(
            model_type=args.model_type,
            input_channels=1,
            image_size=image_size,
            follow_head_type=args.follow_head_type,
            stage4_variant=args.stage4_variant,
            stage4_1_ablation=args.stage4_1_ablation,
            stage_channels=args.stage_channels,
            stem_channels=args.stem_channels,
            stem_mode=args.stem_mode,
        )

    return SSDMobileNetV2Raw(
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        image_size=image_size,
        input_channels=args.input_channels,
    )


def load_init_checkpoint(model, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)
    adaptation_report = []
    if isinstance(model, HybridFollowNet):
        state_dict, adaptation_report = adapt_hybrid_follow_state_dict_to_model(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded init checkpoint: {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if adaptation_report:
        print(
            "  Adapted hybrid_follow tensors for target architecture "
            f"(first 10): {adaptation_report[:10]}"
        )
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    return checkpoint


def apply_stage4_heads_only_freeze(model) -> None:
    if not isinstance(model, HybridFollowNet):
        raise ValueError("--stage4-heads-only is only supported for HybridFollowNet.")
    trainable_prefixes = (
        "stage4.",
        "head_x.",
        "head_size.",
        "head_vis.",
        "head.",
    )
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        requires_grad = name.startswith(trainable_prefixes)
        param.requires_grad = requires_grad
        if requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(
        "Applied stage4+heads freeze: "
        f"trainable_params={trainable}, frozen_params={frozen}"
    )


def apply_stem_heads_only_freeze(model) -> None:
    model_type = str(getattr(model, "model_type", ""))
    if not is_quant_native_follow_model_type(model_type):
        raise ValueError("--stem-heads-only is only supported for quant-native follow models.")
    apply_trainable_prefix_freeze(model, ("stem.", "output_head."))


def resolve_dotted_module_path(model, module_name: str):
    current = model
    for part in module_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def apply_trainable_prefix_freeze(model, prefixes) -> None:
    prefixes = tuple(str(prefix) for prefix in prefixes if str(prefix))
    if not prefixes:
        raise ValueError("apply_trainable_prefix_freeze requires at least one non-empty prefix.")

    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        requires_grad = any(name.startswith(prefix) for prefix in prefixes)
        param.requires_grad = requires_grad
        if requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    print(
        "Applied prefix freeze: "
        f"prefixes={prefixes}, trainable_params={trainable}, frozen_params={frozen}"
    )


def apply_activation_module_freeze(model, module_names) -> None:
    module_names = tuple(dict.fromkeys(module_names))
    trainable_param_names = set()
    for module_name in module_names:
        module = resolve_dotted_module_path(model, module_name)
        alpha = getattr(module, "alpha", None)
        if not torch.is_tensor(alpha):
            raise ValueError(
                f"Activation-only QAT expected an alpha parameter at {module_name}, "
                f"found {module.__class__.__name__}"
            )
        trainable_param_names.add(f"{module_name}.alpha")

    trainable = 0
    frozen = 0
    matched = set()
    for name, param in model.named_parameters():
        requires_grad = name in trainable_param_names
        param.requires_grad = requires_grad
        if requires_grad:
            trainable += param.numel()
            matched.add(name)
        else:
            frozen += param.numel()

    missing = sorted(trainable_param_names - matched)
    if missing:
        raise ValueError(f"Could not match activation-only QAT parameters: {missing}")

    print(
        "Applied activation-only QAT freeze: "
        f"modules={module_names}, trainable_params={trainable}, frozen_params={frozen}"
    )


def collect_qat_calib_samples(train_loader, device, max_batches: int):
    samples = []
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max(int(max_batches), 0):
            break
        images, _targets = batch
        images = images.to(device)
        for image_idx in range(images.shape[0]):
            samples.append({"tensor": images[image_idx:image_idx + 1].detach()})
    return samples


def enable_quant_aware_finetune(model, train_loader, device, args):
    if not args.quant_aware_finetune:
        return model
    if not is_follow_model_type(args.model_type):
        raise ValueError("--quant-aware-finetune is only supported for follow models.")

    import nemo
    import sys
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent
    exporter_dir = project_dir / "nemo"
    if str(exporter_dir) not in sys.path:
        sys.path.insert(0, str(exporter_dir))

    from export_nemo_quant import (
        patch_model_to_graph_compat,
        resolve_dotted_module,
        run_activation_calibration,
    )

    patch_model_to_graph_compat()
    dummy_input = torch.randn(
        1,
        args.input_channels,
        args.height,
        args.width,
        device=device,
    )
    model_q = nemo.transform.quantize_pact(model, dummy_input=dummy_input)
    model_q.to(device)
    model_q.change_precision(
        bits=args.qat_bits,
        scale_weights=True,
        scale_activations=True,
    )
    calib_samples = collect_qat_calib_samples(train_loader, device, args.qat_calib_batches)
    if calib_samples:
        model_q.eval()
        run_activation_calibration(model_q, calib_samples)
        model_q.train()

    range_reg_modules = (
        tuple(args.qat_train_activation_modules)
        if args.qat_train_activation_modules
        else (
            ("stage4.0.out_relu", "stage4.1.relu1", "stage4.1.out_relu")
            if args.model_type == "hybrid_follow"
            else (
                "stem.relu",
                "stem.post.relu",
                "stage1.downsample.relu",
                "stage1.refine.relu",
                "stage2.downsample.relu",
                "stage2.refine.relu",
                "stage3.downsample.relu",
                "stage3.refine.relu",
            )
        )
    )
    for module_name in range_reg_modules:
        try:
            module = resolve_dotted_module(model_q, module_name)
        except (AttributeError, IndexError, KeyError):
            continue
        alpha = getattr(module, "alpha", None)
        if torch.is_tensor(alpha):
            module._range_reg_target_alpha = float(alpha.detach().cpu().item())

    print(
        "Enabled quant-aware fine-tune: "
        f"bits={args.qat_bits}, calib_batches={args.qat_calib_batches}"
    )
    return model_q


def activation_range_regularization_loss(model, weight: float):
    reg_loss = None
    for _name, module in model.named_modules():
        target_alpha = getattr(module, "_range_reg_target_alpha", None)
        alpha = getattr(module, "alpha", None)
        if target_alpha is None or not torch.is_tensor(alpha):
            continue
        term = torch.square(alpha - float(target_alpha)).sum()
        reg_loss = term if reg_loss is None else reg_loss + term
    if reg_loss is None:
        return None
    return reg_loss * float(weight)


def resolve_output_dir(args, repo_root: Path) -> Path:
    if args.output_dir:
        return repo_root / args.output_dir

    if args.model_type == "hybrid_follow":
        return repo_root / "pytorch_ssd/training/hybrid_follow"
    if is_quant_native_follow_model_type(args.model_type):
        stem_suffix = ""
        if args.stem_mode != DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE:
            stem_suffix = f"_{args.stem_mode}"
        return repo_root / "pytorch_ssd" / "training" / f"{args.model_type}_{args.follow_head_type}{stem_suffix}"
    return repo_root / "pytorch_ssd/training/person_ssd_pytorch"


def checkpoint_name(model_type: str, epoch: int) -> str:
    if is_follow_model_type(model_type):
        return f"{model_type}_epoch_{epoch:03d}.pth"
    return f"ssd_mbv2_epoch_{epoch:03d}.pth"


def train_or_eval_epoch(
    model,
    loader,
    device,
    model_type: str,
    follow_head_type: str | None = None,
    optimizer=None,
    track_follow_metrics: bool = False,
    vis_thresh: float = 0.5,
    extra_train_loss_fn=None,
    max_batches: Optional[int] = None,
    loss_weights: dict[str, float] | None = None,
    vis_threshold_sweep: tuple[float, ...] | None = None,
    hard_negative_image_scores: dict[int, float] | None = None,
    hard_negative_boost: float = 0.0,
    hard_negative_ema: float = 0.70,
):
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    running_x = 0.0
    running_size = 0.0
    running_visibility = 0.0
    running_x_residual = 0.0
    running_extra_train_loss = 0.0
    running_negative_visibility_confidence = 0.0
    running_negative_visibility_count = 0
    max_batches = None if max_batches is None or int(max_batches) <= 0 else int(max_batches)
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    processed_batches = 0
    follow_outputs_epoch = []
    follow_targets_epoch = []
    follow_true_no_person_epoch = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Val", ncols=100, total=total_batches)
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            processed_batches += 1
            if is_follow_model_type(model_type):
                images, targets = batch
                images = images.to(device)
                follow_targets = targets["follow_target"].to(device)
                image_ids = targets.get("image_id")
                true_no_person = targets.get("true_no_person")

                predictions = model(images)
                visibility_sample_weights = None
                if true_no_person is not None:
                    negative_mask = true_no_person.view(-1).to(device=device) > 0
                    if torch.any(negative_mask):
                        negative_visibility_confidence = visibility_confidence_from_outputs(
                            predictions[negative_mask],
                            follow_head_type,
                            model_type=model_type,
                        )
                        running_negative_visibility_confidence += float(
                            negative_visibility_confidence.detach().sum().cpu().item()
                        )
                        running_negative_visibility_count += int(negative_mask.sum().item())
                        if float(hard_negative_boost) > 0.0:
                            visibility_sample_weights = torch.ones(
                                predictions.shape[0],
                                device=device,
                                dtype=predictions.dtype,
                            )
                            visibility_sample_weights[negative_mask] = 1.0 + (
                                float(hard_negative_boost) * negative_visibility_confidence.detach()
                            )
                loss_items = compute_follow_task_loss(
                    predictions,
                    follow_targets,
                    follow_head_type,
                    model_type=model_type,
                    loss_weights=loss_weights,
                    visibility_sample_weights=visibility_sample_weights,
                )
                losses = loss_items["total"]
                extra_train_loss_value = 0.0
                if is_train and extra_train_loss_fn is not None:
                    extra_train_loss = extra_train_loss_fn(model)
                    if extra_train_loss is not None:
                        losses = losses + extra_train_loss
                        extra_train_loss_value = float(extra_train_loss.detach().cpu().item())
                loss_value = float(losses.detach().cpu().item())

                if is_train:
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                running_loss += loss_value
                running_x += float(loss_items["x"].cpu().item())
                running_size += float(loss_items["size"].cpu().item())
                running_visibility += float(loss_items["visibility"].cpu().item())
                running_x_residual += float(
                    loss_items.get("x_residual", predictions.new_zeros(())).cpu().item()
                )
                running_extra_train_loss += extra_train_loss_value
                if true_no_person is not None and image_ids is not None and hard_negative_image_scores is not None:
                    negative_mask_cpu = true_no_person.view(-1).detach().cpu() > 0
                    if torch.any(negative_mask_cpu):
                        visibility_confidence_cpu = visibility_confidence_from_outputs(
                            predictions,
                            follow_head_type,
                            model_type=model_type,
                        ).detach().cpu().view(-1)
                        image_ids_cpu = image_ids.detach().cpu().view(-1)
                        for local_index, is_negative in enumerate(negative_mask_cpu.tolist()):
                            if not is_negative:
                                continue
                            img_id = int(image_ids_cpu[local_index].item())
                            confidence = float(visibility_confidence_cpu[local_index].item())
                            previous = hard_negative_image_scores.get(img_id)
                            if previous is None:
                                updated = confidence
                            else:
                                updated = (
                                    float(hard_negative_ema) * float(previous)
                                    + (1.0 - float(hard_negative_ema)) * confidence
                                )
                            hard_negative_image_scores[img_id] = updated
                if track_follow_metrics:
                    follow_outputs_epoch.append(predictions.detach().cpu())
                    follow_targets_epoch.append(follow_targets.detach().cpu())
                    if true_no_person is not None:
                        follow_true_no_person_epoch.append(true_no_person.detach().cpu().view(-1))

                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "x": f"{float(loss_items['x'].cpu().item()):.4f}",
                        "size": f"{float(loss_items['size'].cpu().item()):.4f}",
                        "vis": f"{float(loss_items['visibility'].cpu().item()):.4f}",
                        "xres": f"{float(loss_items.get('x_residual', predictions.new_zeros(())).cpu().item()):.4f}",
                        "reg": f"{extra_train_loss_value:.4f}",
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

    num_batches = max(processed_batches, 1)
    stats = {
        "loss": running_loss / num_batches,
    }
    if is_follow_model_type(model_type):
        stats["x_offset"] = running_x / num_batches
        stats["size_proxy"] = running_size / num_batches
        stats["visibility"] = running_visibility / num_batches
        stats["x_residual"] = running_x_residual / num_batches
        stats["negative_visibility_confidence"] = _safe_div(
            running_negative_visibility_confidence,
            float(running_negative_visibility_count),
        )
        if extra_train_loss_fn is not None:
            stats["extra_train_loss"] = running_extra_train_loss / num_batches
        if track_follow_metrics and follow_outputs_epoch:
            outputs_epoch = torch.cat(follow_outputs_epoch, dim=0)
            targets_epoch = torch.cat(follow_targets_epoch, dim=0)
            true_no_person_epoch = None
            if follow_true_no_person_epoch:
                true_no_person_epoch = torch.cat(follow_true_no_person_epoch, dim=0)
            stats.update(
                compute_follow_metrics(
                    outputs_epoch,
                    targets_epoch,
                    head_type=follow_head_type,
                    model_type=model_type,
                    vis_thresh=vis_thresh,
                    true_no_person=true_no_person_epoch,
                )
            )
            if vis_threshold_sweep:
                sweep_results = []
                best_result = None
                for sweep_threshold in vis_threshold_sweep:
                    sweep_metrics = compute_follow_metrics(
                        outputs_epoch,
                        targets_epoch,
                        head_type=follow_head_type,
                        model_type=model_type,
                        vis_thresh=float(sweep_threshold),
                        true_no_person=true_no_person_epoch,
                    )
                    sweep_result = {
                        "threshold": float(sweep_threshold),
                        "selection_score": float(_threshold_selection_score(sweep_metrics)),
                        "precision": float(sweep_metrics["precision"]),
                        "recall": float(sweep_metrics["recall"]),
                        "f1": float(sweep_metrics["f1"]),
                        "no_person_fp_rate": float(sweep_metrics["no_person_fp_rate"]),
                        "accuracy": float(sweep_metrics["accuracy"]),
                    }
                    sweep_results.append(sweep_result)
                    ranking_key = (
                        sweep_result["selection_score"],
                        sweep_result["precision"],
                        -sweep_result["no_person_fp_rate"],
                        sweep_result["recall"],
                    )
                    if best_result is None or ranking_key > best_result[0]:
                        best_result = (ranking_key, sweep_result, sweep_metrics)
                stats["vis_threshold_sweep"] = sweep_results
                if best_result is not None:
                    stats["selected_vis_threshold"] = float(best_result[1]["threshold"])
                    stats["selected_vis_threshold_score"] = float(best_result[1]["selection_score"])
                    stats["selected_vis_metrics"] = {
                        key: float(value) for key, value in best_result[2].items()
                    }
    return stats


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.train_ann, args.val_ann = resolve_annotation_paths(args)

    if args.activation_range_regularization and not args.quant_aware_finetune:
        raise ValueError(
            "--activation-range-regularization requires --quant-aware-finetune "
            "so the stage4 activation clipping parameters exist."
        )
    if args.qat_train_activation_modules and not args.quant_aware_finetune:
        raise ValueError(
            "--qat-train-activation-modules requires --quant-aware-finetune."
        )
    if args.stage4_heads_only and args.stem_heads_only:
        raise ValueError("Choose only one of --stage4-heads-only or --stem-heads-only.")
    if args.stage4_heads_only and args.trainable_module_prefixes:
        raise ValueError("Choose either --stage4-heads-only or --trainable-module-prefixes, not both.")
    if args.stem_heads_only and args.trainable_module_prefixes:
        raise ValueError("Choose either --stem-heads-only or --trainable-module-prefixes, not both.")
    if args.stem_heads_only and not is_quant_native_follow_model_type(args.model_type):
        raise ValueError("--stem-heads-only is only supported for quant-native follow models.")

    if is_follow_model_type(args.model_type):
        print(f"{args.model_type} train annotations: {args.train_ann}")
        print(f"{args.model_type} val annotations: {args.val_ann}")

    train_ds, val_ds, collate_fn = build_datasets(args, repo_root)
    follow_training_policy = None
    follow_sampling_metadata = None
    hard_negative_image_scores: dict[int, float] | None = None
    if is_follow_model_type(args.model_type):
        follow_training_policy = build_follow_training_policy(args)
        hard_negative_image_scores = {}
        if follow_training_policy["sampler_enabled"]:
            print("Building follow sampling metadata from annotations...")
            follow_sampling_metadata = build_follow_sampling_metadata(train_ds)
            print("Follow sampling metadata:")
            print(
                "  visible_after_preprocess={visible_count}, not_visible={not_visible_count}, "
                "true_no_person={true_no_person_count}, crop_negative={crop_negative_count}".format(
                    **follow_sampling_metadata
                )
            )
        print("Follow training policy:")
        print(
            f"  base_loss_weights={follow_training_policy['base_loss_weights']}, "
            f"phase1_loss_weights={follow_training_policy['phase1_loss_weights']}"
        )
        print(
            "  visible_fraction_target={visible_fraction}, hard_negative_start_epoch={start}, "
            "hard_negative_boost={boost}, hard_negative_ema={ema}, phase1_epochs={phase1}".format(
                visible_fraction=follow_training_policy["visible_fraction"],
                start=follow_training_policy["hard_negative_start_epoch"],
                boost=follow_training_policy["hard_negative_boost"],
                ema=follow_training_policy["hard_negative_ema"],
                phase1=follow_training_policy["phase1_epochs"],
            )
        )

    base_train_loader = DataLoader(
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
    if args.init_ckpt:
        init_ckpt_path = (repo_root / args.init_ckpt).resolve()
        if not init_ckpt_path.is_file():
            raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt_path}")
        load_init_checkpoint(model, init_ckpt_path, device)

    model = enable_quant_aware_finetune(model, base_train_loader, device, args)
    if args.stage4_heads_only:
        apply_stage4_heads_only_freeze(model)
    if args.stem_heads_only:
        apply_stem_heads_only_freeze(model)
    if args.trainable_module_prefixes:
        apply_trainable_prefix_freeze(model, tuple(args.trainable_module_prefixes))
    if args.qat_train_activation_modules:
        apply_activation_module_freeze(model, tuple(args.qat_train_activation_modules))

    extra_train_loss_fn = None
    if args.activation_range_regularization:
        extra_train_loss_fn = lambda current_model: activation_range_regularization_loss(
            current_model,
            args.activation_range_reg_weight,
        )

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
        epoch_policy = {
            "phase_name": "single_phase",
            "loss_weights": None,
            "hard_negative_active": False,
        }
        train_loader = base_train_loader
        sampler_summary = None
        if follow_training_policy is not None:
            epoch_policy = resolve_follow_epoch_policy(follow_training_policy, epoch)
            print(
                "Follow epoch policy: "
                f"phase={epoch_policy['phase_name']}, "
                f"loss_weights={epoch_policy['loss_weights']}, "
                f"hard_negative_active={epoch_policy['hard_negative_active']}"
            )
            if follow_training_policy["sampler_enabled"] and follow_sampling_metadata is not None:
                epoch_sampler, sampler_summary = build_follow_epoch_sampler(
                    dataset=train_ds,
                    sampling_metadata=follow_sampling_metadata,
                    visible_fraction=follow_training_policy["visible_fraction"],
                    epoch=epoch,
                    hard_negative_start_epoch=follow_training_policy["hard_negative_start_epoch"],
                    hard_negative_boost=follow_training_policy["hard_negative_boost"],
                    hard_negative_image_scores=hard_negative_image_scores,
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=epoch_sampler,
                    num_workers=args.num_workers,
                    collate_fn=collate_fn,
                )
                print(
                    "Follow sampler: "
                    f"target_visible={sampler_summary['target_visible_fraction']:.3f}, "
                    f"expected_visible={sampler_summary['expected_visible_fraction']:.3f}, "
                    f"expected_true_no_person={sampler_summary['expected_true_no_person_fraction']:.3f}, "
                    f"boosted_true_no_person={sampler_summary['boosted_true_no_person_count']}"
                )
        train_stats = train_or_eval_epoch(
            model=model,
            loader=train_loader,
            device=device,
            model_type=args.model_type,
            follow_head_type=args.follow_head_type,
            optimizer=optimizer,
            extra_train_loss_fn=extra_train_loss_fn,
            max_batches=args.max_train_batches,
            loss_weights=epoch_policy["loss_weights"],
            hard_negative_image_scores=hard_negative_image_scores,
            hard_negative_boost=(
                follow_training_policy["hard_negative_boost"]
                if epoch_policy["hard_negative_active"] and follow_training_policy is not None
                else 0.0
            ),
            hard_negative_ema=(
                follow_training_policy["hard_negative_ema"]
                if follow_training_policy is not None
                else 0.70
            ),
        )
        train_stats["phase_name"] = epoch_policy["phase_name"]
        train_stats["active_loss_weights"] = epoch_policy["loss_weights"] or {}
        if sampler_summary is not None:
            train_stats["sampler"] = sampler_summary
        scheduler.step()

        val_stats = train_or_eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            model_type=args.model_type,
            follow_head_type=args.follow_head_type,
            optimizer=None,
            track_follow_metrics=is_follow_model_type(args.model_type),
            vis_thresh=args.vis_thresh,
            max_batches=args.max_val_batches,
            loss_weights=epoch_policy["loss_weights"],
            vis_threshold_sweep=args.vis_threshold_sweep if is_follow_model_type(args.model_type) else None,
        )
        if is_follow_model_type(args.model_type):
            val_stats["phase_name"] = epoch_policy["phase_name"]
            val_stats["active_loss_weights"] = epoch_policy["loss_weights"] or {}

        if is_follow_model_type(args.model_type):
            print(
                "Train loss: {loss:.4f} (x={x_offset:.4f}, size={size_proxy:.4f}, vis={visibility:.4f})".format(
                    **train_stats
                )
            )
            print(
                "Train gating pressure: "
                f"negative_visibility_confidence={train_stats['negative_visibility_confidence']:.4f}"
            )
            if "extra_train_loss" in train_stats:
                print(f"Train range regularization: {train_stats['extra_train_loss']:.4f}")
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
            if is_quant_native_follow_model_type(args.model_type):
                print(f"  x_exact_match_rate={val_stats['x_exact_match_rate']:.4f}")
                print(f"  x_adjacent_match_rate={val_stats['x_adjacent_match_rate']:.4f}")
                if val_stats.get("size_exact_match_rate", 0.0) > 0.0:
                    print(f"  size_exact_match_rate={val_stats['size_exact_match_rate']:.4f}")
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
            if "selected_vis_threshold" in val_stats:
                selected_metrics = val_stats["selected_vis_metrics"]
                print(
                    "Recommended visibility threshold: "
                    f"{val_stats['selected_vis_threshold']:.2f} "
                    f"(score={val_stats['selected_vis_threshold_score']:.4f}, "
                    f"precision={selected_metrics['precision']:.4f}, "
                    f"recall={selected_metrics['recall']:.4f}, "
                    f"f1={selected_metrics['f1']:.4f}, "
                    f"no_person_fp_rate={selected_metrics['no_person_fp_rate']:.4f})"
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

        if is_follow_model_type(args.model_type):
            # High visibility confidence does not guarantee the drone can steer well.
            # For person-follow control, horizontal centering error is the primary
            # selection target, with size acting only as a secondary tie-breaker.
            current_x_mae = val_stats["x_mae"]
            if current_x_mae < best_x_mae:
                best_x_mae = current_x_mae
                best_x_epoch = epoch
                best_x_path = output_dir / f"{args.model_type}_best_x.pth"
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
                best_follow_score_path = output_dir / f"{args.model_type}_best_follow_score.pth"
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

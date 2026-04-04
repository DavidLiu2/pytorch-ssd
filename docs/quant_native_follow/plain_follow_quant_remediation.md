# Plain Follow Quant Remediation

This note captures the current repo entry points for the three highest-value levers we added after the stem audit:

1. deployment-matched calibration selection
2. stem-specific PTQ calibration/range overrides
3. light QAT on only the stem and output head

## 1. Build A Calibration Manifest

Use [../../export/build_follow_calibration_manifest.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/build_follow_calibration_manifest.py) to rank real images by:

- hard negatives (`true_no_person`, `crop_negative`)
- decision-boundary proximity for `x`, `size`, and crop-driven visibility
- difficult lighting / contrast

Example:

```bash
python pytorch_ssd/export/build_follow_calibration_manifest.py \
  --image-dir pytorch_ssd/logs/hybrid_follow_val/1_real_image_validation/input_sets/representative16_20260324 \
  --annotations pytorch_ssd/data/coco/annotations/instances_val2017.json \
  --model-type plain_follow \
  --follow-head-type xbin9_size_bucket4 \
  --target-count 64 \
  --output pytorch_ssd/logs/plain_follow_quant_val/calibration_manifest.json \
  --overwrite
```

The JSON keeps the selected subset in order, and the sibling markdown file summarizes the selected tags.

## 2. Run PTQ With Stem-Specific Calibration Controls

[../../export/evaluate_quant_native_follow.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/evaluate_quant_native_follow.py) now accepts:

- `--calib-manifest`
- `--stem-activation-module`
- `--stem-activation-policy {none,percentile,mse}`
- `--stem-activation-percentile`

It also writes `calibration_summary.json` plus calibration sections inside `summary.json` / `summary.md`, including:

- calibration tag counts
- stem activation outlier diagnostics
- the selected stem activation override
- whether stem per-channel weight quantization is supported cleanly in the current NeMO path

Example:

```bash
python pytorch_ssd/export/evaluate_quant_native_follow.py \
  --ckpt pytorch_ssd/training/plain_follow/plain_follow_best_follow_score.pth \
  --output-dir pytorch_ssd/logs/plain_follow_quant_val/stem_calib_percentile \
  --calib-manifest pytorch_ssd/logs/plain_follow_quant_val/calibration_manifest.json \
  --stem-activation-policy percentile \
  --stem-activation-percentile 99.9 \
  --overwrite
```

## 3. Run Light QAT Only Where PTQ Fails

[../../train.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/train.py) now supports:

- `--stem-heads-only`
- `--trainable-module-prefixes`
- `--stem-channels`
- `--stem-mode {conv_bn_relu,delayed_relu}`
- `--stage-channels`

That means we can quant-aware fine-tune only the quant-sensitive front end plus the output head, without opening up the full network.

Example:

```bash
python pytorch_ssd/train.py \
  --model-type plain_follow \
  --follow-head-type xbin9_size_bucket4 \
  --stem-mode delayed_relu \
  --init-ckpt pytorch_ssd/training/plain_follow/plain_follow_best_follow_score.pth \
  --quant-aware-finetune \
  --activation-range-regularization \
  --stem-heads-only \
  --epochs 4 \
  --batch_size 8 \
  --lr 5e-4 \
  --qat-bits 8 \
  --qat-calib-batches 16 \
  --output_dir pytorch_ssd/training/plain_follow_stem_head_qat
```

For a custom freeze set, replace `--stem-heads-only` with `--trainable-module-prefixes stem.,output_head.` or another prefix list.

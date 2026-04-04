# dronet_lite_follow

`dronet_lite_follow` is the residual variant of the quant-native follow family. It keeps the same small grayscale `128x128` setup as `plain_follow`, but adds residual downsample blocks in the early stages to recover some capacity without returning to the heavier `hybrid_follow` pipeline.

## Why It Exists

The goal of `dronet_lite_follow` is to sit between the two extremes:

- more expressive than `plain_follow`,
- much easier to export and debug than `hybrid_follow`.

In practice, that goal is mostly true for the pipeline: the export flow is far cleaner than the hybrid path. The current caveat is that the latest checkpoint still shows real pre-runtime quant drift.

## Architecture

Implementation: [../../models/quant_native_follow_net.py](../../models/quant_native_follow_net.py)

Default structure:

| Block | Operation | Output shape from `1x128x128` input |
| --- | --- | --- |
| `stem` | `ConvBNReLU(1 -> 16, stride=2)` | `16x64x64` |
| `stage1` | `ResidualDownsampleStage(16 -> 24)` + refine | `24x32x32` |
| `stage2` | `ResidualDownsampleStage(24 -> 32)` + refine | `32x16x16` |
| `stage3` | `StraightStage(32 -> 48)` | `48x8x8` |
| `global_pool` | `AvgPool2d(8)` | `48x1x1` |
| `output_head` | `Linear(48 -> head_dim)` | `head_dim` |

Each `ResidualDownsampleStage` contains:

- a stride-2 main branch `ConvBNReLU`,
- a stride-1 main branch `ConvBN`,
- a stride-2 `1x1` projection on the skip branch,
- an add,
- a final ReLU.

Stages 1 and 2 each add an extra stride-1 `ConvBNReLU` refine block after the residual add.

This is still much simpler than the `hybrid_follow` late-stage residual stack, which is why the export/debug story is better.

## Head Contracts

Shared head definitions live in [../../utils/follow_task.py](../../utils/follow_task.py).

`dronet_lite_follow` supports the same three quant-native heads:

- `xbin9_size_scalar`
- `xbin9_size_bucket4`
- `lcr3_residual_size_scalar`

The currently validated quant/export checkpoint uses:

- checkpoint: [../../training/dronet_lite_follow/dronet_lite_follow_best_x.pth](../../training/dronet_lite_follow/dronet_lite_follow_best_x.pth)
- head type: `xbin9_size_scalar`

That head predicts:

- a 9-bin x offset class,
- a visibility logit,
- a scalar size logit.

## Training Defaults

Training entrypoint: [../../train.py](../../train.py)

Model-specific policy from `build_follow_training_policy()`:

- balanced follow sampler enabled
- default visible fraction target: `0.50`
- default two-phase schedule is enabled
- phase 1 length defaults to `max(4, epochs // 5)` unless overridden by `--phase1-epochs`
- `--disable-dronet-two-phase` collapses training back to a single phase
- hard-negative mining starts at `max(phase1_epochs + 1, 4)`
- hard-negative boost: `2.0`
- hard-negative EMA: `0.75`

Base loss weights:

- visibility: `2.5`
- x: `1.0`
- size: `0.3`
- residual: `0.5`

Phase-1 visibility-focused loss weights:

- visibility: `3.5`
- x: `0.75`
- size: `0.2`
- residual: `0.25`

Augmentation and preprocessing from [../../utils/transforms.py](../../utils/transforms.py):

- train: center-crop square -> resize -> random horizontal flip `p=0.25` -> grayscale tensor
- val: center-crop square -> resize -> grayscale tensor

The reduced flip probability is intentional so the visibility gate settles before the residual path starts chasing larger x offsets.

## Current Validation And Export Snapshot

Primary reports:

- float + quant summary: [../../logs/dronet_lite_follow_val/summary.md](../../logs/dronet_lite_follow_val/summary.md)
- machine-readable summary: [../../logs/dronet_lite_follow_val/summary.json](../../logs/dronet_lite_follow_val/summary.json)
- pre/post-quant rep16 overlays: [../../logs/dronet_lite_follow_val/comparison_summary.md](../../logs/dronet_lite_follow_val/comparison_summary.md)
- pre/post contact sheet: [../../logs/dronet_lite_follow_val/comparison_contact_sheet.png](../../logs/dronet_lite_follow_val/comparison_contact_sheet.png)

Export artifacts:

- ONNX and DORY package: [../../export/dronet_lite_follow](../../export/dronet_lite_follow)
- generated app: [../../export/dronet_lite_follow/application](../../export/dronet_lite_follow/application)

Float validation metrics from [../../logs/dronet_lite_follow_val/summary.md](../../logs/dronet_lite_follow_val/summary.md):

- `follow_score = 0.3583`
- `x_mae = 0.3053`
- `size_mae = 0.1766`
- `no_person_fp_rate = 0.1589`

Pre/post rep16 comparison from [../../logs/dronet_lite_follow_val/comparison_summary.md](../../logs/dronet_lite_follow_val/comparison_summary.md):

- pre-quant `follow_score = 0.1054`
- post-quant `follow_score = 0.6133`
- `visibility_gate_agreement = 0.5`
- `x_bin_exact_match_rate = 0.25`
- post-quant `no_person_fp_rate = 1.0`

Pipeline-complexity readout from [../../logs/dronet_lite_follow_val/summary.md](../../logs/dronet_lite_follow_val/summary.md):

- custom export patches required: `0`
- graph repair needed: `False`
- head collapse needed: `False`
- residual rescue needed: `False`

Quant-fidelity readout:

- earliest bad boundary: `fp_to_fq`
- first bad local op: `stem`
- `id_to_onnx` is effectively clean relative to the earlier float-to-quant stages

Interpretation:

- the deployment path is much more straightforward than `hybrid_follow`,
- the current corruption is not mainly a DORY/runtime plumbing issue,
- the main remaining problem is earlier quant sensitivity inside the model/export quantization path itself.

## Entry Points

Useful scripts for this model:

- export wrapper: [../../run_all.sh](../../run_all.sh)
- validation wrapper: [../../run_val.sh](../../run_val.sh)
- quant-native evaluator: [../../export/evaluate_quant_native_follow.py](../../export/evaluate_quant_native_follow.py)
- rep16 pre/post overlays: [../../export/compare_quant_native_follow_rep16_overlays.py](../../export/compare_quant_native_follow_rep16_overlays.py)

Notes:

- `run_all.sh` still defaults to `MODEL_TYPE=hybrid_follow`, so `dronet_lite_follow` needs explicit env overrides.
- `run_val.sh compare` can be pointed at the quant-native evaluator with `COMPARE_EVAL_SCRIPT=export/evaluate_quant_native_follow.py`.

## When To Reach For It

Use `dronet_lite_follow` when:

- we want a small residual follow model without re-entering the full hybrid-follow export complexity,
- we want a graph that is still simple enough to localize quant drift by stage,
- we are prepared to spend effort on quant sensitivity mitigation rather than runtime patching.

Right now it is best viewed as the cleaner deployment experiment, not yet the cleanest deployable outcome.

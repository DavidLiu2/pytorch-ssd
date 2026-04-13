# plain_follow

`plain_follow` is the simplest quant-native follow model in the repo. It removes residual connections entirely and keeps the network to a short sequence of `Conv + BN + ReLU` blocks plus a linear head.

## Why It Exists

`plain_follow` is the low-complexity baseline for the new follow family:

- easy to train,
- easy to inspect,
- easy to export conceptually,
- useful as a control when a more expressive model starts drifting during quantization.

If we want to answer "is the problem the follow task itself, or the architecture/export path?", this is usually the first model to check.

## Architecture

Implementation: [../../models/quant_native_follow_net.py](../../models/quant_native_follow_net.py)

Default structure:

| Block | Operation | Output shape from `1x128x128` input |
| --- | --- | --- |
| `stem` | `ConvBNReLU(1 -> 16, stride=2)` | `16x64x64` |
| `stage1` | `StraightStage(16 -> 24)` | `24x32x32` |
| `stage2` | `StraightStage(24 -> 32)` | `32x16x16` |
| `stage3` | `StraightStage(32 -> 48)` | `48x8x8` |
| `global_pool` | `AvgPool2d(8)` | `48x1x1` |
| `output_head` | `Linear(48 -> head_dim)` | `head_dim` |

Experimental stem variant:

- `--stem-mode delayed_relu` changes the stem to `ConvBN(1 -> stem_channels, stride=2)` followed by `ConvBNReLU(stem_channels -> stem_channels, stride=1)`.
- This delays the first activation until after one extra convolution/BN step, which is useful when the first deployable activation is the earliest quant-sensitive break.

Each `StraightStage` is:

- one stride-2 `ConvBNReLU` downsample block,
- followed by one stride-1 `ConvBNReLU` refine block.

There are no skip paths and no residual adds.

## Head Contracts

Shared head definitions live in [../../utils/follow_task.py](../../utils/follow_task.py).

`plain_follow` supports:

- `xbin9_size_scalar`
- `xbin9_size_bucket4`
- `lcr3_residual_size_scalar`

The checked-in rep16 validation artifact currently uses:

- checkpoint: [../../training/plain_follow/plain_follow_best_follow_score.pth](../../training/plain_follow/plain_follow_best_follow_score.pth)
- head type: `xbin9_size_bucket4`

That head predicts:

- a 9-bin x offset class,
- a visibility logit,
- a 4-bin size bucket.

## Training Defaults

Training entrypoint: [../../train.py](../../train.py)

Model-specific policy from `build_follow_training_policy()`:

- single-phase training, with no dronet-style warmup
- balanced follow sampler enabled
- default visible fraction target: `0.60`
- hard-negative mining starts at `max(4, epochs // 5)`
- hard-negative boost: `1.5`
- hard-negative EMA: `0.70`

Base loss weights:

- visibility: `2.0`
- x: `1.0`
- size: `0.3`
- residual: `0.5`

Augmentation and preprocessing from [../../utils/transforms.py](../../utils/transforms.py):

- train: center-crop square -> resize -> random horizontal flip `p=0.5` -> grayscale tensor
- val: center-crop square -> resize -> grayscale tensor

If `--output_dir` is not provided, quant-native follow training defaults to `pytorch_ssd/training/<model_type>_<follow_head_type>`. The checked-in `pytorch_ssd/training/plain_follow` folder was produced with an explicit output path.

## Current Validation Snapshots

Current float-side rep16 artifacts:

- summary: [../../logs/plain_follow_val/summary.json](../../logs/plain_follow_val/summary.json)
- contact sheet: [../../logs/plain_follow_val/contact_sheet.png](../../logs/plain_follow_val/contact_sheet.png)
- per-image overlays: [../../logs/plain_follow_val](../../logs/plain_follow_val)

Standalone float-only metrics from [../../logs/plain_follow_val/summary.json](../../logs/plain_follow_val/summary.json):

- `visibility_bce = 0.4297`
- `follow_score = 0.1394`
- `x_mae = 0.1106`
- `size_mae = 0.0962`
- `accuracy = 0.8125`
- `precision = 0.8182`
- `recall = 0.9000`
- `f1 = 0.8571`
- `x_exact_match_rate = 0.90`
- `size_exact_match_rate = 0.70`
- `no_person_fp_rate = 0.3333`

For directly comparable pre/post-quant performance on the same rep16 harness, use:

- comparison summary: [../../logs/plain_follow_quant_val/stem_integerization_study/overlays/current/comparison_summary.json](../../logs/plain_follow_quant_val/stem_integerization_study/overlays/current/comparison_summary.json)
- quant summary: [../../logs/plain_follow_quant_val/no_fusion_compare/summary.md](../../logs/plain_follow_quant_val/no_fusion_compare/summary.md)

Pre-quant metrics from the comparison harness:

- `visibility_bce = 0.5773`
- `follow_score = 0.1449`
- `x_mae = 0.1106`
- `size_mae = 0.1145`
- `accuracy = 0.6250`
- `precision = 0.8333`
- `recall = 0.5000`
- `f1 = 0.6250`
- `x_exact_match_rate = 0.9000`
- `size_exact_match_rate = 0.6000`
- `no_person_fp_rate = 0.1667`

Post-quant ONNX metrics from the same comparison harness:

- `visibility_bce = 0.6492`
- `follow_score = 0.0721`
- `x_mae = 0.0439`
- `size_mae = 0.0939`
- `accuracy = 0.6250`
- `precision = 0.8333`
- `recall = 0.5000`
- `f1 = 0.6250`
- `x_exact_match_rate = 1.0000`
- `size_exact_match_rate = 0.8000`
- `no_person_fp_rate = 0.1667`

Pre-to-post preservation metrics from that same artifact:

- `visibility_gate_agreement = 1.0000`
- `x_bin_exact_match_rate = 0.9375`
- `size_bucket_exact_match_rate = 0.8750`
- `x_value_mae = 0.0417`
- `size_value_mae = 0.0313`

Interpretation:

- on this small rep16 slice, `plain_follow` remains a readable float baseline for x and size,
- the checked-in pre/post-quant comparison now shows that the deployed `xbin9_size_bucket4` head preserves the decoded control output much better than the raw logits themselves,
- the quantized model still shifts confidence and internal logit margins, but most decoded `x` and `size` bins survive the export path.

## Quantization Status

`plain_follow` is part of the quant-native family and uses the same shared tooling:

- production wrapper: [../../run_plain_follow.sh](../../run_plain_follow.sh)
- release driver: [../../export/run_plain_follow_release.py](../../export/run_plain_follow_release.py)
- evaluator: [../../export/evaluate_quant_native_follow.py](../../export/evaluate_quant_native_follow.py)
- DORY app-seed artifact generator: [../../export/generate_dory_io_artifacts.py](../../export/generate_dory_io_artifacts.py)
- float overlays: [../../export/validate_follow_rep16_overlays.py](../../export/validate_follow_rep16_overlays.py)
- pre/post overlays: [../../export/compare_quant_native_follow_rep16_overlays.py](../../export/compare_quant_native_follow_rep16_overlays.py)

The current production path is:

1. Build a deployment-matched calibration manifest from COCO val.
2. Materialize a local validation bundle containing `rep16`, `hard_case`, and extra ranked COCO val images.
3. Run float overlays on `rep16`, `hard_case`, and the expanded pack.
4. Run the canonical ONNX export/eval to produce `model_id.onnx`, clean `model_id_dory.onnx`, and emit a `dory_cleanup_report`.
5. Run the DORY-graph semantic simulator on `rep16`, `hard_case`, and the expanded pack using cleaned `model_id_dory.onnx` as the parsed source graph.
6. Select the deployment visibility threshold on those DORY-semantic predictions.
7. Run pre/post overlay comparisons on `rep16`, `hard_case`, and the expanded pack against those DORY-semantic predictions.
8. Generate image-specific `input.txt`, `output.txt`, and `out_layer*.txt` artifacts from cleaned `model_id_dory.onnx` for the selected GVSOC image, copy them beside the DORY ONNX, and then run `network_generate`.
9. Run the GAP8 app smoke in GVSOC against that seeded app build and record the final tensor plus the generated layer-checksum metadata.
10. Read the consolidated release summary under `logs/plain_follow_prod`.

So the safe summary today is:

- the model architecture is export-oriented,
- the float checkpoint is already strong enough to promote into a production-style validation flow,
- the supported path is now the dedicated plain_follow wrapper rather than the older one-off study scripts,
- the wrapper now clamps rounded out-of-range Conv/Gemm weights into signed int8 before DORY app generation, which fixes the earlier weight-wrap bug,
- the wrapper now validates deployment metrics with a DORY-graph semantic simulator rather than ONNXRuntime on cleaned `model_id_dory.onnx`,
- the GVSOC smoke now seeds DORY app generation with image-specific `input.txt`, `output.txt`, and `out_layer*.txt` files from cleaned `model_id_dory.onnx`, so generated apps carry nonzero layer checksums for the smoke image,
- the generated GAP8 app now also gets an automatic `int64` requant patch before GVSOC, which fixes the known app-side BN/requant overflow drift in the generated `pulp_nn_utils.c` helpers,
- GVSOC is still the final deployment gate because the generated app/runtime remains the true end target.

## App Drift Fix

The remaining app-level drift that showed up after the DORY-semantic validation fix turned out to be a generated GAP8 runtime issue, not a model-weight mismatch:

- generated weights and BN constants already matched the parsed DORY graph,
- the first bad layer was `BNReluConvolution1`,
- the bad outputs were explained exactly by `int32` overflow in the generated BN/requant helpers,
- patching those helpers to use `int64` intermediates before the requant right shift removed the reproduced smoke-case drift completely.

Relevant tooling:

- generated-app patcher: [../../tools/patch_gap8_bn_quant_int64.py](../../tools/patch_gap8_bn_quant_int64.py)
- GVSOC runner that auto-applies the patch: [../../tools/run_aideck_val_impl.sh](../../tools/run_aideck_val_impl.sh)
- layer-by-layer runtime comparer: [../../export/compare_gap8_layer_bytes.py](../../export/compare_gap8_layer_bytes.py)
- deployment semantic helper: [../../export/dory_semantic_follow_inference.py](../../export/dory_semantic_follow_inference.py)
- passing repro artifact: [../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/runtime_layer_compare/runtime_layer_compare.json](../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/runtime_layer_compare/runtime_layer_compare.json)

The DORY semantic helper now prefers image-path restaging over caller-provided staged tensors when the input bundle requests `runtime_preprocess_uint8`. That keeps deployment threshold sweeps and compare overlays aligned with the seeded app/runtime input contract instead of a slightly different evaluation-only staging path.

## Reproduce The Fix

From the repo root, the main production wrapper already applies the generated-app patch automatically:

```bash
./pytorch_ssd/run_plain_follow.sh \
  --output-dir pytorch_ssd/logs/plain_follow_prod \
  --overwrite
```

If you want a faster smoke run while keeping the same patched GVSOC path:

```bash
./pytorch_ssd/run_plain_follow.sh \
  --output-dir pytorch_ssd/logs/plain_follow_prod_smoke \
  --calib-target-count 4 \
  --calib-max-images 32 \
  --expanded-pack-extra-count 1 \
  --expanded-pack-max-images 32 \
  --overwrite
```

If you want to inspect a generated app directly, patch it manually and compare its runtime layers:

```bash
./nemoenv/bin/python pytorch_ssd/tools/patch_gap8_bn_quant_int64.py \
  --app-dir /abs/path/to/generated/app
```

```bash
./nemoenv/bin/python pytorch_ssd/export/compare_gap8_layer_bytes.py \
  --gvsoc-log /abs/path/to/gvsoc_run.log \
  --layer-manifest /abs/path/to/gap8_layer_manifest.json \
  --output-dir /abs/path/to/runtime_layer_compare
```

The current known-good smoke repro is:

- GVSOC run log: [../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/gvsoc_run.log](../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/gvsoc_run.log)
- runtime layer compare: [../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/runtime_layer_compare/runtime_layer_compare.json](../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/runtime_layer_compare/runtime_layer_compare.json)
- final tensor: [../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/gvsoc_final_tensor.json](../../logs/plain_follow_prod_smoke_appseed/runtime_trace_rerun_int64_patch_regen/gvsoc_final_tensor.json)

## When To Reach For It

Use `plain_follow` when:

- we want the simplest new-model baseline,
- we want to isolate task/head behavior from residual-path behavior,
- we want a lower-risk starting point before spending time on quant drift mitigation in a more complex graph.

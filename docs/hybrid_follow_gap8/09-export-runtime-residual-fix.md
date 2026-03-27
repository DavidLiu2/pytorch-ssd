# Hybrid Follow Residual Export And Runtime Status

This note tracks the current `hybrid_follow` residual-state debugging as of March 26, 2026.

The short version is:

- the old GAP8 raw-residual runtime patches are still required and are present
- deploy-stage ONNX still matches the in-memory NEMO ID graph on the known sample
- the active issue is still a staged quantization-equivalence problem, not a generic ONNX export collapse
- the March 26 fix was upstream of `PACT_IntegerAdd`: deploy-time fused conv biases must be integerized with `eps_out_static` after `id_stage()`
- after that fix, the stage4.1 main branch no longer shows the earlier severe pre-add distortion on the known sample
- `PACT_IntegerAdd` still matters, but the best default policy changed after the bias fix: the representative 16-image batch now favors `legacy`

## Current Status

What is currently true on the known sample `data/coco/images/val2017/000000493613.jpg`:

- `deploy == ONNX` to numerical noise
- raw residual GAP8 patches are present in `application/`
- the first material semantic drift is still present before runtime
- the stage4.1 main branch is no longer the dominant upstream distortion source after the deploy conv-bias fix
- the largest tracked drift point on the current known-sample run is `stage4.1.add post-requant`
- changing the `PACT_IntegerAdd` scale-selection policy still moves the final outputs, but now it is tuning a smaller residual mismatch instead of compensating for a broken conv path

Current reference artifacts:

- [summary.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260326_conv_bias_fix/summary.md)
- [residual_upstream_report.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260326_conv_bias_fix/residual_upstream_report.md)
- [integer_add_policy_sweep.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260326_conv_bias_fix/integer_add_policy_sweep.md)
- [debug_report.json](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260326_conv_bias_fix/debug_report.json)
- [policy_batch_compare.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/policy_batch_compare_legacy_vs_fanin_20260326_conv_bias_fix/policy_batch_compare.md)

Current workflow update:

- `run_all.sh` now also runs `export/sweep_hybrid_follow_quant_drift.py` when the rep16 eval set is available
- that sweep identifies the earliest material FQ -> ID tap on `000000493613.jpg`, compares a small operator-specific policy set there, then scores the result on rep16
- each run writes `summary.{md,json}`, `local_operator_sweep.{md,json}`, and `batch_score_compare.{md,json}` under `export/hybrid_follow/quant_operator_sweep/run_all/`
- the sweep ends with one explicit recommendation: keep baseline, patch one operator with one policy, or reject all tested policies
- `run_all.sh` now reapplies the raw-residual GAP8 runtime patch set from the repo-local template at `export/hybrid_follow/gap8_runtime_patch_template/` instead of depending on an already-patched generated app

## What Was Ruled Out

These were the key narrowing steps:

1. Export-side zero collapse was ruled out on the known sample because the exported ONNX matches the in-memory ID graph.
2. The old runtime-only residual bug was ruled out as the sole blocker because the raw residual GAP8 patch set is present.
3. The remaining warning is the residual-add scale selection and divisor choice in the NEMO QD/ID path, after correcting deploy conv bias scaling.

So the active problem is a staged quantization-equivalence issue, not a generic exporter failure.

## Exact NEMO Code Path

The audited code path is the installed `nemo.quant.pact` implementation:

- `PACT_IntegerAdd.get_output_eps()`
- `PACT_IntegerAdd.forward()`
- `pact_integer_requantize_add()`

Their behavior matters because:

- `get_output_eps()` selects the common residual output quantum
- `forward()` uses integer requantization once the graph is both deployment-ready and integerized
- `pact_integer_requantize_add()` computes rounded integer branch multipliers and then does `floor(sum(...) / D)`

The repo-local exporter monkeypatches only the `get_output_eps()` policy during export/debug runs. It does not change the downstream integer add formula.

## March 26 Upstream Fix

The key exporter fix landed in `export_nemo_quant.py`, not in ONNX export and not in the GAP8 generated app:

- after `model_q.id_stage()`, every fused deploy-time `PACT_Conv2d` / `PACT_Conv1d` bias is now integerized with `round(bias / eps_out_static)`
- this is applied by `integerize_deploy_conv_biases(model_q)` before probing the in-memory ID model or exporting ONNX
- the fix uses `eps_out_static`, not `eps_static`

Why this mattered:

- NeMO integerizes fused conv weights, but its deploy path was still carrying fused conv bias in semantic float units
- our earlier branch diagnostics were also reading the wrong conv semantic output scale in a few places because they preferred `eps_static` over `eps_out_static`
- once both were corrected, the large stage4.1 pre-add distortion mostly disappeared on the known sample

Known-sample effect of the deploy conv-bias fix:

- `stage4.1.conv1 output` FQ -> ID mean abs diff: `0.266821 -> 0.018721`
- `stage4.1 activation between conv1 and conv2` FQ -> ID mean abs diff: `0.091894 -> 0.005373`
- `stage4.1.conv2 output` FQ -> ID mean abs diff: `0.318061 -> 0.014137`
- final decoded drift improved to `x=0.023793`, `size=0.295018`, `vis_conf=0.090408` on the current active run

## Corrected Baseline Semantics

One important doc correction:

- the true NEMO baseline is `eps_out = max(eps_in_list)`
- the earlier `max * fan_in` interpretation was wrong and should be treated as a discarded intermediate hypothesis

In repo-local policy names:

- `legacy` means the original NEMO rule, `max(eps_in_list)`
- `sqrt_fanin` means `max(eps_in_list) * sqrt(branch_count)`
- `midpoint` means halfway between `legacy` and `sqrt_fanin`
- `fanin` means `max(eps_in_list) * branch_count`

## Known-Sample Policy Comparison After The Bias Fix

The exporter writes a focused policy sweep report for the known sample.

Selection metric:

- `score_final_output = x_abs_diff + size_abs_diff + vis_conf_abs_diff`

Measured result on `000000493613.jpg` after the deploy conv-bias fix:

| Policy | `stage4.1.add eps_out` | `D` | Final score | Largest tracked drift |
| --- | --- | --- | --- | --- |
| `legacy` | `0.0091613736` | `32768` | `0.406244` | `stage4.1.add post-requant = 0.018155` |
| `sqrt_fanin` | `0.0129561387` | `32768` | `0.416341` | `stage4.1.add post-requant = 0.021956` |
| `midpoint` | `0.0110587571` | `32768` | `0.404355` | `stage4.1.add post-requant = 0.015361` |
| `fanin` | `0.0183227472` | `65536` | `0.409219` | `stage4.1.add post-requant = 0.023875` |

Current interpretation:

- the known sample alone prefers `midpoint`
- the default export policy is now `legacy`

Why the default changed anyway:

- after the conv-bias fix, the representative 16-image batch favors `legacy` over `fanin`
- we intentionally kept the wider comparison narrow instead of opening a broader policy search again
- `midpoint` remains available for single-sample debugging, but it is not the default deployment policy

## Representative 16-Image Batch After The Bias Fix

Batch comparison artifacts:

- [policy_batch_compare.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/policy_batch_compare_legacy_vs_fanin_20260326_conv_bias_fix/policy_batch_compare.md)
- [policy_batch_compare.json](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/policy_batch_compare_legacy_vs_fanin_20260326_conv_bias_fix/policy_batch_compare.json)

Aggregate result on the representative 16-image set:

- `legacy`: mean final score `0.561764`
- `fanin`: mean final score `0.569508`
- `legacy` beat `fanin` on `12 / 16` images
- mean score delta `fanin - legacy = +0.007744`

That is why `legacy` is now the default policy in `export_nemo_quant.py`.

## Focused Residual Numbers

On the current known-sample run with the conv-bias fix:

- `stage4.1.conv1 input`: `0.011530`
- `stage4.1.conv1 output`: `0.018721`
- `stage4.1 activation between conv1 and conv2`: `0.005373`
- `stage4.1.conv2 input`: `0.005373`
- `stage4.1.conv2 output`: `0.014137`
- `stage4.1 residual skip input`: `0.011530`

On the same run with the active `fanin` artifact:

- `stage4.1.add pre-requant`: `0.021403`
- `stage4.1.add post-requant`: `0.023875`
- `global pool output`: `0.017574`
- `head input`: `0.017574`

This is the key interpretation:

- before the conv-bias fix, the main residual branch was already badly distorted before the add
- after the fix, the main branch and skip branch are much closer through `conv1`, activation, and `conv2`
- the residual add is once again the dominant local drift point on the known sample
- the remaining mismatch is smaller, but it is still real enough to affect final decoded outputs

## Runtime Raw Residual State

The runtime-side raw residual fixes are still part of the valid deployment path.

That includes:

- raw helper prototypes in [pulp_nn_kernels.h](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/inc/pulp_nn_kernels.h)
- raw helper implementations in [pulp_nn_add.c](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_add.c)
- int32 raw-output behavior in [pulp_nn_conv_Ho_parallel.c](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_conv_Ho_parallel.c)
- int32 accumulation in [pulp_nn_matmul.c](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_matmul.c)
- int32 bias handling in [pulp_nn_linear_out_32.c](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_linear_out_32.c)
- raw residual wrapper repair via [reapply_gap8_raw_residual_patches.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/tools/reapply_gap8_raw_residual_patches.py)

Those fixes remain necessary, but they are no longer the main active suspect for the current known-sample drift.

## Repro Commands

Known-sample focused debug:

```powershell
python .\export_nemo_quant.py `
  --model-type hybrid_follow `
  --ckpt .\training\hybrid_follow\hybrid_follow_best_follow_score.pth `
  --out .\export\hybrid_follow\debug_export_quant_collapse_493613_20260326_conv_bias_fix\hybrid_follow_debug.onnx `
  --stage id `
  --bits 8 `
  --eps-in 0.00392156862745098 `
  --height 128 --width 128 --input-channels 1 `
  --calib-dir .\export\hybrid_follow\debug_export_quant_collapse_493613_20260325\input `
  --calib-batches 1 `
  --debug-quant-drift-dir .\export\hybrid_follow\debug_export_quant_collapse_493613_20260326_conv_bias_fix
```

This writes:

- `summary.md`
- `residual_upstream_report.json/.md`
- `residual_focus_report.json/.md`
- `integer_add_audit.json/.md`
- `integer_add_policy_sweep.json/.md`
- `debug_report.json`

Representative 16-image `legacy` vs `fanin` comparison after the bias fix was run in-process with the same repo-local exporter logic and calibration set. The output is in:

- `export/hybrid_follow/policy_batch_compare_legacy_vs_fanin_20260326_conv_bias_fix/`

## What To Do Next

If the known sample still fails after runtime checks and deploy matches ONNX:

1. Re-run the focused drift report on `000000493613.jpg`.
2. Read `export/hybrid_follow/quant_operator_sweep/run_all/summary.md` before changing deploy-time quant policy defaults.
3. Check whether deploy conv biases are integerized with `eps_out_static` before assuming the add policy is the root cause.
4. Read `residual_upstream_report.md` before changing `PACT_IntegerAdd` policy.
5. Use the representative batch comparison to decide the default policy, not the known sample alone.
6. Keep raw residual GAP8 patch verification in place, but do not assume it explains FQ -> ID drift.

This note replaces the older claim that the active blocker was a generic export-side collapse.

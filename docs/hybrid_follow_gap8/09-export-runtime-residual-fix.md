# Hybrid Follow Residual Export And Runtime Status

This note tracks the current `hybrid_follow` residual-state debugging as of March 25, 2026.

The short version is:

- the old GAP8 raw-residual runtime patches are still required and are present
- deploy-stage ONNX now matches the in-memory NEMO ID graph on the known sample
- the active issue is no longer a generic ONNX export collapse
- the current open drift appears during the FakeQuantized -> IntegerDeployable transition around residual adds, especially `stage4.1.add`
- the currently selected residual-add policy in `export_nemo_quant.py` is `fanin`

## Current Status

What is currently true on the known sample `data/coco/images/val2017/000000493613.jpg`:

- `deploy == ONNX` to numerical noise
- raw residual GAP8 patches are present in `application/`
- the first material semantic drift is already present before runtime
- the largest tracked drift point is still `stage4.1.add pre-requant`
- changing the `PACT_IntegerAdd` scale-selection policy improves the known sample slightly, but does not fully remove the FQ -> ID gap

Current reference artifacts:

- [summary.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260325_fanin_default/summary.md)
- [residual_focus_report.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260325_fanin_default/residual_focus_report.md)
- [integer_add_policy_sweep.md](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260325_fanin_default/integer_add_policy_sweep.md)
- [debug_report.json](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/debug_export_quant_collapse_493613_20260325_fanin_default/debug_report.json)

## What Was Ruled Out

These were the key narrowing steps:

1. Export-side zero collapse was ruled out on the known sample because the exported ONNX matches the in-memory ID graph.
2. The old runtime-only residual bug was ruled out as the sole blocker because the raw residual GAP8 patch set is present.
3. The remaining warning is the residual-add scale selection and divisor choice in the NEMO QD/ID path.

So the current issue is a staged quantization-equivalence problem, not a generic exporter failure.

## Exact NEMO Code Path

The audited code path is the installed `nemo.quant.pact` implementation:

- `PACT_IntegerAdd.get_output_eps()`
- `PACT_IntegerAdd.forward()`
- `pact_integer_requantize_add()`

Their behavior matters because:

- `get_output_eps()` selects the common residual output quantum
- `forward()` uses integer requantization once the graph is both deployment-ready and integerized
- `pact_integer_requantize_add()` computes rounded integer branch multipliers and then does `floor(sum(...) / D)`

The repo-local exporter now monkeypatches only the `get_output_eps()` policy during export/debug runs. It does not change the downstream integer add formula.

## Corrected Baseline Semantics

One important doc correction:

- the true NEMO baseline is `eps_out = max(eps_in_list)`
- the earlier `max * fan_in` interpretation was wrong and should be treated as a discarded intermediate hypothesis

In repo-local policy names:

- `legacy` means the original NEMO rule, `max(eps_in_list)`
- `sqrt_fanin` means `max(eps_in_list) * sqrt(branch_count)`
- `midpoint` means halfway between `legacy` and `sqrt_fanin`
- `fanin` means `max(eps_in_list) * branch_count`

## Known-Sample Policy Comparison

The exporter now writes a focused policy sweep report for the known sample.

Selection metric:

- `score_final_output = x_abs_diff + size_abs_diff + vis_conf_abs_diff`

Measured result on `000000493613.jpg`:

| Policy | `stage4.1.add eps_out` | `D` | Final score | Largest tracked drift |
| --- | --- | --- | --- | --- |
| `legacy` | `0.0091613736` | `32768` | `1.003436` | `stage4.1.add pre-requant = 0.391681` |
| `sqrt_fanin` | `0.0129561387` | `32768` | `1.009752` | `stage4.1.add pre-requant = 0.391817` |
| `midpoint` | `0.0110587571` | `32768` | `1.004311` | `stage4.1.add pre-requant = 0.391937` |
| `fanin` | `0.0183227472` | `65536` | `0.996848` | `stage4.1.add pre-requant = 0.391293` |

Current selected policy:

- `fanin`

Why it was kept:

- it gave the best final-output score on the known sample
- it also gave the smallest tracked drift at `stage4.1.add pre-requant` among the tested policies
- the improvement is modest, but real

## Focused Residual Numbers

With the current `fanin` policy:

- `stage4.0.add eps_in = [1.165186e-05, 3.091901e-05]`
- `stage4.0.add eps_out = 6.183803e-05`
- `stage4.0.add D = 256`
- `stage4.0.add mul = [48, 128]`

- `stage4.1.add eps_in = [1.380884e-05, 9.161374e-03]`
- `stage4.1.add eps_out = 1.832275e-02`
- `stage4.1.add D = 65536`
- `stage4.1.add mul = [49, 32768]`

Tracked FQ -> ID drift points on the same sample:

- `stage4.1.conv2 output`: `0.318061`
- `stage4.1 residual skip input`: `0.197471`
- `stage4.1.add pre-requant`: `0.391293`
- `stage4.1.add post-requant`: `0.390068`
- `global pool output`: `0.205429`
- `head input`: `0.205429`

Final decoded FQ -> ID drift with `fanin`:

- `x_abs_diff = 0.050862`
- `size_abs_diff = 0.398137`
- `vis_conf_abs_diff = 0.547849`

This is the key interpretation:

- the add requant step is not the only damage
- by the time we reach `stage4.1.add pre-requant`, the branches already disagree materially
- the residual-add policy still matters because it changes downstream eps propagation and final decoded outputs

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

## Repro Command

PowerShell:

```powershell
python .\export_nemo_quant.py `
  --model-type hybrid_follow `
  --ckpt .\training\hybrid_follow\hybrid_follow_best_follow_score.pth `
  --out .\export\hybrid_follow\debug_export_quant_collapse_493613_20260325_fanin_default\hybrid_follow_debug.onnx `
  --stage id `
  --bits 8 `
  --eps-in 0.00392156862745098 `
  --height 128 --width 128 --input-channels 1 `
  --calib-dir .\export\hybrid_follow\debug_export_quant_collapse_493613_20260325\input `
  --calib-batches 1 `
  --debug-quant-drift-dir .\export\hybrid_follow\debug_export_quant_collapse_493613_20260325_fanin_default
```

This writes:

- `summary.md`
- `residual_focus_report.json/.md`
- `integer_add_audit.json/.md`
- `integer_add_policy_sweep.json/.md`
- `debug_report.json`

## What To Do Next

If the known sample still fails after runtime checks and deploy matches ONNX:

1. Re-run the focused drift report on `000000493613.jpg`.
2. Read `integer_add_policy_sweep.md` before changing export logic.
3. Treat `stage4.1.add` as the first policy suspect, not ONNX export as a whole.
4. Keep raw residual GAP8 patch verification in place, but do not assume it explains FQ -> ID drift.

This note replaces the older claim that the active blocker was a generic export-side collapse.

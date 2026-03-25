# Hybrid Follow Export Fix And Raw Residual Reapply

Historical note as of March 24, 2026: the repo-local exporter was later restored to the older `906c1aa` baseline after retraining work showed the main regression source was not the export path itself. Use [10-baseline-restoration.md](10-baseline-restoration.md) for the current recommended baseline and verification commands.

## Scope

This note records the fix for the `hybrid_follow` NEMO -> ONNX -> DORY pipeline as of March 24, 2026.

The key outcome is:

- the export-side zero-collapse is fixed before runtime patching
- the generated GAP8 app now uses the raw residual add path again

## What Was Actually Broken

The main bug was not the old GAP8 residual runtime bug by itself.

The exported quantized ONNX was already wrong on the known failing sample:

- the final `stage4.1` residual add was nonzero before requantization
- the old exported graph then requantized through the wrong residual mapping and effectively wiped the activation path
- that produced all-zero pooled / head-input tensors downstream

## Findings Summary

The validated findings are:

- the root bug was export-side graph repair, not just GAP8 runtime residual handling
- the old broken export zeroed the final pooled/head path after the last `stage4.1` residual requant
- after the exporter fix, the known failing sample keeps nonzero activations through `stage4.1.add`, `global_pool`, and the head input
- fused export with the current collapsed head works after the fix
- fused export with the original three heads also works after the fix
- the unfused variant now also completes the quantized deploy transition and does not collapse on the debug sample
- weight clamping changes numerics, but it was not the root-cause fix for the collapse
- only after the export-side fix was validated was the raw residual GAP8 patch set reapplied

## Export-Side Fix

The export-side repair lives in [export_nemo_quant.py](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export_nemo_quant.py).

The important changes are:

- added a strict debug harness behind `--debug-quant-drift-dir`
- saved PyTorch and ONNX intermediate tensor dumps around `stage4.1`
- dumped module quant metadata before `qd_stage()`, after `qd_stage()`, and after `id_stage()`
- added strict JSON reporting and failure conditions for true zero-collapse
- kept the pipeline strict: no silent `qd_stage()` fallback
- restored the residual graph repair so `PACT_IntegerAdd` nodes map to the final ONNX residual add nodes rather than internal helper nodes
- made the residual graph repair work for both fused and unfused quantized graphs
- added a fallback for the older `nemoenv` / PyTorch 3.8 graph shape where projection and some residual nodes lose their PyTorch names and appear as anonymous `/Conv_*`, `/Add_*`, or `/Relu_*` nodes
- patched the deploy-graph rebuild path so the repaired mapping is not silently undone later in export
- added optional `--clamp-dory-weights` as a secondary experiment only

## Known Failing Sample Result

Measured result on COCO sample `000000493613.jpg`:

- `first_bad_location = null`
- `onnx.stage4_1_add_pre_requant.nonzero_count = 1280`
- `onnx.stage4_1_add_post_requant.nonzero_count = 1147`
- `onnx.global_pool_post_requant.nonzero_count = 55`
- `onnx.head_input.nonzero_count = 55`

That is the acceptance-critical change: the exported ONNX no longer collapses the pooled tensor or Gemm input to all zeros on the known failing sample.

The local debug export directories used during bring-up were temporary and have been cleaned up. The commands below regenerate the same style of report when needed.

## Variant Comparison

The debug harness compares three export variants with the same input:

- `variant_a_current_fused_single_head`: no collapse, semantic ONNX drift stayed small
- `variant_c_fused_three_heads`: no collapse, so head collapsing is not required for the fix
- `variant_b_unfused_single_head`: now also completes without collapse after the graph-repair fix

So the resolved bug is not tied to head collapsing, and the current fused and unfused paths are numerically healthy on the validated debug sample.

## Clamp Result

`--clamp-dory-weights` is now available in the exporter.

Observed behavior on the known sample:

- it changes ONNX numerics
- it does not explain or fix the original residual-collapse bug

Do not treat weight clamping as the root-cause fix unless a future drift report proves it.

## Runtime Raw Residual Reapply

After the export-side fix was confirmed, the generated app raw residual path was restored in `application/`.

`run_all.sh` now reapplies the raw-residual GAP8 patch set automatically after DORY code generation for `hybrid_follow`.

The runtime-side reapply includes:

- raw helper prototypes in [pulp_nn_kernels.h](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/inc/pulp_nn_kernels.h)
- raw helper implementations in [pulp_nn_add.c](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_add.c)
- restored int32-output behavior in [pulp_nn_conv_Ho_parallel.c](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_conv_Ho_parallel.c)
- restored int32 accumulation output path in [pulp_nn_matmul.c](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_matmul.c)
- int32 bias handling in [pulp_nn_linear_out_32.c](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/application/src/pulp_nn_linear_out_32.c)
- int32 bias casts in the raw residual producer wrappers
- direct raw residual add calls in `ReluQAddition4/7/11/14/18/21/25/28`

The key generated wrappers now route through:

- `pulp_nn_add_raw_i32_u8(...)`
- `pulp_nn_add_raw_i32_u8_mixed(...)`

instead of `pulp_nn_add(...)`.

To make that reproducible after regeneration, the repo includes:

- [reapply_gap8_raw_residual_patches.py](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/tools/reapply_gap8_raw_residual_patches.py)

That tool restores the tracked patched generated-runtime files from `HEAD`, rewrites the `ReluQAddition*.c` wrappers into the raw-helper form, and verifies the expected runtime fragments. `run_all.sh` now invokes it by default for `hybrid_follow`, and the latest verification report is:

- [gap8_raw_residual_patch_report.json](/c:/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/gap8_raw_residual_patch_report.json)

## Repro Commands

### 1. Debug export on the known failing sample

PowerShell:

```powershell
New-Item -ItemType Directory -Force .\tools\known_samples\hybrid_follow_493613 | Out-Null
Copy-Item .\data\coco\images\val2017\000000493613.jpg .\tools\known_samples\hybrid_follow_493613\000000493613.jpg -Force

python .\export_nemo_quant.py `
  --model-type hybrid_follow `
  --ckpt .\training\hybrid_follow\hybrid_follow_best_follow_score.pth `
  --out .\export\hybrid_follow\hybrid_follow_quant_debugfixed.onnx `
  --stage id `
  --bits 8 `
  --eps-in 0.00392156862745098 `
  --height 128 --width 128 --input-channels 1 `
  --calib-dir .\tools\known_samples\hybrid_follow_493613 `
  --calib-batches 1 `
  --force-cpu `
  --debug-quant-drift-dir .\export\hybrid_follow\debug_quant_drift_493613
```

### 2. Debug export on a small batch

PowerShell:

```powershell
python .\export_nemo_quant.py `
  --model-type hybrid_follow `
  --ckpt .\training\hybrid_follow\hybrid_follow_best_follow_score.pth `
  --out .\export\hybrid_follow\hybrid_follow_quant_debug_smallbatch.onnx `
  --stage id `
  --bits 8 `
  --eps-in 0.00392156862745098 `
  --height 128 --width 128 --input-channels 1 `
  --calib-dir .\data\coco\images\val2017 `
  --calib-batches 4 `
  --force-cpu `
  --debug-quant-drift-dir .\export\hybrid_follow\debug_quant_drift_smallbatch
```

### 3. Final export after the fix

PowerShell:

```powershell
python .\export_nemo_quant.py `
  --model-type hybrid_follow `
  --ckpt .\training\hybrid_follow\hybrid_follow_best_follow_score.pth `
  --out .\export\hybrid_follow\hybrid_follow_quant.onnx `
  --stage id `
  --bits 8 `
  --eps-in 0.00392156862745098 `
  --height 128 --width 128 --input-channels 1 `
  --calib-dir .\data\coco\images\val2017 `
  --calib-batches 128 `
  --force-cpu
```

### 4. DORY conversion after the fix

From PowerShell via WSL/bash:

```powershell
bash -lc "cd '/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd' && MODEL_TYPE=hybrid_follow CKPT='training/hybrid_follow/hybrid_follow_best_follow_score.pth' CALIB_DIR='data/coco/images/val2017' CALIB_BATCHES=8 OUT_ONNX='export/hybrid_follow/hybrid_follow_quant.onnx' DORY_ONNX='export/hybrid_follow/hybrid_follow_dory.onnx' RUN_STAGE_DRIFT=0 RUN_DORY=1 SYNC_TO_CRAZYFLIE=0 ./run_all.sh"
```

### 5. Reapply the raw residual GAP8 patch set after code generation

PowerShell:

```powershell
python .\tools\reapply_gap8_raw_residual_patches.py `
  --json-out .\export\hybrid_follow\gap8_raw_residual_patch_report.json
```

This is now optional for the default `hybrid_follow` flow because `run_all.sh` performs the same step automatically. It remains useful as a manual repair or verification command.

### 6. Verify the generated app still uses the raw residual path

PowerShell:

```powershell
python .\tools\reapply_gap8_raw_residual_patches.py `
  --check-only `
  --json-out .\export\hybrid_follow\gap8_raw_residual_patch_report.json
```

Healthy result:

- the JSON report says `"ok": true`
- raw helper declarations appear in `application/inc/pulp_nn_kernels.h`
- raw helper implementations appear in `application/src/pulp_nn_add.c`
- `ReluQAddition*.c` files call `pulp_nn_add_raw_i32_u8*`
- raw residual producer `Convolution*.c` files cast bias to `const int32_t *`

## Current Status

Export-side acceptance is met on the known failing sample.

The repo-local `run_all.sh` path now completes through DORY code generation with the fixed exporter.

The default reproducible flow is now:

1. run `run_all.sh`
2. optionally run `python .\tools\reapply_gap8_raw_residual_patches.py --check-only`

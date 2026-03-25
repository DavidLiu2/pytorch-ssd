# Hybrid Follow Baseline Restoration

## Scope

This note records the exporter rollback performed on March 24, 2026.

The goal was to restore the last known-good `hybrid_follow` export behavior that previously produced matching GVSOC outputs, then rerun the deployment pipeline before making any new export-side assumptions.

## Restored Baseline

The restored behavioral baseline is commit `906c1aa` (`generated app fixes and gvsoc validation; added docs`).

Files:

- `export_nemo_quant.py`: restored to the `906c1aa` functional path
- `models/hybrid_follow_net.py`: already matched `906c1aa`, so no functional edit was needed

Checkpoint / calibration / config used for verification:

- checkpoint: `training/hybrid_follow/hybrid_follow_best_follow_score.pth`
- calibration dir: `data/coco/images/val2017`
- calibration batches: `8`
- DORY config: `export/hybrid_follow/config_hybrid_follow_runtime.json`
- real-image validation set: `data/rep_images` with `--limit 16`

The only post-rollback code difference from the original `906c1aa` exporter is a parser-only `--calib-seed` argument so the current `run_all.sh` can still invoke the restored exporter without changing export behavior.

## Commands

### Rebuild the pipeline

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
CKPT=training/hybrid_follow/hybrid_follow_best_follow_score.pth \
CALIB_DIR=data/coco/images/val2017 \
CALIB_BATCHES=8 \
RUN_COMPAT_CHECKS=0 \
RUN_STAGE_DRIFT=0 \
SYNC_TO_CRAZYFLIE=0 \
./run_all.sh
```

`RUN_COMPAT_CHECKS=0` is required for this rollback because `export/check_model_compatibility.py` imports helper functions that only exist in the newer exporter rewrite.

### Staged GVSOC validation

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
PLATFORM=gvsoc ./run_aideck_val.sh
```

### Real-image validation rerun

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
STAGE_DRIFT=0 ./run_real_image_val.sh \
  --images-dir data/rep_images \
  --limit 16 \
  --results-dir export/hybrid_follow/real_image_validation/baseline_906c1aa_rep16 \
  --overwrite
```

## Verification Results

### `run_all.sh`

The rebuilt pipeline completed successfully on March 24, 2026 through:

- ID export
- ONNX simplification
- DORY frontend / backend generation
- raw-residual GAP8 patch reapply

### `run_aideck_val.sh`

The staged GVSOC smoke run passed on March 24, 2026.

Key result:

- `ReluConvolution0`: checksum OK
- final tensor: `6048 3223 181711`
- golden comparison: exact match

This is the first direct proof that the earlier `ReluConvolution0` divergence is gone under the restored exporter baseline.

### `run_real_image_val.sh`

The rerun over `data/rep_images` passed `16/16` on March 24, 2026.

Result directory:

- `export/hybrid_follow/real_image_validation/baseline_906c1aa_rep16`

Observed outcome:

- every sample matched its generated ONNX golden tensor exactly
- sample-specific `gvsoc.log` files still print layer-checksum failures starting at `ReluConvolution0`, because those embedded checksum tables are generated for the staged default `inputs.hex`, not for each overridden real-image input
- the acceptance signal for the real-image flow is therefore the exact final-tensor match recorded in `summary.json`, not the raw per-layer checksum printout inside each sample log

## Comparison With `crazyflie_ssd/generated`

The regenerated `pytorch_ssd/application` is still not byte-identical to `crazyflie_ssd/generated`.

That reference tree remains useful as a “worked before” example, but the differing hashes in generated C / network glue / first-layer weights strongly suggest that checkpoint or generated-artifact content had also changed since that older app was produced.

So the successful rollback result here should be interpreted as:

- the exporter baseline is healthy again
- exact identity with the archived `crazyflie_ssd/generated` tree is not the right acceptance criterion by itself

## Inferred Regression Source

There is no later committed exporter change after `906c1aa`; the regression came from the later uncommitted exporter rewrite.

Based on the rollback diff and the restored passing results, the smallest later behavioral change that appears to have introduced the deployment drift was:

- replacing the baseline best-effort `qd_stage()` fallback plus `id_stage(eps_in=dict)` flow with the newer hybrid-follow-specific graph-repair path that forces direct `qd_stage()` / `id_stage()` conversion

That inference is supported by two observations:

- the restored `906c1aa` exporter path passes both staged and real-image GVSOC validation again without any validation-script changes
- the later rewrite changed multiple diagnostics, graph-repair hooks, calibration handling, and ID requant logic, but the first export-side behavior shift large enough to change the deployed graph is the QD/ID transition rewrite itself

## Practical Guidance

Use the restored `906c1aa` exporter behavior as the deployment baseline first.

If new export-side work is needed later, reintroduce changes incrementally on top of this passing baseline and validate after each of these categories separately:

1. calibration preprocessing changes
2. QD / ID transition changes
3. hybrid-follow graph-repair changes
4. ONNX post-export weight / requant cleanup changes

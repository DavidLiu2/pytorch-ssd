# Using `run_all.sh`

`pytorch_ssd/run_all.sh` is the export and codegen pipeline for this project. For `hybrid_follow`, it drives the current repo-local deployment flow into `pytorch_ssd/application`.

## What It Does

For `MODEL_TYPE=hybrid_follow`, the script does all of the following:

1. Ensures the `doryenv` and `nemoenv` virtual environments exist.
2. Selects the best available `hybrid_follow` checkpoint for deployment.
3. Selects a representative calibration image directory, preferring COCO validation images over diagnostic `top_fn` or `top_fp` folders.
4. Runs a PyTorch compatibility preflight before export starts.
5. Exports the model through `export_nemo_quant.py`.
6. Simplifies and cleans the ONNX for DORY.
7. Runs an ONNX compatibility check on both the raw and DORY-clean graphs.
8. Generates DORY IO artifacts and weight text dumps.
9. Runs `network_generate.py`.
10. Writes the generated app into `pytorch_ssd/application`.
11. Runs the hybrid-follow quant drift sweep on the known bad sample plus the rep16 batch when the eval set is available.
12. Leaves the whole pipeline inside `pytorch_ssd` unless `SYNC_TO_CRAZYFLIE=1` is set manually.

## Default Hybrid-Follow Settings

Important defaults for the hybrid model:

- `MODEL_TYPE=hybrid_follow`
- checkpoint selection prefers `training/hybrid_follow/hybrid_follow_best_x.pth`, then `hybrid_follow_best_follow_score.pth`, then `hybrid_follow_best_total_loss.pth`
- final export stage: `id`
- strict stage conversion: enabled
- output app dir: `application`
- `RUN_DORY=1`
- `SYNC_TO_CRAZYFLIE=0`
- `RUN_QUANT_POLICY_SWEEP=1`
- quant sweep output dir: `export/hybrid_follow/quant_operator_sweep/run_all`
- quant sweep known bad image: `data/coco/images/val2017/000000493613.jpg`
- quant sweep eval set: `logs/hybrid_follow_val/1_real_image_validation/input_sets/representative16_20260324`

## Basic Usage

From WSL or any Linux-like shell:

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
./run_all.sh
```

From the repo root:

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS
bash pytorch_ssd/run_all.sh
```

## Common Overrides

Use environment variables when you want to override the defaults.

### Choose a specific checkpoint

```bash
CKPT=training/hybrid_follow/hybrid_follow_epoch_012.pth ./run_all.sh
```

### Choose a specific calibration directory

```bash
CALIB_DIR=training/hybrid_follow/eval_epoch_012 ./run_all.sh
```

### Generate the app somewhere else

```bash
DORY_APP_DIR=/tmp/hybrid_gap8_app SYNC_TO_CRAZYFLIE=0 ./run_all.sh
```

### Opt in to the old external sync

```bash
SYNC_TO_CRAZYFLIE=1 CRAZYFLIE_APP_DIR=../crazyflie_ssd ./run_all.sh
```

### Skip DORY generation and only do export

```bash
RUN_DORY=0 ./run_all.sh
```

### Skip the compatibility checker

```bash
RUN_COMPAT_CHECKS=0 ./run_all.sh
```

### Reproduce the current known-sample residual debug run

```bash
CKPT=training/hybrid_follow/hybrid_follow_best_follow_score.pth \
CALIB_DIR=export/hybrid_follow/debug_export_quant_collapse_493613_20260325/input \
CALIB_BATCHES=1 \
RUN_COMPAT_CHECKS=1 \
RUN_STAGE_DRIFT=1 \
STAGE_DRIFT_IMAGE=data/coco/images/val2017/000000493613.jpg \
STAGE_DRIFT_NEMO_STAGE=auto \
SYNC_TO_CRAZYFLIE=0 \
./run_all.sh
```

### Make the preflight dry-run cheaper

```bash
COMPAT_CALIB_BATCHES=4 ./run_all.sh
```

### Skip the quant drift sweep

```bash
RUN_QUANT_POLICY_SWEEP=0 ./run_all.sh
```

### Use the promoted `microblock_add_only` export preset

```bash
HYBRID_FOLLOW_EXPORT_PRESET=microblock_add_only \
RUN_QUANT_POLICY_SWEEP=0 \
./run_all.sh
```

That preset keeps `conv1` and `conv2` on the baseline policy and only patches `stage4.1.add`:

- add activation range: `percentile_99_0`
- add output scale rule: `mse_selected_joint`

### Point the quant drift sweep at a different eval batch

```bash
QUANT_POLICY_EVAL_DIR=logs/hybrid_follow_val/1_real_image_validation/input_sets/representative16_20260324 \
QUANT_POLICY_BATCH_LIMIT=8 \
./run_all.sh
```

## Important Outputs

The hybrid export pipeline writes the main artifacts here:

- ONNX export: `pytorch_ssd/export/hybrid_follow/hybrid_follow_quant.onnx`
- simplified ONNX: `pytorch_ssd/export/hybrid_follow/hybrid_follow_quant_sim.onnx`
- DORY-ready ONNX: `pytorch_ssd/export/hybrid_follow/hybrid_follow_dory.onnx`
- PyTorch compatibility report: `pytorch_ssd/export/hybrid_follow/model_compat_python.json`
- ONNX compatibility report: `pytorch_ssd/export/hybrid_follow/model_compat_onnx.json`
- generated DORY app: `pytorch_ssd/application`
- DORY weights text: `pytorch_ssd/export/hybrid_follow/weights_txt`
- DORY manifest: `pytorch_ssd/export/hybrid_follow/nemo_dory_artifacts.json`
- golden output tensor: `pytorch_ssd/export/hybrid_follow/output.txt`
- quant drift sweep summary: `pytorch_ssd/export/hybrid_follow/quant_operator_sweep/run_all/summary.md`
- quant drift sweep local report: `pytorch_ssd/export/hybrid_follow/quant_operator_sweep/run_all/local_operator_sweep.md`
- quant drift sweep batch report: `pytorch_ssd/export/hybrid_follow/quant_operator_sweep/run_all/batch_score_compare.md`

## How The Default Staged Sample Is Created

The current single-sample smoke test is still generated automatically during `run_all.sh`.

- `export/generate_dory_io_artifacts.py` writes `pytorch_ssd/export/hybrid_follow/input.txt`.
- That file currently contains a random uint8 input tensor shaped `1 x 1 x 128 x 128`.
- The same script runs ONNXRuntime on `hybrid_follow_dory.onnx` and writes the matching final tensor to `pytorch_ssd/export/hybrid_follow/output.txt`.
- When DORY runs `network_generate.py`, it converts `input.txt` into `pytorch_ssd/application/hex/inputs.hex`.

That staged sample is useful for a quick smoke test, but it is not a real image. The real-image path is documented in [06-real-image-validation.md](06-real-image-validation.md).

## How It Chooses The Model

For `hybrid_follow`, the script first tries to use:

- `training/hybrid_follow/hybrid_follow_best_x.pth`
- `training/hybrid_follow/hybrid_follow_best_follow_score.pth`
- `training/hybrid_follow/hybrid_follow_best_total_loss.pth`

If that file does not exist, it tries the latest `hybrid_follow_epoch_*.pth`.

If no checkpoint exists at all, it can create a bootstrap random-init checkpoint so the export path can still be smoke-tested.

## How It Chooses Calibration Data

If `CALIB_DIR` is not provided, the script scans:

- `pytorch_ssd/data/coco/images/val2017`
- `pytorch_ssd/data/coco/images/train2017`
- `pytorch_ssd/data/rep_images`

The automatic path intentionally avoids `training/hybrid_follow/eval_epoch_*` as a first choice because those folders often contain `top_fn` and `top_fp` diagnostic slices rather than representative deployment data.

If no representative calibration images exist, the export falls back to random calibration tensors.

## Hybrid-Follow Export Notes

The current repo-local export path is the one implemented in `export_nemo_quant.py` today.

Important consequences:

- hybrid-follow export stays on the strict direct `qd_stage(eps_in=1/255)` then `id_stage()` path
- the PyTorch and ONNX compatibility checks are part of the normal flow
- raw ONNX Conv/Gemm initializer range issues are treated as diagnostics, not silent export-time clipping
- `HYBRID_FOLLOW_EXPORT_PRESET=baseline` is still the default, and `microblock_add_only` is the current promoted preset for the stage4.1 add-only patch
- the exporter now includes residual-stage drift reporting and an integer-add policy sweep for the known sample
- `run_all.sh` now also runs the two-loop hybrid-follow quant drift sweep and writes `summary.{md,json}`, `local_operator_sweep.{md,json}`, and `batch_score_compare.{md,json}`
- the current default residual-add scale policy is documented in [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md)
- `REAPPLY_GAP8_RAW_RESIDUAL_PATCHES=1` now reapplies the verified GAP8 runtime patch set from `export/hybrid_follow/gap8_runtime_patch_template/`

## What Success Looks Like

A successful run ends with output like this conceptually:

- final quant stage is reported
- the PyTorch compatibility preflight status is printed
- the ONNX compatibility status is printed
- DORY config path is printed
- DORY application dir is printed
- DORY weight text dir is printed
- DORY artifact manifest is printed
- if sync is enabled, the `crazyflie_ssd` sync path is printed

The most important success condition for deployment is that these files exist afterward:

- `pytorch_ssd/application/src/network.c`
- `pytorch_ssd/application/inc/network.h`
- `pytorch_ssd/application/hex/inputs.hex`
- `pytorch_ssd/application/vars.mk`

## What `run_all.sh` Does Not Do

`run_all.sh` does not prove the generated app is numerically correct on GAP8 by itself.

After generation, the next step is:

```bash
./run_aideck_val.sh
```

That build-and-validate flow is documented in [05-run-aideck-val.md](05-run-aideck-val.md).

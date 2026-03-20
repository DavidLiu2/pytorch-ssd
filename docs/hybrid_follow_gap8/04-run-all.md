# Using `run_all.sh`

`pytorch_ssd/run_all.sh` is the export and codegen pipeline for this project. For `hybrid_follow`, it already defaults to the new deployment flow.

## What It Does

For `MODEL_TYPE=hybrid_follow`, the script does all of the following:

1. Ensures the `doryenv` and `nemoenv` virtual environments exist.
2. Selects the latest useful `hybrid_follow` checkpoint.
3. Selects a calibration image directory from `training/hybrid_follow/eval_epoch_*` when available.
4. Exports the model through `export_nemo_quant.py`.
5. Simplifies and cleans the ONNX for DORY.
6. Generates DORY IO artifacts and weight text dumps.
7. Runs `network_generate.py`.
8. Writes the generated app into `crazyflie_ssd/generated`.
9. Syncs `hex/` and `vars.mk` back into `crazyflie_ssd/`.

## Default Hybrid-Follow Settings

Important defaults for the hybrid model:

- `MODEL_TYPE=hybrid_follow`
- checkpoint default: `training/hybrid_follow/hybrid_follow_best_visibility.pth`
- final export stage: `id`
- output app dir: `../crazyflie_ssd/generated`
- `RUN_DORY=1`
- `SYNC_TO_CRAZYFLIE=1`

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

### Skip DORY generation and only do export

```bash
RUN_DORY=0 ./run_all.sh
```

## Important Outputs

The hybrid export pipeline writes the main artifacts here:

- ONNX export: `pytorch_ssd/export/hybrid_follow/hybrid_follow_quant.onnx`
- simplified ONNX: `pytorch_ssd/export/hybrid_follow/hybrid_follow_quant_sim.onnx`
- DORY-ready ONNX: `pytorch_ssd/export/hybrid_follow/hybrid_follow_dory.onnx`
- generated DORY app: `crazyflie_ssd/generated`
- DORY weights text: `pytorch_ssd/export/hybrid_follow/weights_txt`
- DORY manifest: `pytorch_ssd/export/hybrid_follow/nemo_dory_artifacts.json`
- golden output tensor: `pytorch_ssd/export/hybrid_follow/output.txt`

## How It Chooses The Model

For `hybrid_follow`, the script first tries to use:

- `training/hybrid_follow/hybrid_follow_best_visibility.pth`

If that file does not exist, it tries the latest `hybrid_follow_epoch_*.pth`.

If no checkpoint exists at all, it can create a bootstrap random-init checkpoint so the export path can still be smoke-tested.

## How It Chooses Calibration Data

If `CALIB_DIR` is not provided, the script scans:

- `training/hybrid_follow/eval_epoch_*`

and chooses the newest directory that exists. If no calibration images exist, the export falls back to random calibration tensors.

## What Success Looks Like

A successful run ends with output like this conceptually:

- final quant stage is reported
- DORY config path is printed
- DORY application dir is printed
- DORY weight text dir is printed
- DORY artifact manifest is printed
- if sync is enabled, the `crazyflie_ssd` sync path is printed

The most important success condition for deployment is that these files exist afterward:

- `crazyflie_ssd/generated/src/network.c`
- `crazyflie_ssd/generated/inc/network.h`
- `crazyflie_ssd/generated/hex/inputs.hex`
- `crazyflie_ssd/generated/vars.mk`

## What `run_all.sh` Does Not Do

`run_all.sh` does not prove the generated app is numerically correct on GAP8 by itself.

After generation, the next step is:

```bash
./run_aideck_val.sh
```

That build-and-validate flow is documented in [05-run-aideck-val.md](05-run-aideck-val.md).

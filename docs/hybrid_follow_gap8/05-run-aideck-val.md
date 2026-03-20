# Using `run_aideck_val.sh`

`pytorch_ssd/run_aideck_val.sh` is the build-and-validate step for the generated GAP8 app. It uses the AI-Deck Docker image and compares the final GVSOC tensor with the golden export.

## What It Does

The script performs these steps:

1. If started from MINGW, MSYS, or Cygwin, it re-enters through WSL automatically.
2. It treats `crazyflie_ssd/generated` as the primary generated app.
3. It checks whether that app is fresh relative to the latest export artifacts.
4. If needed, it can copy a fresh fallback app or rerun `run_all.sh`.
5. It stages the app into the AI-Deck example tree.
6. It overlays `aideck_val_main_hybrid.c`.
7. It builds the app inside the `bitcraze/aideck` container.
8. It runs GVSOC by default.
9. It compares the final tensor in the log against `pytorch_ssd/export/hybrid_follow/output.txt`.

## Freshness Logic

The primary source is:

- `crazyflie_ssd/generated`

The script considers the app fresh when it contains:

- `src/network.c`
- `inc/network.h`
- `hex/inputs.hex`
- `vars.mk`

and when `src/network.c` is newer than:

- `pytorch_ssd/export/hybrid_follow/hybrid_follow_nomin.onnx`
- `pytorch_ssd/export/hybrid_follow/nemo_dory_artifacts.json`

If that app is not fresh:

- it tries `pytorch_ssd/application` as a fallback source
- if that is also stale, it reruns `pytorch_ssd/run_all.sh`

## Basic Usage

From WSL:

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
./run_aideck_val.sh
```

From a Windows shell that launches Git Bash or MSYS Bash, the script can hand itself off to WSL automatically:

```bash
bash pytorch_ssd/run_aideck_val.sh
```

## Important Environment Variables

Common overrides:

- `HOST_APP_DIR`: generated app source, default `crazyflie_ssd/generated`
- `AUTO_REFRESH_APP`: set to `0` to forbid automatic refresh
- `REFRESH_SCRIPT`: script used when a refresh is needed, default `pytorch_ssd/run_all.sh`
- `PLATFORM`: `gvsoc` by default, can be changed to `board`
- `CONTAINER_NAME`: Docker container name, default `aideck`
- `AIDECK_IMAGE`: Docker image, default `bitcraze/aideck`
- `VERIFY_AFTER_RUN`: set to `0` to skip the final tensor comparison
- `DETACH_RUN`: set to `1` for a detached run

Example:

```bash
PLATFORM=gvsoc AUTO_REFRESH_APP=0 ./run_aideck_val.sh
```

## Logs And Outputs

Important output locations:

- host build dir: `aideck-gap8-examples/examples/other/dory_examples/application/BUILD/GAP8_V2/GCC_RISCV_PULPOS`
- host run log: `.../run_gvsoc.log`
- build log inside the app dir: `build_gvsoc.log`
- expected tensor: `pytorch_ssd/export/hybrid_follow/output.txt`

## What Success Looks Like

The successful path is:

- Docker container starts or is reused
- build completes without GAP SDK errors
- GVSOC run completes
- the compare script reports that the final tensor matches

For the current working setup, the validated final tensor is:

```text
901 11011 257657
```

## Typical Validation Command

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
./run_aideck_val.sh
```

## When To Rerun `run_all.sh`

Rerun `run_all.sh` when:

- the checkpoint changed
- calibration data changed
- export-side code changed
- DORY parser or backend kernel code changed
- `crazyflie_ssd/generated` is missing or stale

In most normal cases you do not need to copy files manually. `run_all.sh` already generates directly into `crazyflie_ssd/generated`, and `run_aideck_val.sh` already prefers that location.

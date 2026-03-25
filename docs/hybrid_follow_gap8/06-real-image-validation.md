# Real Image Validation

This is the end-to-end validation path for real images on the deployed `hybrid_follow` GAP8 app.

## What Changed

The pipeline now stays inside `pytorch_ssd` by default.

- `run_all.sh` generates the DORY app into `pytorch_ssd/application`.
- `run_aideck_val.sh` validates that local app by default.
- `run_real_image_val.sh` is now the canonical real-image entrypoint. It stages real images into `input.txt` plus `inputs.hex`, runs the existing AI-Deck Docker flow once per image, writes a batch summary, and can also regenerate overlays with `--overlay-only`.
- Each sample now also gets a `prediction_overlay.png` that shows the crop, predicted x-position, and largest COCO person box when available.
- When available, `run_real_image_val.sh` automatically uses `../doryenv/bin/python3`, so it can run from a clean WSL shell without a manual `source ../doryenv/bin/activate`.

The old single-sample smoke test still works. It still comes from the staged `export/hybrid_follow/input.txt` and `output.txt` pair that `run_all.sh` generates automatically.

## True Deployment Preprocessing

For `hybrid_follow`, the staged input matches the validation transform from `utils/transforms.py`:

1. center-crop the source image to a square
2. resize it to `128 x 128`
3. convert it to grayscale
4. convert it with `ToTensor()` and stage the resulting bytes as uint8

There is no extra per-image mean/std normalization in the deployed input artifact path.

## How To Run

From WSL:

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
source ../doryenv/bin/activate
./run_real_image_val.sh \
  --images-dir training/hybrid_follow/eval_epoch_015/top_fn \
  --limit 3
```

If the local app is stale, the first validation run will refresh it through `run_all.sh`.

## Where Results Go

By default the batch writes to:

- `pytorch_ssd/export/hybrid_follow/real_image_validation/<timestamp>/`

For each image it creates a folder like:

- `0001_<image-stem>/input.txt`
- `0001_<image-stem>/inputs.hex`
- `0001_<image-stem>/output.txt`
- `0001_<image-stem>/metadata.json`
- `0001_<image-stem>/run_aideck_val.log`
- `0001_<image-stem>/gvsoc.log`
- `0001_<image-stem>/gvsoc_final_tensor.json`
- `0001_<image-stem>/comparison.json`
- `0001_<image-stem>/prediction_overlay.png`

At the batch root it also writes:

- `summary.csv`
- `summary.json`

If you want to regenerate overlays for an existing results folder without rerunning GVSOC:

```bash
./run_real_image_val.sh --overlay-only \
  --results-dir export/hybrid_follow/real_image_validation/batch_smoke
```

That writes:

- `overlay_summary.csv`
- `overlay_summary.json`

`run_real_image_overlay.sh` still works as a compatibility wrapper, but it now delegates into `run_real_image_val.sh --overlay-only`.

## Notes

- `inputs.hex` is what GVSOC actually loads through readfs.
- `output.txt` is the Python-side golden tensor produced by ONNXRuntime on the same staged input.
- `HOST_INPUT_HEX` and `HOST_EXPECTED_OUTPUT` are the hooks that let `run_aideck_val.sh` validate each staged real image without regenerating the network.
- Use `--skip-overlays` if you want the old behavior without creating annotated images.

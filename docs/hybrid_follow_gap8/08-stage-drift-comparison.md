# Stage Drift Comparison

`hybrid_follow` stage drift means: run the same image through the same deployed preprocessing once, then compare the 3-output prediction vector across export and deployment checkpoints:

- PyTorch checkpoint
- optional in-memory NEMO quantized stage
- exported ONNX
- golden output artifact
- optional GVSOC final tensor

The goal is to answer a very specific question: where do `x_offset`, `size_proxy`, and `visibility` start changing enough that the model is no longer semantically the same?

## What The Tool Uses

The comparison script is [compare_hybrid_follow_stages.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/compare_hybrid_follow_stages.py).

It reuses the real deployed preprocessing:

- center-crop to square
- resize to `128x128`
- grayscale
- `ToTensor()`
- no extra normalization unless explicitly configured elsewhere

It keeps the outputs explicit:

- raw vector `[x_offset, size_proxy, visibility_logit]`
- decoded `visibility_confidence = sigmoid(visibility_logit)`

For fixed-point deployment outputs, it decodes with the current repository default scale `32768.0`.

## Manual Usage

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd

python export/compare_hybrid_follow_stages.py \
  --image training/hybrid_follow/eval_epoch_015/top_fn/01_p0.0114_000000132408.jpg \
  --ckpt training/hybrid_follow/hybrid_follow_best_follow_score.pth \
  --onnx export/hybrid_follow/hybrid_follow_dory.onnx \
  --output-dir export/hybrid_follow/stage_drift/manual_example \
  --overwrite \
  --nemo-stage auto
```

Optional runtime inputs:

```bash
python export/compare_hybrid_follow_stages.py \
  --image <image> \
  --ckpt <ckpt> \
  --onnx <onnx> \
  --golden <sample-dir/output.txt> \
  --gvsoc-json <sample-dir/gvsoc_final_tensor.json> \
  --output-dir <sample-dir/stage_drift> \
  --overwrite
```

## Pipeline Hooks

`run_all.sh` now runs a one-image stage-drift report by default for `hybrid_follow`.

Useful env vars:

- `RUN_STAGE_DRIFT=1`
- `STAGE_DRIFT_IMAGE=training/hybrid_follow/eval_epoch_015/top_fn/01_p0.0114_000000132408.jpg`
- `STAGE_DRIFT_OUTPUT_DIR=export/hybrid_follow/stage_drift/run_all`
- `STAGE_DRIFT_NEMO_STAGE=auto`

`run_aideck_val.sh` now accepts optional env hooks so a caller can attach stage drift to a runtime validation sample:

- `RUN_STAGE_DRIFT_DEBUG=1`
- `HOST_STAGE_DRIFT_IMAGE=<image>`
- `HOST_STAGE_DRIFT_OUTPUT_DIR=<sample-dir/stage_drift>`
- `HOST_FINAL_TENSOR_JSON=<sample-dir/gvsoc_final_tensor.json>`

`run_real_image_val.sh` now enables stage drift by default and forwards env-based overrides into the batch validator:

```bash
STAGE_DRIFT_NEMO_STAGE=skip ./run_real_image_val.sh ...
```

That writes a per-sample `stage_drift/` directory beside the normal validation artifacts.

To disable it for a batch run:

```bash
STAGE_DRIFT=0 ./run_real_image_val.sh ...
```

## Output Layout

The comparison output directory contains:

- `preprocessed_input_preview.png`
- `preprocessed_tensor_float.npy`
- `preprocessed_tensor_uint8.npy`
- `raw_outputs/*.json`
- `decoded_outputs/*.json`
- `pairwise_diff_report.json`
- `stage_drift_report.json`
- `summary.md`

## Interpreting Results

Healthy result:

- PyTorch and optional NEMO stay close
- ONNX stays close to the preceding stage
- ONNX and golden match
- golden and GVSOC match when runtime is available

Concerning result:

- the first large warning appears before runtime
- for example, PyTorch and NEMO are close, but ONNX collapses or shifts sharply

The default warnings are:

- `x_offset` abs diff `> 0.05`
- `size_proxy` abs diff `> 0.05`
- `visibility_confidence` abs diff `> 0.10`

When that happens, the summary calls out the first adjacent pair where those thresholds trip so it is obvious where semantic drift begins.

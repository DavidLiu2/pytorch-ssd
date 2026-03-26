# Hybrid Follow Stage Drift Summary

- Image: `/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/data/coco/images/val2017/000000493613.jpg`
- Output dir: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325`
- Preprocess: Center-crop to square, resize to 128x128, convert to grayscale, then torchvision ToTensor() with no extra normalization.
- Integer decode scale: `32768.0`
- Warning thresholds: `{"x_abs_diff": 0.05, "size_abs_diff": 0.05, "vis_conf_abs_diff": 0.1}`
- Golden artifact: generated for this run at `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/generated_golden_output.txt`

## Drift Onset

- Configured drift thresholds first trip between PyTorch checkpoint and Exported ONNX (ID).
- WARNING: x_offset abs diff 0.112076 exceeds 0.050000
- WARNING: size_proxy abs diff 0.166048 exceeds 0.050000
- WARNING: visibility_confidence abs diff 0.142454 exceeds 0.100000

## Stage Outputs

- PyTorch checkpoint: x=0.071823, size=0.401915, vis_logit=6.259085, vis_conf=0.998091
- nemo: ERROR (RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.)
- Exported ONNX (ID): x=0.183899, size=0.567963, vis_logit=1.779510, vis_conf=0.855636
- Golden output artifact: x=0.183899, size=0.567963, vis_logit=1.779510, vis_conf=0.855636
- GVSOC final tensor: SKIPPED

## Pairwise Diffs

- pytorch_vs_quantized: SKIPPED (One or both stages did not complete successfully.)
- quantized_vs_onnx: SKIPPED (One or both stages did not complete successfully.)
- pytorch_vs_onnx: status=WARN x_abs=0.112076 size_abs=0.166048 vis_conf_abs=0.142454
- WARNING: x_offset abs diff 0.112076 exceeds 0.050000
- WARNING: size_proxy abs diff 0.166048 exceeds 0.050000
- WARNING: visibility_confidence abs diff 0.142454 exceeds 0.100000
- onnx_vs_golden: status=OK x_abs=0.000000 size_abs=0.000000 vis_conf_abs=0.000000
- golden_vs_gvsoc: SKIPPED (One or both stages did not complete successfully.)
- pytorch_vs_gvsoc: SKIPPED (One or both stages did not complete successfully.)

## Artifacts

- Preview: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/preprocessed_input_preview.png`
- Float tensor dump: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/preprocessed_tensor_float.npy`
- Uint8 tensor dump: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/preprocessed_tensor_uint8.npy`
- Raw per-stage JSON: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/raw_outputs`
- Decoded per-stage JSON: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/decoded_outputs`
- Pairwise JSON: `export/hybrid_follow/stage_drift/debug_compare_current_dory_493613_auto_20260325/pairwise_diff_report.json`

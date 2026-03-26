# Hybrid Follow Stage Drift Summary

- Image: `/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/data/coco/images/val2017/000000493613.jpg`
- Output dir: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325`
- Preprocess: Center-crop to square, resize to 128x128, convert to grayscale, then torchvision ToTensor() with no extra normalization.
- Integer decode scale: `32768.0`
- Warning thresholds: `{"x_abs_diff": 0.05, "size_abs_diff": 0.05, "vis_conf_abs_diff": 0.1}`

## Drift Onset

- Configured drift thresholds first trip between PyTorch checkpoint and NEMO ID (in-memory).
- WARNING: x_offset abs diff 0.083176 exceeds 0.050000
- WARNING: size_proxy abs diff 0.215486 exceeds 0.050000
- WARNING: visibility_confidence abs diff 0.137081 exceeds 0.100000

## Stage Outputs

- PyTorch checkpoint: x=0.071823, size=0.401915, vis_logit=6.259085, vis_conf=0.998091
- NEMO ID (in-memory): x=0.154999, size=0.617401, vis_logit=1.823700, vis_conf=0.861009
- Exported ONNX: SKIPPED (ONNX file not found: export/hybrid_follow/debug_export_quant_collapse_493613_20260325_scale_patch/hybrid_follow_debug.onnx)
- Golden output artifact: SKIPPED
- GVSOC final tensor: SKIPPED

## Pairwise Diffs

- pytorch_vs_quantized: status=WARN x_abs=0.083176 size_abs=0.215486 vis_conf_abs=0.137081
- WARNING: x_offset abs diff 0.083176 exceeds 0.050000
- WARNING: size_proxy abs diff 0.215486 exceeds 0.050000
- WARNING: visibility_confidence abs diff 0.137081 exceeds 0.100000
- quantized_vs_onnx: SKIPPED (One or both stages did not complete successfully.)
- pytorch_vs_onnx: SKIPPED (One or both stages did not complete successfully.)
- onnx_vs_golden: SKIPPED (One or both stages did not complete successfully.)
- golden_vs_gvsoc: SKIPPED (One or both stages did not complete successfully.)
- pytorch_vs_gvsoc: SKIPPED (One or both stages did not complete successfully.)

## Artifacts

- Preview: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/preprocessed_input_preview.png`
- Float tensor dump: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/preprocessed_tensor_float.npy`
- Uint8 tensor dump: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/preprocessed_tensor_uint8.npy`
- Raw per-stage JSON: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/raw_outputs`
- Decoded per-stage JSON: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/decoded_outputs`
- Pairwise JSON: `export/hybrid_follow/stage_drift/debug_compare_scale_patch_493613_20260325/pairwise_diff_report.json`

# Quant-Native Follow Models

This folder summarizes the two newer follow-model families in this repo: `plain_follow` and `dronet_lite_follow`.

Both models share the same high-level goal:

- keep the follow task in a small grayscale `128x128` network,
- use export-friendly conv/BN/ReLU building blocks,
- avoid the late-stage complexity that made `hybrid_follow` harder to quantize and debug,
- support discrete follow heads that are easier to reason about than the old direct-regression contract.

## Document Map

- [plain_follow.md](plain_follow.md): the simplest quant-native baseline, with no residual adds.
- [dronet_lite_follow.md](dronet_lite_follow.md): a small residual variant that keeps the export graph much cleaner than `hybrid_follow`.
- [../hybrid_follow_gap8/README.md](../hybrid_follow_gap8/README.md): older hybrid GAP8 notes for comparison.

## Shared Assumptions

- Input is true 1-channel grayscale only.
- Input size is fixed at `128x128`.
- Preprocessing for train/val is center-crop square -> resize -> grayscale tensor.
- The implementation lives in [../../models/quant_native_follow_net.py](../../models/quant_native_follow_net.py).
- Checkpoint metadata and factory loading live in [../../models/follow_model_factory.py](../../models/follow_model_factory.py).
- Head contracts and decode/loss logic live in [../../utils/follow_task.py](../../utils/follow_task.py).

Supported quant-native follow heads:

- `xbin9_size_scalar`: 9-way x bin classification + visibility logit + scalar size logit.
- `xbin9_size_bucket4`: 9-way x bin classification + visibility logit + 4-way size bucket.
- `lcr3_residual_size_scalar`: 3-way left/center/right coarse x + residual + visibility + scalar size.

## At A Glance

| Model | Core idea | Graph style | Training policy | Current repo status |
| --- | --- | --- | --- | --- |
| `plain_follow` | Minimal straight-through baseline | Pure downsample + refine stages, no residual adds | Single-phase | Float rep16 validation and checked-in quant/export comparison artifacts are available |
| `dronet_lite_follow` | Small residual follow net inspired by DroNet-style downsample blocks | Residual downsample in stages 1-2, straight stage 3 | Two-phase by default | Full quant/export/app generation path is checked in, but current quant drift is still significant |

## Current Snapshot

As of `2026-04-10`, the repo state is:

- `plain_follow` currently looks like the cleaner functional baseline on the rep16 slice, and it now has the main production wrapper in place. That wrapper validates deployment behavior with a DORY-graph semantic simulator built from cleaned `model_id_dory.onnx`, emits a `dory_cleanup_report`, and uses GVSOC as the final app-level gate. See [plain_follow.md](plain_follow.md), [../../logs/plain_follow_val/summary.json](../../logs/plain_follow_val/summary.json), and [../../logs/plain_follow_quant_val/no_fusion_compare/summary.md](../../logs/plain_follow_quant_val/no_fusion_compare/summary.md).
- `dronet_lite_follow` is the cleaner deployment pipeline. The export path completed without the hybrid-style patch stack, and the main artifacts are in [../../export/dronet_lite_follow](../../export/dronet_lite_follow). See [../../logs/dronet_lite_follow_val/summary.md](../../logs/dronet_lite_follow_val/summary.md) and [../../logs/dronet_lite_follow_val/comparison_summary.md](../../logs/dronet_lite_follow_val/comparison_summary.md).
- The important nuance is that `dronet_lite_follow` is simpler to debug than `hybrid_follow`, but the current checkpoint still shows meaningful semantic drift after quantization. Its earliest bad boundary is currently `fp_to_fq`.
- The earlier `plain_follow` app-level drift root cause is now understood: generated GAP8 BN/requant helpers were overflowing `int32` intermediates. The pipeline now patches those generated helpers to use `int64` intermediates before GVSOC, and the known smoke repro now matches the seeded golden tensors exactly layer by layer.
- One remaining caveat for `plain_follow`: the DORY-graph simulator is much closer to deploy semantics than ONNXRuntime on cleaned `model_id_dory.onnx`, but GVSOC is still the final gate because the deployed app is the real compiler/runtime path.
- The GVSOC smoke path now seeds `network_generate` with image-specific `input.txt`, `output.txt`, and `out_layer*.txt` files generated from cleaned `model_id_dory.onnx`, so the app build carries meaningful per-layer checksums for the selected smoke image.

## Entry Points

- Training: [../../train.py](../../train.py)
- Production wrapper: [../../run_plain_follow.sh](../../run_plain_follow.sh)
- Plain-follow release driver: [../../export/run_plain_follow_release.py](../../export/run_plain_follow_release.py)
- Quant/export validation: [../../export/evaluate_quant_native_follow.py](../../export/evaluate_quant_native_follow.py)
- DORY-semantic deployment simulator: [../../export/dory_semantic_follow_inference.py](../../export/dory_semantic_follow_inference.py)
- DORY app-seed artifact generator: [../../export/generate_dory_io_artifacts.py](../../export/generate_dory_io_artifacts.py)
- GAP8 requant patcher: [../../tools/patch_gap8_bn_quant_int64.py](../../tools/patch_gap8_bn_quant_int64.py)
- GAP8 layer comparer: [../../export/compare_gap8_layer_bytes.py](../../export/compare_gap8_layer_bytes.py)
- Calibration manifest builder: [../../export/build_follow_calibration_manifest.py](../../export/build_follow_calibration_manifest.py)
- Float rep16 overlays: [../../export/validate_follow_rep16_overlays.py](../../export/validate_follow_rep16_overlays.py)
- Pre/post-quant rep16 overlays: [../../export/compare_quant_native_follow_rep16_overlays.py](../../export/compare_quant_native_follow_rep16_overlays.py)
- Legacy/general wrappers: [../../run_all.sh](../../run_all.sh) and [../../run_val.sh](../../run_val.sh)

Historical one-off studies were moved out of the main path into [../../export/archive/README.md](../../export/archive/README.md).

Use `plain_follow` when we want the simplest quant-native baseline to compare against. Use `dronet_lite_follow` when we want a residual model that still stays inside a much more manageable export/debug path than `hybrid_follow`.

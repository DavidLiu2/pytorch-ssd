# Hybrid Follow GAP8 Notes

This folder explains how the working `hybrid_follow` GAP8 deployment differs from the reference Frontnet runtime in `nanocockpit`, which patches were required to make it work, and how to regenerate and validate the app.

## Document Map

- [01-network-comparison.md](01-network-comparison.md): describe the `hybrid_follow` application layout and the runtime API structure.
- [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md): the complete patch set that made the generated network work.
- [03-residual-branch-debugging.md](03-residual-branch-debugging.md): the residual-branch dtype investigation and the final root cause.
- [04-run-all.md](04-run-all.md): how to export, quantize, generate, and stage the app with `run_all.sh`.
- [05-run-aideck-val.md](05-run-aideck-val.md): how to build and validate the generated app with the AI-Deck Docker flow.
- [06-real-image-validation.md](06-real-image-validation.md): how to run real-image validation batches and where the per-image artifacts land.
- [07-model-compatibility-checks.md](07-model-compatibility-checks.md): summarize the model-architecture findings and explain the PyTorch/ONNX compatibility checker.
- [08-stage-drift-comparison.md](08-stage-drift-comparison.md): explain how to compare PyTorch, NEMO, ONNX, golden, and GVSOC outputs for semantic drift.
- [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md): current residual export/runtime status, including the conv-bias fix and the latest `PACT_IntegerAdd` policy comparisons.

## Current Validated State

- `pytorch_ssd/run_all.sh` defaults to `MODEL_TYPE=hybrid_follow`.
- The generated DORY app is written to `pytorch_ssd/application`.
- `run_all.sh` no longer touches `crazyflie_ssd` unless `SYNC_TO_CRAZYFLIE=1` is set explicitly.
- `pytorch_ssd/run_val.sh aideck` uses `pytorch_ssd/application` as the primary app source.
- Real-image batch validation writes outputs under `pytorch_ssd/export/hybrid_follow/real_image_validation/`.
- The generated app is back on the raw residual add path instead of the plain `uint8_t` residual helper path.
- `run_all.sh` can now promote the current exporter winner with `HYBRID_FOLLOW_EXPORT_PRESET=microblock_add_only`.
- The raw-residual GAP8 patch set is reapplied from `pytorch_ssd/export/hybrid_follow/gap8_runtime_patch_template/` during fresh hybrid-follow exports.
- `nemo/export_nemo_quant.py` is now a side-effect-free compatibility facade; scoped integer-add policy changes live in `nemo/export_nemo_quant_scopes.py`.
- The exporter now includes focused FQ -> ID residual drift reporting, stage4.1 upstream conv instrumentation, and an integer-add policy sweep on the known sample.
- `run_all.sh` now also runs a two-loop quant drift sweep on `000000493613.jpg` plus the rep16 batch when those inputs are available.
- Stage-drift reports now collect explicit `FP`, `FQ`, `ID`, `ONNX`, `golden`, and `GVSOC` stages instead of folding everything into one generic quantized stage.
- `run_aideck_val` / real-image validation can optionally trace per-layer GVSOC output bytes, and `export/build_gap8_layer_manifest.py` joins those runtime dumps with the DORY golden layer order.
- `export/check_export_nemo_quant_determinism.py` checks import safety, scoped patch restoration, repeated export determinism, and debug-path no-op behavior for baseline and preset exports.
- Deploy-time fused conv biases are integerized with `eps_out_static` after `id_stage()`.
- The active default integer-add scale policy is `legacy`.
- The current open issue is residual-stage quantization drift before runtime, not a generic ONNX collapse.
- A current rep16 task-metric snapshot is checked in at [rep16_performance_snapshot.md](../../logs/hybrid_follow_val/rep16_performance_snapshot.md): float checkpoint `follow_score = 0.0194`, exported ONNX `follow_score = 0.6545`, and baseline GVSOC/application `follow_score = 0.6716`.

## Important Directories

- Frontnet reference: `nanocockpit/src/gap/examples/pulp-frontnet/app/networks/frontnet-160x32-bgaug`
- Working generated app: `pytorch_ssd/application`
- Hybrid export artifacts: `pytorch_ssd/export/hybrid_follow`
- Real-image validation outputs: `pytorch_ssd/export/hybrid_follow/real_image_validation`

## Recommended Workflow

1. Run `run_all.sh` to preflight-check the model, export it, clean the ONNX, and generate the DORY app into `pytorch_ssd/application`.
2. Recheck the generated runtime against the checklist in [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md).
3. Run `run_val.sh aideck` to build the app inside the AI-Deck Docker container and compare the final GVSOC tensor with the golden export.
4. Run `run_val.sh real` when you want the same validation flow driven by real images instead of the default staged sample.
5. Run the stage-drift comparison when you need to pinpoint where semantic drift begins before runtime.
6. Read the quant sweep recommendation in `export/hybrid_follow/quant_operator_sweep/run_all/summary.md` before changing deploy-time scale or bias policy defaults.
7. Use [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md) when the drift is centered around `stage4.1.add` or other residual adds.
8. Run `python export/check_export_nemo_quant_determinism.py --overwrite` after exporter refactors so importer safety and preset determinism are checked before broader validation.
9. When you need runtime layer localization, generate `export/hybrid_follow/gap8_layer_manifest.json` and enable `--trace-layer-outputs` in the real-image validation flow.

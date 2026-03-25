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
- [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md): historical note for the later exporter rewrite that was evaluated before the baseline rollback.
- [10-baseline-restoration.md](10-baseline-restoration.md): records the restored `906c1aa` exporter baseline, the March 24, 2026 verification run, and the inferred regression source.

## Current Validated State

- `pytorch_ssd/run_all.sh` defaults to `MODEL_TYPE=hybrid_follow`.
- The generated DORY app is written to `pytorch_ssd/application`.
- `run_all.sh` no longer touches `crazyflie_ssd` unless `SYNC_TO_CRAZYFLIE=1` is set explicitly.
- `pytorch_ssd/run_aideck_val.sh` uses `pytorch_ssd/application` as the primary app source.
- Real-image batch validation writes outputs under `pytorch_ssd/export/hybrid_follow/real_image_validation/`.
- `export_nemo_quant.py` has been restored to the last known-good `906c1aa` hybrid-follow behavior, with only a parser-only `--calib-seed` compatibility shim added for `run_all.sh`.
- `models/hybrid_follow_net.py` already matched that baseline, so no functional model rollback was needed.
- The March 24, 2026 staged `run_aideck_val.sh` run passed with `ReluConvolution0` checksum OK and final tensor `6048 3223 181711`.
- The March 24, 2026 `run_real_image_val.sh` rerun over `data/rep_images` passed `16/16` on final-tensor comparison; per-layer checksum prints in those sample-specific GVSOC logs still reflect the staged default `inputs.hex` and are not the acceptance signal for alternate real-image inputs.
- The generated app is back on the raw residual add path instead of the plain `uint8_t` residual helper path.

## Important Directories

- Frontnet reference: `nanocockpit/src/gap/examples/pulp-frontnet/app/networks/frontnet-160x32-bgaug`
- Working generated app: `pytorch_ssd/application`
- Hybrid export artifacts: `pytorch_ssd/export/hybrid_follow`
- Real-image validation outputs: `pytorch_ssd/export/hybrid_follow/real_image_validation`

## Recommended Workflow

1. Run `run_all.sh` to preflight-check the model, export it, clean the ONNX, and generate the DORY app into `pytorch_ssd/application`.
2. Recheck the generated runtime against the checklist in [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md).
3. Run `run_aideck_val.sh` to build the app inside the AI-Deck Docker container and compare the final GVSOC tensor with the golden export.
4. Run `run_real_image_val.sh` when you want the same validation flow driven by real images instead of the default staged sample.
5. Run the stage-drift comparison when you need to pinpoint where semantic drift begins before runtime.
6. Use [10-baseline-restoration.md](10-baseline-restoration.md) for the current restored exporter baseline and verified commands.
7. Treat [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md) as historical context for the later exporter rewrite rather than the current recommended baseline.

# Hybrid Follow GAP8 Notes

This folder explains how the working `hybrid_follow` GAP8 deployment differs from the reference Frontnet runtime in `nanocockpit`, which patches were required to make it work, and how to regenerate and validate the app.

## Document Map

- [01-network-comparison.md](01-network-comparison.md): describe the `hybrid_follow` application layout and the runtime API structure.
- [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md): the complete patch set that made the generated network work.
- [03-residual-branch-debugging.md](03-residual-branch-debugging.md): the residual-branch dtype investigation and the final root cause.
- [04-run-all.md](04-run-all.md): how to export, quantize, generate, and stage the app with `run_all.sh`.
- [05-run-aideck-val.md](05-run-aideck-val.md): how to build and validate the generated app with the AI-Deck Docker flow.
- [06-real-image-validation.md](06-real-image-validation.md): how to run real-image validation batches and where the per-image artifacts land.

## Current Validated State

- `pytorch_ssd/run_all.sh` defaults to `MODEL_TYPE=hybrid_follow`.
- The generated DORY app is written to `pytorch_ssd/application`.
- `run_all.sh` no longer touches `crazyflie_ssd` unless `SYNC_TO_CRAZYFLIE=1` is set explicitly.
- `pytorch_ssd/run_aideck_val.sh` uses `pytorch_ssd/application` as the primary app source.
- Real-image batch validation writes outputs under `pytorch_ssd/export/hybrid_follow/real_image_validation/`.

## Important Directories

- Frontnet reference: `nanocockpit/src/gap/examples/pulp-frontnet/app/networks/frontnet-160x32-bgaug`
- Working generated app: `pytorch_ssd/application`
- Hybrid export artifacts: `pytorch_ssd/export/hybrid_follow`
- Real-image validation outputs: `pytorch_ssd/export/hybrid_follow/real_image_validation`

## Recommended Workflow

1. Run `run_all.sh` to export the model, clean the ONNX, and generate the DORY app into `pytorch_ssd/application`.
2. Recheck the generated runtime against the checklist in [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md).
3. Run `run_aideck_val.sh` to build the app inside the AI-Deck Docker container and compare the final GVSOC tensor with the golden export.
4. Run `run_real_image_val.sh` when you want the same validation flow driven by real images instead of the default staged sample.

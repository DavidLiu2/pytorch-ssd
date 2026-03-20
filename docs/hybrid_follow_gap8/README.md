# Hybrid Follow GAP8 Notes

This folder explains how the working `hybrid_follow` GAP8 deployment differs from the reference Frontnet runtime in `nanocockpit`, which patches were required to make it work, and how to regenerate and validate the app.

## Document Map

- [01-network-comparison.md](01-network-comparison.md): compare the Frontnet reference runtime with the generated `hybrid_follow` runtime.
- [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md): the complete patch set that made the generated network work.
- [03-residual-branch-debugging.md](03-residual-branch-debugging.md): the residual-branch dtype investigation and the final root cause.
- [04-run-all.md](04-run-all.md): how to export, quantize, generate, and sync the app with `run_all.sh`.
- [05-run-aideck-val.md](05-run-aideck-val.md): how to build and validate the generated app with the AI-Deck Docker flow.

## Current Validated State

- `pytorch_ssd/run_all.sh` defaults to `MODEL_TYPE=hybrid_follow`.
- The generated DORY app is written to `crazyflie_ssd/generated`.
- `run_all.sh` also syncs `hex/` and `vars.mk` back into `crazyflie_ssd/`.
- `pytorch_ssd/run_aideck_val.sh` uses `crazyflie_ssd/generated` as the primary app source.
- GVSOC validation currently passes with final tensor `901 11011 257657`.

## Important Directories

- Frontnet reference: `nanocockpit/src/gap/examples/pulp-frontnet/app/networks/frontnet-160x32-bgaug`
- Working generated app: `crazyflie_ssd/generated`
- Hybrid export artifacts: `pytorch_ssd/export/hybrid_follow`
- Deployment wrapper app: `crazyflie_ssd`

## Recommended Workflow

1. Run `run_all.sh` to export the model, clean the ONNX, generate the DORY app, and sync it into `crazyflie_ssd/generated`.
2. Recheck the generated runtime against the checklist in [02-generated-network-patch-guide.md](02-generated-network-patch-guide.md).
3. Run `run_aideck_val.sh` to build the app inside the AI-Deck Docker container and compare the final GVSOC tensor with the golden export.

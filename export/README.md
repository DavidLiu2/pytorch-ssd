# Export Entry Points

Supported quant/export entry points for current follow-model work:

- `run_plain_follow_release.py`: canonical plain_follow production wrapper. Builds the calibration manifest, materializes the expanded validation pack, runs float overlays, runs quant export/eval, and writes a single release summary.
- `evaluate_quant_native_follow.py`: canonical plain-follow and dronet-lite quant/export evaluation.
- `build_follow_calibration_manifest.py`: ranked deployment-matched calibration or sample-subset manifests.
- `compare_quant_native_follow_rep16_overlays.py`: rep16 pre/post overlay generation.
- `../run_plain_follow.sh`: top-level shell wrapper for the plain_follow production flow.
- `../run_all.sh` and `../run_val.sh`: legacy/general wrappers that remain useful for hybrid-follow and older validation paths.

Historical one-off studies and legacy debugging helpers were moved under [archive](./archive/README.md). They are intentionally out of the main path and are no longer the recommended workflow.

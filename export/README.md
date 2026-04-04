# Export Entry Points

Supported quant/export entry points for current follow-model work:

- `evaluate_quant_native_follow.py`: canonical plain-follow and dronet-lite quant/export evaluation.
- `build_follow_calibration_manifest.py`: ranked deployment-matched calibration or sample-subset manifests.
- `run_plain_follow_quant_improvement.py`: supported plain-follow baseline-vs-focused-QAT study loop.
- `compare_quant_native_follow_rep16_overlays.py`: rep16 pre/post overlay generation.
- `../run_all.sh` and `../run_val.sh`: canonical wrapper scripts for export and validation.

Historical one-off studies and legacy debugging helpers were moved under [archive](./archive/README.md). They are intentionally out of the main path and are no longer the recommended workflow.

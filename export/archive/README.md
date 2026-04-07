# Archived Export Studies

This folder holds older experimental helpers that were useful during quantization bring-up, but are no longer part of the supported export workflow.

Archived scripts:

- `run_quant_native_follow_study.py`
- `run_hybrid_follow_stage4_variant_study.py`
- `run_hybrid_follow_earliest_bad_op.py`
- `compare_gap8_final_tensor.py`
- `compare_gap8_outputs.py`
- `batch_localize_hybrid_follow_stages.py`

They are kept for reference and recoverability, but new work should start from [../README.md](../README.md) and the current supported entry points in `pytorch_ssd/export/`.

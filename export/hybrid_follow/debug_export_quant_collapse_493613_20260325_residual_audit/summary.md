# Hybrid Follow Quant Drift Debug Report

- Requested stage: `id`
- Exported stage: `id`
- Selected input: `export/hybrid_follow/debug_export_quant_collapse_493613_20260325/input/000000493613.jpg`
- First bad location: `None`
- Suspect scale metadata count: `4`
- Fusion/head-collapse effect: `no_collapse_in_completed_variants;errors=variant_b_unfused_single_head`
- Clamp changed result: `None`
- Round export params: `False`
- Diagnosis: `deploy_stage_residual_scale_warning_but_export_matches_deploy`
- FQ->deploy x abs diff: `0.146770`
- FQ->deploy size abs diff: `0.224178`
- FQ->deploy vis conf abs diff: `0.162816`
- Deploy->ONNX raw max abs diff: `0.000000`
- Residual scale warning severity: `warning`
- Largest FQ->ID drift point: `stage4.1.add pre-requant`
- stage4.1.add pre->post requant mean abs diff: `0.008937`

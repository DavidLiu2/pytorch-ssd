# Hybrid Follow Quant Drift Debug Report

- Requested stage: `id`
- Exported stage: `id`
- Selected input: `export\hybrid_follow\debug_export_quant_collapse_493613_20260325\input\000000493613.jpg`
- First bad location: `None`
- Suspect scale metadata count: `2`
- Fusion/head-collapse effect: `no_collapse_in_completed_variants;errors=variant_b_unfused_single_head`
- Clamp changed result: `None`
- Round export params: `False`
- Integer add scale policy: `sqrt_fanin`
- Diagnosis: `no_collapse_detected`
- FQ->deploy x abs diff: `0.057118`
- FQ->deploy size abs diff: `0.399602`
- FQ->deploy vis conf abs diff: `0.553032`
- Deploy->ONNX raw max abs diff: `0.000000`
- Largest FQ->ID drift point: `stage4.1.add pre-requant`
- stage4.1.add pre->post requant mean abs diff: `0.006304`
- Integer add selected policy on known sample: `fanin`
- Integer add active/selected score: `1.009752065082208` -> `0.996848421333701`

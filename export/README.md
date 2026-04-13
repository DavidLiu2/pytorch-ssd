# Export Entry Points

Supported quant/export entry points for current follow-model work:

- `run_plain_follow_release.py`: canonical plain_follow production wrapper. Builds the calibration manifest, materializes the expanded validation pack, runs float overlays, runs quant export/eval, runs the DORY-semantic deployment simulator built from cleaned `model_id_dory.onnx`, selects the deployment threshold on those predictions, runs deployment-side compare overlays, runs a GVSOC smoke, and writes a single release summary.
- `evaluate_quant_native_follow.py`: canonical plain-follow and dronet-lite quant/export evaluation. Exports both `model_id.onnx` and cleaned `model_id_dory.onnx`, audits Conv/Gemm weight ranges for DORY, clamps rounded out-of-range weights into signed int8 when needed, and emits a `dory_cleanup_report`.
- `dory_semantic_follow_inference.py`: DORY-graph semantic simulator for deployment-side follow validation. It parses the cleaned DORY ONNX through the DORY frontend/backend graph path and emits raw integer predictions that match DORY codegen semantics much more closely than ONNXRuntime on the cleaned ONNX. The preferred bundle contract now restages uint8 inputs from the source image using the same preprocessing path as the seeded DORY app flow.
- `generate_dory_io_artifacts.py`: image- or seed-driven DORY artifact writer. Generates `input.txt`, `output.txt`, and `out_layer*.txt` files for the exact cleaned DORY ONNX so `network_generate` can embed meaningful layer checksums into the app.
- `compare_gap8_layer_bytes.py`: layer-by-layer GAP8 runtime comparer. Reads `LAYER_BYTES` traces from the GVSOC log and compares them against the `out_layer*.txt` golden tensors that were seeded into `network_generate`.
- `../tools/patch_gap8_bn_quant_int64.py`: targeted generated-app patcher for GAP8 requant helpers. It upgrades the generated `pulp_nn_utils.c` BN/requant helpers to use `int64` intermediates before the right shift, which avoids the app-side overflow that caused the remaining `plain_follow` runtime drift.
- `build_follow_calibration_manifest.py`: ranked deployment-matched calibration or sample-subset manifests.
- `compare_quant_native_follow_rep16_overlays.py`: rep16 pre/post overlay generation.
- `../run_plain_follow.sh`: top-level shell wrapper for the plain_follow production flow.
- `../run_all.sh` and `../run_val.sh`: legacy/general wrappers that remain useful for hybrid-follow and older validation paths.

Historical one-off studies and legacy debugging helpers were moved under [archive](./archive/README.md). They are intentionally out of the main path and are no longer the recommended workflow.

Current deployment caveat:

- The release wrapper now treats the cleaned `model_id_dory.onnx` as the source graph for a DORY-semantic Python simulator, and that simulator is the deployment-facing validator for threshold tuning and compare overlays.
- ONNXRuntime on cleaned `model_id_dory.onnx` is no longer the main deployment proxy, because DORY integerizes some requant/BN parameters again during codegen.
- The generated GAP8 app now gets an automatic `int64` requant patch before GVSOC. The root cause was `int32` overflow in generated BN/requant helpers such as `pulp_nn_bn_quant_u8`; the patch is applied both by `run_plain_follow_release.py` and by `tools/run_aideck_val_impl.sh` when `HOST_PATCH_BN_QUANT_INT64=1`.
- The GVSOC smoke now seeds `network_generate` with image-specific `input.txt`, `output.txt`, and `out_layer*.txt` files generated from cleaned `model_id_dory.onnx`, so the generated app carries nonzero per-layer checksums for the selected smoke image.
- When app drift still needs debugging, the supported repro path is: generate the seeded app, run GVSOC with layer tracing enabled, then run `compare_gap8_layer_bytes.py` on the emitted log plus the generated layer manifest.
- GVSOC tensor verification remains the final deployment gate.

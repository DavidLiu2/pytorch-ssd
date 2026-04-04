# Generated Network Patch Guide

This is the patch set that made the generated `hybrid_follow` GAP8 app work.

Current note as of March 25, 2026:

- these runtime patches are still required
- they are not the only thing to check anymore
- the current open debug issue is earlier, in the FQ -> ID residual-add scale transition documented in [09-export-runtime-residual-fix.md](09-export-runtime-residual-fix.md)

## The Short Version

The working system needed fixes in four places:

1. Export the model in a DORY-friendly form.
2. Fix DORY layout handling for the final fully connected layer.
3. Fix the generated runtime so raw branch outputs stay `int32_t` until the residual add is finished.
4. Fix the final requantization helper so large negative accumulators do not wrap and turn into `255`.
5. Verify deploy-time fused conv biases were integerized with `eps_out_static` after `id_stage()`.
6. Audit and tune `PACT_IntegerAdd` scale selection when FQ -> ID drift remains after export/runtime checks pass.

## Persistent Fixes vs Generated Fixes

Some fixes live in source files that survive regeneration. Others live inside the generated app and should be rechecked after every fresh `run_all.sh`.

## Persistent Fixes

These should remain in place across regenerations.

| File | Change | Why |
| --- | --- | --- |
| `pytorch_ssd/nemo/export_nemo_quant.py` | collapse the three hybrid heads into one 3-output linear layer | avoids a DORY-hostile multi-head export with separate outputs and `Concat` |
| `pytorch_ssd/nemo/export_nemo_quant.py` | clamp `Conv` / `Gemm` / `MatMul` initializers into int8 range | prevents export-time weights from exceeding what the GAP kernels expect |
| `pytorch_ssd/nemo/export_nemo_quant.py` | integerize deploy-time fused conv biases with `eps_out_static` after `id_stage()` | fixes the March 26, 2026 upstream FQ -> ID distortion in the stage4.1 main branch |
| `pytorch_ssd/nemo/export_nemo_quant.py` | instrument `PACT_IntegerAdd` scale selection and compare candidate policies | current residual-stage drift is centered around FQ -> ID integer add scaling once the upstream conv path is fixed |
| `dory/dory/Hardware_targets/PULP/Common/HW_Parser.py` | find the true source node for fully connected layout adjustment | keeps FC weight layout correct when the producer is not simply `node_id - 1` |
| `dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit/src/pulp_nn_utils.c` | use `int32_t` inside `pulp_nn_quant_u8()` | fixes requant overflow in future codegen |
| `dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/64bit/src/pulp_nn_utils.c` | same `pulp_nn_quant_u8()` fix | keeps both backend copies aligned |
| `crazyflie_ssd/app_config.h` | set `APP_NET_OUTPUT_BYTES` to `12u` | the model outputs 3 `int32` values |
| `crazyflie_ssd/src/net_runner.c` | integrate with the generated app using app-level config instead of missing helper macros | keeps deployment code simple and stable |

## Generated-App Fixes

These live under `crazyflie_ssd/generated` and should be rechecked after regeneration.

| File group | Change | Why |
| --- | --- | --- |
| `src/network.c` | fix `network_run_async()` arg packing to use `args[5]` | prevents corrupted cluster arguments |
| `src/Convolution*.c` raw branch wrappers | cast bias to `const int32_t *` and preserve the raw output path | raw branch outputs feed residuals and cannot be clipped early |
| `src/ReluQAddition*.c` wrappers | replace generated add path with raw-branch helpers | ensures residual math happens in the intended dtype |
| `src/pulp_nn_conv_Ho_parallel.c` | treat no-ReLU/no-BN output as `int32_t` | fixes raw convolution output storage |
| `src/pulp_nn_matmul.c` | keep raw output path in `int32_t`, fix bias handling | same issue for the matmul backend |
| `src/pulp_nn_linear_out_32.c` | read bias as true `int32_t` | final FC output must be numerically correct |
| `src/pulp_nn_add.c` | add raw residual kernels | keeps residual accumulation in the intended dtype |
| `src/pulp_nn_utils.c` | fix `pulp_nn_quant_u8()` | this was the final correctness fix |

## Export-Side Changes

## 1. Collapse the Three Hybrid Heads Into One FC Layer

The original model had three separate heads:

- `head_x`
- `head_size`
- `head_vis`

For DORY export, these were collapsed into one 3-output linear layer in `HybridFollowExportNet` inside `pytorch_ssd/nemo/export_nemo_quant.py`.

Why this mattered:

- DORY handled a single fully connected output path much better than three separate heads followed by concatenation.
- The generated GAP app now ends in one `FullyConnected30` layer with 12 output bytes.

## 2. Clamp DORY Weight Initializers

`clamp_dory_weight_initializers_to_int8()` now scans the ONNX graph and clips weight initializers for `Conv`, `Gemm`, and `MatMul` nodes into `[-128, 127]`.

Why this mattered:

- the GAP kernels are effectively int8 kernels
- a few exported weights were outside the valid range
- clipping removed a silent source of numeric mismatch before the model even reached DORY

## DORY Parser Change

## 3. Fix FC Layout Tracking

`HW_Parser.py` now uses `_find_source_node()` when adjusting fully connected weights.

Why this mattered:

- the old logic assumed the producer of a fully connected layer was always the previous node
- that assumption is fragile in generated graphs
- the final FC layout could be rearranged incorrectly when the true source was a different node or had CHW layout

## Runtime Integration Changes

## 4. Update the App-Level Contract

`crazyflie_ssd/app_config.h` now sets:

```c
#define APP_NET_OUTPUT_BYTES (12u)
```

`crazyflie_ssd/src/net_runner.c` allocates the output buffer with that size and runs the generated network without depending on missing `NETWORK_OUTPUT_*` helpers.

Why this mattered:

- the deployed app now agrees with the actual model output size
- this avoids stale assumptions from older SSD-style deployments

## Generated Runtime Changes

## 5. Fix Cluster Arg Packing

`crazyflie_ssd/generated/src/network.c` uses `unsigned int args[5]` in `network_run_async()`.

Why this mattered:

- the generated runtime passes five values into the cluster entry
- using four slots corrupts the `initial_dir` argument and destabilizes execution

## 6. Keep Raw Branch Convolution Outputs in `int32_t`

The raw branch wrappers are:

- `Convolution1.c`
- `Convolution3.c`
- `Convolution6.c`
- `Convolution8.c`
- `Convolution10.c`
- `Convolution13.c`
- `Convolution15.c`
- `Convolution17.c`
- `Convolution20.c`
- `Convolution22.c`
- `Convolution24.c`
- `Convolution27.c`

The important rules are:

- biases must be read as `int32_t`
- no-ReLU and no-BN output must stay in `int32_t`
- the output must not be clipped to `uint8_t` before the residual add consumes it

The core support for this lives in `pulp_nn_conv_Ho_parallel.c` and `pulp_nn_matmul.c`.

## 7. Patch Residual Add Wrappers

The residual wrappers are:

- `ReluQAddition4.c`
- `ReluQAddition7.c`
- `ReluQAddition11.c`
- `ReluQAddition14.c`
- `ReluQAddition18.c`
- `ReluQAddition21.c`
- `ReluQAddition25.c`
- `ReluQAddition28.c`

These now call raw helpers in `pulp_nn_add.c`:

- `pulp_nn_add_raw_i32_u8()`
- `pulp_nn_add_raw_i32_u8_mixed()`

Why this mattered:

- projection-style residuals can require `int32 + int32`
- identity-style residuals can require `int32 + uint8`
- requantization should happen once, at the residual output boundary

## 8. Fix Requant Overflow

The final correctness bug was in `pulp_nn_quant_u8()`:

```c
int32_t x = (m * phi) >> d;
```

This must stay `int32_t`. A narrower temporary can wrap negative values and make them clip to `255`.

Why this mattered:

- this was the last remaining source of final tensor mismatch
- after this fix, the GVSOC final tensor matched the golden export exactly

## Temporary Debug Instrumentation

Temporary debug dumps were used during bring-up in:

- `pulp_nn_utils.c`
- `pulp_nn_add.c`
- selected `Convolution*.c` wrappers

The dumps report:

- checksum-style `sum_mod32`
- `hash32`
- first 8 values

They were useful for validating the first working app and were later removed from `pytorch_ssd/application` once the pipeline stabilized.

## Minimal Recheck After A Fresh `run_all.sh`

After every fresh generation, verify these points before blaming the model:

1. `crazyflie_ssd/generated/src/network.c` still uses `args[5]`.
2. `crazyflie_ssd/generated/src/pulp_nn_utils.c` still uses `int32_t` inside `pulp_nn_quant_u8()`.
3. Raw `Convolution*.c` wrappers still cast bias to `const int32_t *`.
4. Residual `ReluQAddition*.c` wrappers still call the raw add helpers.
5. `crazyflie_ssd/app_config.h` still says `APP_NET_OUTPUT_BYTES (12u)`.
6. The generated app still ends in a single `FullyConnected30` with a 12-byte output.

## Final Outcome

With the patch set above in place:

- the generated app builds in the AI-Deck Docker flow
- all layer checksum checks pass in GVSOC
- the final tensor matches the export golden output exactly

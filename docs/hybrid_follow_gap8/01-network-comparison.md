# Frontnet vs Hybrid-Follow Network Runtime

This note compares the reference Frontnet runtime in `nanocockpit` with the working generated `hybrid_follow` runtime in `crazyflie_ssd/generated`.

## Why This Comparison Matters

The Frontnet network is a good example of a cleaned-up, application-facing GAP runtime. The `hybrid_follow` network is a much more direct DORY output with residual branches, more layers, and fewer app-friendly metadata helpers. The working deployment came from keeping the generated network structure but patching the places where DORY assumptions did not match the hybrid model.

## High-Level Differences

| Topic | Frontnet reference | Working hybrid-follow runtime | Why it mattered |
| --- | --- | --- | --- |
| Entry point | `network_init()` and async API with explicit input/output pointers | `network_initialize()` plus generated `network_run_async()` / `network_run()` wrappers | Frontnet is easier to integrate into a larger app; hybrid-follow needed local wrapper fixes instead of a full API rewrite |
| Runtime arg passing | `network_args_t` struct | raw `unsigned int args[5]` array | the generated app previously had an arg packing bug; the fixed runtime uses `args[5]` |
| Metadata in `network.h` | typed input/output macros and `NETWORK_L2_BUFFER_SIZE` | mostly raw generated arrays and constants | `crazyflie_ssd` had to rely on app-level config instead of generated helper macros |
| Layer count | 9 layers | 31 layers | the hybrid model is deeper and uses more generated wrappers and kernels |
| Residual topology | no residual branches | multiple residual branches via `branch_input`, `branch_output`, `branch_change` | this is where the dtype mismatch work happened |
| L3 activation buffers | `L3_INPUT_SIZE = 0`, `L3_OUTPUT_SIZE = 0` | both set to `1500000` | hybrid-follow uses the full DORY memory flow |
| Final output | 4 `int32` outputs, 16 bytes | 3 `int32` outputs, 12 bytes | `crazyflie_ssd/app_config.h` had to be updated to 12 bytes |
| Final copy | `memmove(args->l2_output, L2_output, activations_out_size[8])` | manual byte copy of `activations_out_size[30]` | not the main bug, but useful when comparing code style |
| Verbose control | `NETWORK_VERBOSE` macro | `#define VERBOSE 1` in generated file | the generated runtime is less configurable out of the box |
| Output helper | includes `network_dequantize_output()` | no equivalent helper | hybrid-follow output stays in raw integer form |

## Concrete Code Differences

## API and Buffering

Frontnet uses a typed async API and validates `NETWORK_L2_BUFFER_SIZE` before running. It also knows whether the input lives inside or outside the managed L2 arena. This logic is in `nanocockpit/.../src/network.c`.

The generated hybrid runtime uses:

- `network_initialize()`
- `network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)`
- `network_run_wait()`
- `network_run()`

That API is simpler, but it exposes more DORY internals directly to the caller.

## Topology

Frontnet has no residuals:

- `branch_input[9]` is all zero
- `branch_output[9]` is all zero
- `branch_change[9]` is all zero

Hybrid-follow uses residual branches throughout the network:

- `branch_input[31]`
- `branch_output[31]`
- `branch_change[31]`

That branch structure is the main reason the generated app needed manual dtype fixes.

## Output Shape

Frontnet ends in a 4-value fully connected layer and exports `NETWORK_OUTPUT_COUNT = 4`.

Hybrid-follow ends in `FullyConnected30` and `activations_out_size[30] = 12`, which corresponds to 3 `int32` values.

## Integration Style

Frontnet is written like a reusable runtime library. The generated hybrid runtime is written like a codegen artifact. In practice, this means:

- borrow Frontnet design ideas when cleaning up integration code
- do not assume Frontnet can be copied directly onto hybrid-follow
- patch the generated kernels and wrappers where DORY emits the wrong assumptions

## What To Borrow From Frontnet

- Stronger runtime metadata in `network.h`
- Clearer input/output typing
- Safer argument passing through a struct instead of a raw integer array
- Cleaner final output copy
- Optional dequantization helper for app-level debugging

## What Not To Copy Blindly

- The no-residual assumptions
- The 4-output head contract
- The zero-L3 activation assumptions
- The simpler memory schedule

Hybrid-follow works because its generated structure was preserved and the incorrect dtype, parser, and export assumptions were fixed around that structure.

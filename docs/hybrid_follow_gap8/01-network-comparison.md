# Application Layout and API Structure

This note describes how the working `hybrid_follow` GAP8 application is organized in the repo and how the main runtime APIs fit together.

## Big Picture

The deployment is split into three layers:

1. Export and code generation in `pytorch_ssd`
2. The generated DORY network bundle in `pytorch_ssd/application`
3. An optional later sync into an external wrapper app, if you choose to copy it elsewhere

That separation is useful because the generated network changes when the model changes, while the local export and validation pipeline stays relatively stable.

## Directory Layout

| Path | Role |
| --- | --- |
| `pytorch_ssd/` | training, export, codegen, validation scripts, and documentation |
| `pytorch_ssd/export/hybrid_follow/` | ONNX exports, DORY-cleaned ONNX, golden outputs, and weight text artifacts |
| `pytorch_ssd/application/` | the generated GAP8 network bundle that DORY emits |
| `pytorch_ssd/application/inc/` | generated public headers and layer/kernel declarations |
| `pytorch_ssd/application/src/` | generated runtime, layer wrappers, and backend kernels |
| `pytorch_ssd/application/hex/` | readfs weight and IO blobs used by the generated app |

## What Lives In The Generated Bundle

The generated bundle in `pytorch_ssd/application` is a self-contained network application.

### `inc/`

This directory holds:

- `network.h` with the generated runtime API and network metadata
- layer headers such as `Convolution1.h` and `ReluQAddition7.h`
- kernel headers such as `pulp_nn_kernels.h`

### `src/`

This directory holds three main classes of source files:

- `network.c`, which is the top-level runtime orchestrator
- layer wrapper files such as `Convolution*.c`, `ReluConvolution*.c`, `ReluQAddition*.c`, and `FullyConnected30.c`
- backend kernel files such as `pulp_nn_conv_Ho_parallel.c`, `pulp_nn_matmul.c`, `pulp_nn_add.c`, and `pulp_nn_utils.c`

### `hex/`

This directory holds the readfs artifacts that the generated app expects, including:

- weight blobs
- generated input/output artifacts

### Build Metadata

The generated bundle also includes:

- `Makefile`
- `vars.mk`

These are what let the generated app be dropped into the GAP build flow without having to hand-write the layer inventory each time.

## What Lives In The Wrapper App

The local validation pipeline now stops at `pytorch_ssd/application`.

If you later copy this bundle into another wrapper app, the key idea is still the same: the generated bundle owns network internals, while the wrapper owns how the rest of the application interacts with that network.

## Generated Runtime API

The generated runtime API is declared in `pytorch_ssd/application/inc/network.h` and implemented in `pytorch_ssd/application/src/network.c`.

The main entry points are:

- `network_initialize()`
- `network_terminate()`
- `network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)`
- `network_run_wait(struct network_run_token token)`
- `network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)`

### What Each Call Does

- `network_initialize()` allocates L3 buffers and loads the generated readfs weights.
- `network_run_async()` packages the runtime arguments and sends work to the GAP cluster.
- `network_run_wait()` closes out the cluster task and prints final performance stats.
- `network_run()` is the synchronous convenience wrapper around `network_run_async()` plus `network_run_wait()`.
- `network_terminate()` frees the generated runtime's L3 allocations.

## Layering Inside `network.c`

`pytorch_ssd/application/src/network.c` is the top of the generated runtime stack.

Conceptually it does four jobs:

1. Allocate and preload L3 resources
2. Set up the directional allocator for the managed L2 arena
3. Iterate through generated layer wrappers in execution order
4. Copy the final output tensor into the caller-provided output buffer

The layer wrappers then call the lower-level `pulp_nn_*` kernels that do the actual math.

## Data Flow Through The App

The normal flow is:

1. `pytorch_ssd/run_all.sh` exports the model and generates the app into `pytorch_ssd/application`.
2. `pytorch_ssd/application/src/main.c` or `pytorch_ssd/aideck_val_main_hybrid.c` loads `inputs.hex` and prepares the input/output buffer.
3. `network_initialize()` loads the readfs weights into L3.
4. `network_run()` dispatches the generated runtime on the cluster.
5. `network.c` walks through the generated layer wrappers and kernels.
6. The final output tensor is written into the caller-provided output buffer.
7. The caller reads 3 `int32` outputs, which correspond to 12 bytes total.

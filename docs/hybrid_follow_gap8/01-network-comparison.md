# Application Layout and API Structure

This note describes how the working `hybrid_follow` GAP8 application is organized in the repo and how the main runtime APIs fit together.

## Big Picture

The deployment is split into three layers:

1. Export and code generation in `pytorch_ssd`
2. The generated DORY network bundle in `crazyflie_ssd/generated`
3. The application-facing wrapper in `crazyflie_ssd`

That separation is useful because the generated network changes when the model changes, while the wrapper app and pipeline scripts stay relatively stable.

## Directory Layout

| Path | Role |
| --- | --- |
| `pytorch_ssd/` | training, export, codegen, validation scripts, and documentation |
| `pytorch_ssd/export/hybrid_follow/` | ONNX exports, DORY-cleaned ONNX, golden outputs, and weight text artifacts |
| `crazyflie_ssd/generated/` | the generated GAP8 network bundle that DORY emits |
| `crazyflie_ssd/generated/inc/` | generated public headers and layer/kernel declarations |
| `crazyflie_ssd/generated/src/` | generated runtime, layer wrappers, and backend kernels |
| `crazyflie_ssd/generated/hex/` | readfs weight and IO blobs used by the generated app |
| `crazyflie_ssd/` | deployment wrapper app that owns config, runtime allocation, and integration |

## What Lives In The Generated Bundle

The generated bundle in `crazyflie_ssd/generated` is a self-contained network application.

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

The wrapper app in `crazyflie_ssd` is the stable integration layer around the generated network.

The most important files are:

- `app_config.h`, which defines app-level contracts such as input geometry, arena size, and `APP_NET_OUTPUT_BYTES`
- `src/net_runner.c`, which allocates buffers, calls the generated network, and exposes a simpler app-facing API

The key idea is that `crazyflie_ssd/generated` owns network internals, while `crazyflie_ssd` owns how the rest of the application interacts with that network.

## Generated Runtime API

The generated runtime API is declared in `crazyflie_ssd/generated/inc/network.h` and implemented in `crazyflie_ssd/generated/src/network.c`.

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

## Wrapper App API

The wrapper API is implemented in `crazyflie_ssd/src/net_runner.c`.

The main calls are:

- `net_runner_init()`
- `net_runner_get_input_buffer()`
- `net_runner_get_output_buffer()`
- `net_runner_get_output_size()`
- `net_runner_run()`

### Why The Wrapper Exists

The generated runtime API is low-level and DORY-specific. The wrapper app makes the contract simpler:

- the application asks for an input buffer
- the wrapper owns the L2 arena and output allocation
- the wrapper calls `network_initialize()` once and `network_run()` when needed
- the application reads back a fixed-size output buffer

This is also the layer that enforces the 12-byte output contract for the current hybrid-follow model.

## Layering Inside `network.c`

`crazyflie_ssd/generated/src/network.c` is the top of the generated runtime stack.

Conceptually it does four jobs:

1. Allocate and preload L3 resources
2. Set up the directional allocator for the managed L2 arena
3. Iterate through generated layer wrappers in execution order
4. Copy the final output tensor into the caller-provided output buffer

The layer wrappers then call the lower-level `pulp_nn_*` kernels that do the actual math.

## Data Flow Through The App

The normal flow is:

1. `pytorch_ssd/run_all.sh` exports the model and generates the app into `crazyflie_ssd/generated`.
2. `crazyflie_ssd/src/net_runner.c` allocates the input arena and output buffer.
3. `network_initialize()` loads the readfs weights into L3.
4. `network_run()` dispatches the generated runtime on the cluster.
5. `network.c` walks through the generated layer wrappers and kernels.
6. The final output tensor is written into the wrapper's output buffer.
7. The caller reads 3 `int32` outputs, which correspond to 12 bytes total.


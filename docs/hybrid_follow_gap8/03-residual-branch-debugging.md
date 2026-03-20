# Residual Branch Debugging Notes

This note records the residual-branch investigation, because the final fix was slightly different from the original hypothesis.

## Starting Symptom

After the early fixes were in place:

- the bias blob read bug was already fixed
- export-time weight clipping was already fixed
- layer 0 checksum was almost exact
- layer 2 matched exactly
- the final tensor still mismatched

That pointed to a later numeric issue, and the residual branches were the most suspicious part of the graph.

## Original Hypothesis

The first strong hypothesis was:

- raw convolution branch outputs were being written as `uint8_t`
- residual consumers expected those values in `int32_t`
- the add path was requantizing too early

That hypothesis was directionally correct. Several raw/no-ReLU paths did need dtype fixes.

## Instrumentation Added

Temporary debug dumps were added during bring-up at four points:

1. End of raw convolution
2. Start of residual add
3. End of residual add before requantization
4. End of requantization

The dumps print:

- element count
- `sum_mod32`
- `hash32`
- first 8 elements

Files involved during the debug pass:

- `crazyflie_ssd/generated/src/pulp_nn_utils.c`
- `crazyflie_ssd/generated/src/pulp_nn_add.c`
- selected raw wrappers such as `Convolution1.c`, `Convolution10.c`, `Convolution15.c`, `Convolution22.c`, and `Convolution27.c`

Those temporary dumps were removed afterward from the `pytorch_ssd/application` copy once the runtime was validated.

## Residual Wrapper Strategy

The residual wrappers were simplified so they call explicit helpers instead of relying on the original generated add path.

Two helper styles were needed:

- `pulp_nn_add_raw_i32_u8()` for residuals whose two inputs are both logically raw
- `pulp_nn_add_raw_i32_u8_mixed()` for residuals whose bypass path is still `uint8_t`

This made it much easier to reason about where requantization should happen.

## What The Debug Data Showed

The debug pass showed:

- raw branch outputs were no longer collapsing to zero
- residual inputs could be tracked cleanly
- the mismatch was no longer at the network start

The key turning point was that the residual math itself looked structurally correct, but some requantized outputs were still wrong.

## The Actual Final Root Cause

The last failing bug was in `pulp_nn_quant_u8()` inside `pulp_nn_utils.c`.

The intermediate value must stay `int32_t`:

```c
int32_t x = (m * phi) >> d;
```

Before this fix, the code narrowed the accumulator too early. Large negative values wrapped, became positive, and then clipped to `255`.

## Why The Checksums Were Off

During the debug pass, the earliest remaining divergence was `ReluQAddition7`.

The reason was precise:

- 8 requantized values were below `-32768`
- they wrapped positive after narrowing
- they then clipped to `255`
- that added `+2040` to the checksum

So the original residual dtype suspicion found the right area, but the final blocker was actually an overflow inside the generic requant helper.

## Final Validation Result

After the `pulp_nn_quant_u8()` fix:

- all layer checksum checks passed
- the final GVSOC tensor matched the golden export
- the validated final tensor was `901 11011 257657`

The golden output lives in `pytorch_ssd/export/hybrid_follow/output.txt`.

## What To Remember For Future Debugging

When the first layers already match and the final tensor still does not:

1. Check whether raw residual branches are staying in `int32_t`.
2. Check whether bypass inputs are being reloaded in the expected dtype.
3. Check the generic requant helpers for overflow or narrowing.
4. Add dumps around the graph boundary where raw accumulators first become `uint8_t`.

In this project, the last item was the one that closed the issue.

# Model Compatibility Checks

This note summarizes what we learned from getting `hybrid_follow` back onto a clean NEMO + DORY export path, and how to use the compatibility checker that now runs inside `run_all.sh`.

## Main Findings

### `nn.Sequential` Naming Is Not The Root Problem

The current `HybridFollowNet` in [hybrid_follow_net.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/models/hybrid_follow_net.py) still uses:

- `ConvBNReLU(nn.Sequential)` for the stem
- nested `nn.Sequential(...)` blocks for `stage1` through `stage4`

That produces names such as:

- `stem.0`, `stem.1`, `stem.2`
- `stage1.0.conv1`
- `stage1.1.conv2`

These are not ideal for readability, but they are not deployment blockers by themselves.

What they affect:

- debugging readability
- eps-dictionary readability
- export graph inspection

What they do not automatically break:

- `quantize_pact()`
- `qd_stage()`
- `id_stage()`
- DORY generation

### Explicit Heads And Explicit Adds Matter More

The current model does two important things right:

- it keeps output heads explicit with `head_x`, `head_size`, and `head_vis`
- it keeps residual adds explicit with `self.add = PACT_IntegerAdd()`

Those two choices help much more than replacing every `Sequential` with manually named siblings.

### The Real Problem Was PyTorch Name To ONNX Name Drift

The issue was not “ResNet is incompatible.”

The real pain point was the mapping between:

- PyTorch module names
- NEMO graph node names
- final ONNX node and initializer names

Residual blocks amplify that problem because the export path has to keep branch merges, activation eps values, and post-BN deployment transforms aligned across two branches.

That is why we saw failures that looked like naming problems even though the architecture itself was reasonable.

## What Now Works

The current verified path for `hybrid_follow_best_follow_score.pth` is:

1. fuse Conv+BN for the export-ready hybrid-follow model
2. collapse the three heads into the single 3-output export head
3. quantize with NEMO
4. repair fused residual graph names before deploy stages
5. run direct `qd_stage(eps_in=1/255)`
6. run direct `id_stage()`
7. export ONNX
8. clean ONNX for DORY
9. validate with the DORY frontend

Important result:

- the DORY-clean ONNX now matches the validated graph family again
- no `.kappa` / `.lamda` initializers remain in the cleaned graph
- DORY frontend parsing succeeds
- `network_generate.py` succeeds

## Remaining Warning

The exported raw ONNX can still contain a few Conv/Gemm weights that round slightly outside signed int8.

That is now treated as a diagnostic, not as an automatic export failure.

Why:

- the pipeline no longer clips those weights
- the cleaned DORY ONNX is accepted by the DORY frontend
- DORY acceptance is the stronger deployment gate than a raw initializer range heuristic

So the current policy is:

- no silent clipping
- no stage fallback
- no QD fallback
- DORY frontend acceptance remains the final compatibility gate

## Compatibility Checker

The checker lives at [check_model_compatibility.py](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/check_model_compatibility.py).

It has two jobs:

1. PyTorch preflight
2. ONNX graph inspection

### PyTorch Preflight Checks

The checker reports:

- `nn.Sequential` numeric naming usage
- numeric path segments such as `stage1.0.conv1`
- whether heads are explicitly named
- whether add/merge modules are explicit
- whether functional `relu` or plain functional `add` patterns appear
- whether `torch.fx.symbolic_trace()` succeeds
- whether the export-ready model completes a dry-run `qd_stage()` / `id_stage()`

### ONNX Checks

The checker reports:

- ONNX op counts
- unexpected ops outside the currently validated DORY-friendly set
- whether `BatchNormalization` or QDQ nodes remain
- whether `.kappa` / `.lamda` initializers remain
- whether Conv/Gemm/MatMul weights round outside signed int8

### JSON Reports

`run_all.sh` now writes:

- [model_compat_python.json](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/model_compat_python.json)
- [model_compat_onnx.json](/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export/hybrid_follow/model_compat_onnx.json)

These are useful for comparing checkpoints over time.

## Manual Usage

### Check the PyTorch model only

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
source ../nemoenv/bin/activate

python export/check_model_compatibility.py \
  --mode python \
  --model-type hybrid_follow \
  --ckpt training/hybrid_follow/hybrid_follow_best_follow_score.pth \
  --height 128 \
  --width 128 \
  --input-channels 1 \
  --calib-dir data/coco/images/val2017 \
  --fail-on-errors
```

### Check the ONNX graphs

```bash
cd /mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd
source ../doryenv/bin/activate

python export/check_model_compatibility.py \
  --mode onnx \
  --onnx export/hybrid_follow/hybrid_follow_quant.onnx \
  --dory-onnx export/hybrid_follow/hybrid_follow_dory.onnx
```

## Practical Guidance For Future Models

The checker is meant to be useful beyond `hybrid_follow`.

If you build more complex models, the lowest-risk design choices are still:

- keep merges explicit with modules instead of plain `x + y`
- keep activations explicit with modules instead of `F.relu(...)`
- keep skip paths simple
- keep outputs explicitly named
- treat extra branches, attention blocks, dynamic control flow, and unusual ONNX ops as compatibility review points

Using `Sequential` is fine. It just makes the graph harder to inspect when something else goes wrong.

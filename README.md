# pytorch_ssd

This folder is the model-development and export side of the project.
It trains the detector, exports a quantized ONNX with NEMO, cleans the
graph for DORY, and can emit GAP8 application artifacts that are later
used by `../crazyflie_ssd`.

## Current Model

- Task: person-only object detection
- Detector family: torchvision SSD
- Backbone: MobileNetV2
- Width multiplier: `0.1`
- Classes: `2` total (`background`, `person`)
- Main training model: `SSDMobileNetV2Raw`
- Typical deployment input: grayscale `1 x 128 x 128`

If you are trying to redesign the model, the important point is that the
current stack is not just "train a PyTorch model." It is a tightly coupled
train -> quantize -> ONNX cleanup -> DORY -> GAP8 deployment pipeline.

## Folder Map

- `train.py`
  Training loop. Builds `SSDMobileNetV2Raw`, loads the COCO person dataset,
  and writes checkpoints such as `training/person_ssd_pytorch/ssd_mbv2_epoch_030.pth`.
- `models/ssd_mobilenet_v2.py`
  Defines the MobileNetV2 backbone and a plain SSD constructor.
- `models/ssd_mobilenet_v2_raw.py`
  SSD wrapper used for training and export. It supports both:
  - normal torchvision SSD training/inference on `list[Tensor]`
  - raw head export on a single tensor input
- `utils/coco_person.py`
  Person-only COCO dataset loader.
- `utils/transforms.py`
  Grayscale-centered transforms. Can output either 1 or 3 channels.
- `export_nemo_quant.py`
  Loads a checkpoint, handles compatibility remapping, runs NEMO export,
  and writes the ONNX used by later stages.
- `run_all.sh`
  End-to-end export script. Runs NEMO export, onnxsim, custom ONNX cleanup,
  DORY config generation, artifact generation, and `network_generate.py`.
- `export/`
  ONNX files, stripped graphs, DORY configs, manifests, weight text dumps,
  and other export/debug artifacts.
- `training/person_ssd_pytorch/`
  Saved checkpoints and a small demo output image.
- `application/`
  A generated GAP8 application snapshot from this pipeline. Useful as a
  historical artifact, but the more active runtime wrapper now lives in
  `../crazyflie_ssd`.

## How The Current Pipeline Works

1. `train.py` trains `SSDMobileNetV2Raw` on person-only COCO annotations.
2. `export_nemo_quant.py` loads a checkpoint and exports a NEMO-quantized ONNX.
3. `run_all.sh` simplifies and strips unsupported ONNX ops for DORY.
4. `run_all.sh` can generate DORY output into a target app directory. In this
   project that is often `../crazyflie_ssd/generated`.
5. `../crazyflie_ssd` wraps the generated network with camera capture,
   preprocessing, runtime buffer management, and transport/debug code.

## Important Current Quirks

- There is model-format drift across the project history.
  Some checkpoints use `backbone.features.*` keys while the current export
  model uses explicit `backbone.stage*` modules. `export_nemo_quant.py`
  contains key-remapping logic to bridge that gap.
- There is also channel-history drift.
  Export logs show older checkpoints can carry a 3-channel first conv and are
  adapted to 1-channel grayscale during export when needed.
- Training and deployment shapes are not perfectly unified.
  `train.py` defaults to `160 x 160`, while deployment/export is often run at
  `128 x 128`.
- `inference_demo.py` should be treated as a convenience script, not as the
  authoritative deployment contract. It still reflects older assumptions in
  places, including grayscale-to-3-channel replication.
- DORY currently expects the final ONNX to be at the `ID` stage and then
  requires several cleanup passes (`strip_affine_mul_add.py`,
  `strip_transpose.py`, `strip_min.py`, `strip_fake_quant.py`) before codegen.

## What Matters Most For A Redesign

Decide the deployment contract first, not last:

- input size
- input channel count
- output tensor layout
- whether postprocess happens on GAP8 or off-device
- memory budget on GAP8

After that, the files that usually need to move together are:

- `train.py`
- `models/`
- `utils/transforms.py`
- `export_nemo_quant.py`
- `run_all.sh`
- `export/` cleanup scripts and configs
- `../crazyflie_ssd/app_config.h`
- `../crazyflie_ssd/preprocess.c`
- `../crazyflie_ssd/ssd_postprocess.c`
- regenerated artifacts under `../crazyflie_ssd/generated`

## Practical Warning

The hardest part of changing this model is usually not PyTorch training.
The hard parts are:

- staying compatible with NEMO export
- staying compatible with DORY graph restrictions
- fitting GAP8 memory
- matching the runtime assumptions in `../crazyflie_ssd`

Treat this folder and `../crazyflie_ssd` as one system.

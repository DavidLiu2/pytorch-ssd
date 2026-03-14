#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from export_nemo_quant import build_model, load_checkpoint  # noqa: E402


FINAL_DUMP_RE = re.compile(r"^FINAL_TENSOR_I32\s+(\w+)(.*)$")
FINAL_BEGIN_RE = re.compile(r"^FINAL_TENSOR_I32_BEGIN\s+(\w+)\s+count=(\d+)$")
FINAL_END_RE = re.compile(r"^FINAL_TENSOR_I32_END\s+(\w+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GAP8 final SSD head dumps against ONNXRuntime outputs.")
    parser.add_argument("--onnx", required=True, help="Path to the DORY ONNX model.")
    parser.add_argument("--input-txt", required=True, help="Path to input.txt used for deployment.")
    parser.add_argument("--gvsoc-log", required=True, help="Log containing FINAL_TENSOR_I32_* markers.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint used for the export.")
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_input_values(path: Path) -> np.ndarray:
  values: List[int] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    values.append(int(stripped.rstrip(",")))
  return np.asarray(values, dtype=np.uint8)


def get_model_input_shape(model: onnx.ModelProto) -> List[int]:
  initializer_names = {init.name for init in model.graph.initializer}
  for inp in model.graph.input:
    if inp.name in initializer_names:
      continue
    dims = []
    for dim in inp.type.tensor_type.shape.dim:
      if dim.HasField("dim_value"):
        dims.append(int(dim.dim_value))
      else:
        dims.append(1)
    return dims
  raise RuntimeError("No non-initializer ONNX input found.")


def run_onnx(onnx_path: Path, input_values: np.ndarray) -> Dict[str, np.ndarray]:
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  input_meta = sess.get_inputs()[0]
  model_input = input_values.reshape(input_meta.shape).astype(np.float32)
  output_names = [out.name for out in sess.get_outputs()]
  outputs = sess.run(output_names, {input_meta.name: model_input})
  return {name: np.asarray(value) for name, value in zip(output_names, outputs)}


def parse_gap8_dump(log_path: Path) -> Dict[str, np.ndarray]:
  values: Dict[str, List[int]] = {}
  expected_counts: Dict[str, int] = {}
  active_labels: Dict[str, bool] = {}

  for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    begin_match = FINAL_BEGIN_RE.match(line.strip())
    if begin_match:
      label = begin_match.group(1)
      expected_counts[label] = int(begin_match.group(2))
      values[label] = []
      active_labels[label] = True
      continue

    end_match = FINAL_END_RE.match(line.strip())
    if end_match:
      label = end_match.group(1)
      active_labels[label] = False
      continue

    dump_match = FINAL_DUMP_RE.match(line.strip())
    if dump_match:
      label = dump_match.group(1)
      if not active_labels.get(label, False):
        continue
      payload = dump_match.group(2).strip()
      if payload:
        values[label].extend(int(part) for part in payload.split())

  arrays: Dict[str, np.ndarray] = {}
  for label, arr_values in values.items():
    expected = expected_counts.get(label)
    if expected is None:
      continue
    if len(arr_values) != expected:
      raise RuntimeError(
        f"Tensor '{label}' dump count mismatch: expected {expected}, got {len(arr_values)}"
      )
    arrays[label] = np.asarray(arr_values, dtype=np.int32)
  if not arrays:
    raise RuntimeError("No FINAL_TENSOR_I32 dumps found in the provided log.")
  return arrays


def summary_i32(name: str, values: np.ndarray) -> str:
  return (
    f"{name}: count={values.size} sum={int(values.astype(np.int64).sum())} "
    f"min={int(values.min())} max={int(values.max())} first8={values[:8].tolist()}"
  )


def first_bytes(values: np.ndarray, count: int) -> List[int]:
  return values.astype("<i4", copy=False).view(np.uint8)[:count].astype(np.uint8).tolist()


def compare_raw(name: str, onnx_values: np.ndarray, gap_values: np.ndarray) -> None:
  if onnx_values.shape != gap_values.shape:
    print(f"{name} shape mismatch: onnx={onnx_values.shape} gap8={gap_values.shape}")
    return

  diff = gap_values.astype(np.int64) - onnx_values.astype(np.int64)
  abs_diff = np.abs(diff)
  mismatch_count = int(np.count_nonzero(diff))
  print(summary_i32(f"{name} onnx", onnx_values))
  print(summary_i32(f"{name} gap8", gap_values))
  print(
    f"{name} diff: mismatches={mismatch_count} "
    f"max_abs_diff={int(abs_diff.max())} mean_abs_diff={float(abs_diff.mean()):.3f}"
  )
  print(f"{name} onnx first16-bytes={first_bytes(onnx_values, 16)}")
  print(f"{name} gap8 first16-bytes={first_bytes(gap_values, 16)}")
  if mismatch_count:
    mismatch_indices = np.flatnonzero(diff)[:8]
    print(
      f"{name} first_mismatches="
      f"{[(int(idx), int(onnx_values[idx]), int(gap_values[idx])) for idx in mismatch_indices]}"
    )


def run_fp_model(
  ckpt_path: Path,
  input_values: np.ndarray,
  height: int,
  width: int,
  input_channels: int,
  num_classes: int,
  width_mult: float,
) -> Dict[str, np.ndarray]:
  model = build_model(
    num_classes=num_classes,
    width_mult=width_mult,
    image_size=(height, width),
    input_channels=input_channels,
  )
  model = load_checkpoint(model, str(ckpt_path), torch.device("cpu"))
  model.eval()

  x_real = (
    torch.from_numpy(input_values.reshape(1, input_channels, height, width).astype(np.float32))
    .div(255.0)
  )
  with torch.no_grad():
    locs, cls_logits = model.forward_raw(x_real)
  return {
    "locs": locs.detach().cpu().numpy(),
    "cls": cls_logits.detach().cpu().numpy(),
  }


def estimate_scale(raw_values: np.ndarray, fp_values: np.ndarray, label: str) -> float:
  raw_flat = raw_values.reshape(-1).astype(np.float64)
  fp_flat = fp_values.reshape(-1).astype(np.float64)
  threshold = np.percentile(np.abs(fp_flat), 30.0)
  mask = np.abs(fp_flat) > max(threshold, 1e-8)
  if int(mask.sum()) < 512:
    mask = np.abs(fp_flat) > 1e-8
  if int(mask.sum()) == 0:
    raise RuntimeError(f"Cannot estimate {label} scale: FP reference is all zeros.")
  numerator = float(np.dot(raw_flat[mask], fp_flat[mask]))
  denominator = float(np.dot(fp_flat[mask], fp_flat[mask]))
  if denominator == 0.0:
    raise RuntimeError(f"Cannot estimate {label} scale: degenerate denominator.")
  scale = numerator / denominator
  if scale == 0.0 or not np.isfinite(scale):
    raise RuntimeError(f"Invalid {label} scale estimate: {scale}")
  corr = float(np.corrcoef(raw_flat[mask], fp_flat[mask])[0, 1])
  print(f"{label} inferred_scale={scale:.6f} corr={corr:.6f} samples={int(mask.sum())}")
  return scale


def generate_anchors(model, height: int, width: int, input_channels: int) -> torch.Tensor:
  x = torch.zeros(1, input_channels, height, width, dtype=torch.float32)
  with torch.no_grad():
    features = model.backbone.forward_features(x)
  images = ImageList(x, [(height, width)])
  anchors = model.anchor_generator(images, features)
  return anchors[0]


def decode_topk(
  model,
  anchors: torch.Tensor,
  loc_values: np.ndarray,
  cls_values: np.ndarray,
  height: int,
  width: int,
  top_k: int,
) -> List[dict]:
  loc_tensor = torch.from_numpy(loc_values.astype(np.float32))[0]
  cls_tensor = torch.from_numpy(cls_values.astype(np.float32))[0]
  boxes = model.box_coder.decode_single(loc_tensor, anchors)
  boxes = box_ops.clip_boxes_to_image(boxes, (height, width))
  probs = torch.softmax(cls_tensor, dim=-1)
  scores, labels = probs[:, 1:].max(dim=1)
  labels = labels + 1

  keep = scores > 0.05
  boxes = boxes[keep]
  scores = scores[keep]
  labels = labels[keep]
  if boxes.numel() == 0:
    return []

  keep_idx = torchvision.ops.nms(boxes, scores, 0.45)
  boxes = boxes[keep_idx]
  scores = scores[keep_idx]
  labels = labels[keep_idx]

  order = torch.argsort(scores, descending=True)[:top_k]
  boxes = boxes[order]
  scores = scores[order]
  labels = labels[order]

  detections = []
  for box, score, label in zip(boxes, scores, labels):
    detections.append(
      {
        "label": int(label),
        "score": float(score),
        "box": [float(v) for v in box.tolist()],
      }
    )
  return detections


def compare_detections(name: str, onnx_det: List[dict], gap_det: List[dict]) -> None:
  print(f"{name} onnx_topk={onnx_det}")
  print(f"{name} gap8_topk={gap_det}")
  shared = min(len(onnx_det), len(gap_det))
  for idx in range(shared):
    onnx_item = onnx_det[idx]
    gap_item = gap_det[idx]
    box_delta = [
      round(gap_item["box"][i] - onnx_item["box"][i], 4)
      for i in range(4)
    ]
    print(
      f"{name} rank={idx} "
      f"label_pair=({onnx_item['label']},{gap_item['label']}) "
      f"score_delta={gap_item['score'] - onnx_item['score']:.6f} "
      f"box_delta={box_delta}"
    )


def main() -> None:
  args = parse_args()

  onnx_path = Path(args.onnx).resolve()
  input_path = Path(args.input_txt).resolve()
  log_path = Path(args.gvsoc_log).resolve()
  ckpt_path = Path(args.ckpt).resolve()

  model = onnx.load(str(onnx_path))
  input_shape = get_model_input_shape(model)
  input_values = load_input_values(input_path)
  expected_input_count = int(np.prod(input_shape))
  if input_values.size != expected_input_count:
    raise RuntimeError(
      f"input.txt size mismatch: expected {expected_input_count} values for shape {input_shape}, "
      f"got {input_values.size}"
    )

  onnx_outputs = run_onnx(onnx_path, input_values)
  onnx_bbox = np.rint(onnx_outputs["958"]).astype(np.int32).reshape(-1)
  onnx_cls = np.rint(onnx_outputs["1047"]).astype(np.int32).reshape(-1)

  gap8_outputs = parse_gap8_dump(log_path)
  gap_bbox = gap8_outputs["bbox"]
  gap_cls = gap8_outputs["cls"]

  compare_raw("bbox", onnx_bbox, gap_bbox)
  compare_raw("cls", onnx_cls, gap_cls)

  fp_outputs = run_fp_model(
    ckpt_path=ckpt_path,
    input_values=input_values,
    height=args.height,
    width=args.width,
    input_channels=args.input_channels,
    num_classes=args.num_classes,
    width_mult=args.width_mult,
  )
  loc_scale = estimate_scale(onnx_bbox.reshape(fp_outputs["locs"].shape), fp_outputs["locs"], "bbox")
  cls_scale = estimate_scale(onnx_cls.reshape(fp_outputs["cls"].shape), fp_outputs["cls"], "cls")

  onnx_bbox_deq = onnx_bbox.reshape(fp_outputs["locs"].shape).astype(np.float32) / loc_scale
  gap_bbox_deq = gap_bbox.reshape(fp_outputs["locs"].shape).astype(np.float32) / loc_scale
  onnx_cls_deq = onnx_cls.reshape(fp_outputs["cls"].shape).astype(np.float32) / cls_scale
  gap_cls_deq = gap_cls.reshape(fp_outputs["cls"].shape).astype(np.float32) / cls_scale

  fp_model = build_model(
    num_classes=args.num_classes,
    width_mult=args.width_mult,
    image_size=(args.height, args.width),
    input_channels=args.input_channels,
  )
  fp_model = load_checkpoint(fp_model, str(ckpt_path), torch.device("cpu"))
  fp_model.eval()
  anchors = generate_anchors(fp_model, args.height, args.width, args.input_channels)

  onnx_det = decode_topk(fp_model, anchors, onnx_bbox_deq, onnx_cls_deq, args.height, args.width, args.top_k)
  gap_det = decode_topk(fp_model, anchors, gap_bbox_deq, gap_cls_deq, args.height, args.width, args.top_k)
  compare_detections("detections", onnx_det, gap_det)


if __name__ == "__main__":
  main()

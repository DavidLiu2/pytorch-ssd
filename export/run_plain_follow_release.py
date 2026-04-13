#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_WINDOWS_PYTHON = Path("/mnt/c/Python313/python.exe")
DEFAULT_CKPT = PROJECT_DIR / "training" / "plain_follow" / "plain_follow_best_follow_score.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "plain_follow_prod"
DEFAULT_VAL_IMAGE_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_REP16_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "1_real_image_validation"
    / "input_sets"
    / "representative16_20260324"
)
DEFAULT_HARD_CASE_DIR = (
    PROJECT_DIR
    / "logs"
    / "hybrid_follow_val"
    / "5_microblock_add_only_patch"
    / "input_sets"
    / "hard_case_subset"
)
DEFAULT_THRESHOLD_CANDIDATES = "0.45,0.50,0.55,0.60"
DEFAULT_DORY_ROOT = REPO_DIR / "dory"
DEFAULT_DORYENV_DIR = REPO_DIR / "doryenv"
DEFAULT_DORY_CONFIG_TEMPLATE = PROJECT_DIR / "export" / "plain_follow" / "config_plain_follow_runtime.json"
DEFAULT_DORY_SEMANTIC_HELPER = PROJECT_DIR / "export" / "dory_semantic_follow_inference.py"
DEFAULT_DORY_IO_HELPER = PROJECT_DIR / "export" / "generate_dory_io_artifacts.py"
DEFAULT_GAP8_LAYER_MANIFEST_BUILDER = PROJECT_DIR / "export" / "build_gap8_layer_manifest.py"
DEFAULT_GAP8_LAYER_COMPARE_SCRIPT = PROJECT_DIR / "export" / "compare_gap8_layer_bytes.py"
DEFAULT_GAP8_BN_QUANT_PATCH_SCRIPT = PROJECT_DIR / "tools" / "patch_gap8_bn_quant_int64.py"
DEFAULT_GVSOC_SCRIPT = PROJECT_DIR / "tools" / "run_aideck_val_impl.sh"
DEFAULT_GVSOC_COMPARE_SCRIPT = PROJECT_DIR / "export" / "archive" / "compare_gap8_final_tensor.py"
DEFAULT_VALIDATION_MAIN = PROJECT_DIR / "aideck_val_main_plain_follow.c"
DEFAULT_GVSOC_PLATFORM = "gvsoc"
DEFAULT_GVSOC_LABEL = "final"
DEFAULT_GVSOC_TRACE_LAYER_OUTPUT_BYTES_PER_LINE = 64
DEFAULT_IMAGE_SIZE = (128, 128)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DORY_SEMANTIC_PARITY_CAVEAT = (
    "Deployment validation now uses a Python DORY-graph simulator built from the parsed DORY graph. "
    "That matches parser/codegen integerization much more closely than ONNXRuntime on cleaned "
    "model_id_dory.onnx, but GVSOC remains the final runtime gate."
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

from evaluate_quant_native_follow import (  # noqa: E402
    AnnotationIndex as QuantAnnotationIndex,
    build_eval_samples,
    stage_metrics_for_subset,
    tensor_rows,
)
from export_nemo_quant_core import semantic_output  # noqa: E402
from hybrid_follow_image_artifacts import stage_image_artifacts  # noqa: E402
from utils.follow_task import follow_output_metadata, follow_runtime_decode_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical plain_follow production wrapper: build the calibration manifest, "
            "materialize a larger validation pack, run float overlays, run quant export/eval, "
            "sweep a deployment visibility threshold on DORY-semantic predictions built from "
            "cleaned model_id_dory.onnx, render pre/post compare overlays against that "
            "deployment artifact, and smoke-test the exported app in GVSOC."
        )
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-image-dir", default=str(DEFAULT_VAL_IMAGE_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument(
        "--calib-target-count",
        type=int,
        default=128,
        help="Number of calibration images selected from COCO val.",
    )
    parser.add_argument(
        "--calib-max-images",
        type=int,
        default=0,
        help="Optional max-images cap passed to the calibration manifest builder.",
    )
    parser.add_argument(
        "--expanded-pack-extra-count",
        type=int,
        default=48,
        help="Additional COCO val images to add on top of the rep16 set for the expanded validation pack.",
    )
    parser.add_argument(
        "--expanded-pack-max-images",
        type=int,
        default=0,
        help="Optional max-images cap passed to the validation-pack ranking manifest builder.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["auto", "copy", "hardlink"],
        default="auto",
        help="How to materialize the local input-set copies under the output directory.",
    )
    parser.add_argument(
        "--vis-thresh",
        type=float,
        default=None,
        help="Checkpoint-side visibility threshold used for float validation and the initial quant export summary.",
    )
    parser.add_argument(
        "--threshold-candidates",
        default=DEFAULT_THRESHOLD_CANDIDATES,
        help=(
            "Comma-separated deployment thresholds to evaluate on the DORY-semantic deployment path. "
            "Default: 0.45,0.50,0.55,0.60"
        ),
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used for the wrapped export/validation scripts.",
    )
    parser.add_argument(
        "--dory-python",
        default=None,
        help="Python interpreter used for DORY network_generate.",
    )
    parser.add_argument("--dory-root", default=str(DEFAULT_DORY_ROOT))
    parser.add_argument("--dory-semantic-helper", default=str(DEFAULT_DORY_SEMANTIC_HELPER))
    parser.add_argument("--dory-io-helper", default=str(DEFAULT_DORY_IO_HELPER))
    parser.add_argument("--dory-config-template", default=str(DEFAULT_DORY_CONFIG_TEMPLATE))
    parser.add_argument("--gap8-layer-manifest-builder", default=str(DEFAULT_GAP8_LAYER_MANIFEST_BUILDER))
    parser.add_argument("--gap8-layer-compare-script", default=str(DEFAULT_GAP8_LAYER_COMPARE_SCRIPT))
    parser.add_argument("--gap8-bn-quant-patch-script", default=str(DEFAULT_GAP8_BN_QUANT_PATCH_SCRIPT))
    parser.add_argument("--gvsoc-script", default=str(DEFAULT_GVSOC_SCRIPT))
    parser.add_argument("--gvsoc-compare-script", default=str(DEFAULT_GVSOC_COMPARE_SCRIPT))
    parser.add_argument("--validation-main", default=str(DEFAULT_VALIDATION_MAIN))
    parser.add_argument(
        "--gvsoc-trace-layer-output-bytes-per-line",
        type=int,
        default=DEFAULT_GVSOC_TRACE_LAYER_OUTPUT_BYTES_PER_LINE,
    )
    parser.add_argument(
        "--gvsoc-image",
        default=None,
        help="Optional image path to use for the GVSOC smoke. Defaults to the first visible rep16 example.",
    )
    parser.add_argument("--skip-gvsoc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_python(user_value: str | None) -> str:
    if user_value:
        return user_value
    if sys.executable:
        return sys.executable
    if os.name == "nt" and DEFAULT_WINDOWS_PYTHON.is_file():
        return str(DEFAULT_WINDOWS_PYTHON)
    return "python3"


def resolve_dory_python(user_value: str | None) -> str:
    if user_value:
        return user_value
    candidates = (
        DEFAULT_DORYENV_DIR / "bin" / "python3",
        DEFAULT_DORYENV_DIR / "bin" / "python",
        DEFAULT_DORYENV_DIR / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return resolve_python(None)


def parse_threshold_candidates(raw_value: str) -> list[float]:
    values: list[float] = []
    for token in str(raw_value).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("No threshold candidates were provided.")
    return sorted({float(value) for value in values})


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path = PROJECT_DIR,
    extra_env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {format_command(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: see {log_path}")


def discover_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is not a dict: {path}")
    return payload


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint_payload(ckpt_path)
    model_type = str(payload.get("model_type") or "")
    if model_type != "plain_follow":
        raise ValueError(
            f"run_plain_follow_release.py only supports plain_follow checkpoints, got {model_type!r}."
        )

    head_type = str(payload.get("follow_head_type") or "xbin9_size_bucket4")
    checkpoint_vis_thresh = float(((payload.get("val_stats") or {}).get("selected_vis_threshold")) or 0.5)
    quant_eval_vis_thresh = float(args.vis_thresh if args.vis_thresh is not None else checkpoint_vis_thresh)
    image_size = (
        int(payload.get("height") or DEFAULT_IMAGE_SIZE[0]),
        int(payload.get("width") or DEFAULT_IMAGE_SIZE[1]),
    )
    output_metadata = follow_output_metadata(model_type=model_type, head_type=head_type)

    return {
        "ckpt_path": ckpt_path,
        "payload": payload,
        "model_type": model_type,
        "follow_head_type": head_type,
        "checkpoint_vis_thresh": checkpoint_vis_thresh,
        "quant_eval_vis_thresh": quant_eval_vis_thresh,
        "threshold_candidates": parse_threshold_candidates(args.threshold_candidates),
        "python": resolve_python(args.python),
        "dory_python": resolve_dory_python(args.dory_python),
        "output_dir": Path(args.output_dir).expanduser().resolve(),
        "val_image_dir": Path(args.val_image_dir).expanduser().resolve(),
        "annotations": Path(args.annotations).expanduser().resolve(),
        "rep16_dir": Path(args.rep16_dir).expanduser().resolve(),
        "hard_case_dir": Path(args.hard_case_dir).expanduser().resolve(),
        "dory_root": Path(args.dory_root).expanduser().resolve(),
        "dory_semantic_helper": Path(args.dory_semantic_helper).expanduser().resolve(),
        "dory_io_helper": Path(args.dory_io_helper).expanduser().resolve(),
        "dory_config_template": Path(args.dory_config_template).expanduser().resolve(),
        "gap8_layer_manifest_builder": Path(args.gap8_layer_manifest_builder).expanduser().resolve(),
        "gap8_layer_compare_script": Path(args.gap8_layer_compare_script).expanduser().resolve(),
        "gap8_bn_quant_patch_script": Path(args.gap8_bn_quant_patch_script).expanduser().resolve(),
        "gvsoc_script": Path(args.gvsoc_script).expanduser().resolve(),
        "gvsoc_compare_script": Path(args.gvsoc_compare_script).expanduser().resolve(),
        "gvsoc_trace_layer_output_bytes_per_line": int(args.gvsoc_trace_layer_output_bytes_per_line),
        "validation_main": Path(args.validation_main).expanduser().resolve(),
        "gvsoc_image": (Path(args.gvsoc_image).expanduser().resolve() if args.gvsoc_image else None),
        "image_size": image_size,
        "output_metadata": output_metadata,
    }


def materialize_file(src: Path, dst: Path, *, copy_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_input_set(
    source_paths: list[Path],
    output_dir: Path,
    *,
    copy_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    link_modes: list[str] = []
    for index, src in enumerate(source_paths, start=1):
        dst = output_dir / src.name
        link_mode = materialize_file(src, dst, copy_mode=copy_mode)
        link_modes.append(link_mode)
        rows.append(
            {
                "index": index,
                "source_path": str(src),
                "output_path": str(dst),
                "image_name": src.name,
                "materialization": link_mode,
            }
        )
    materialization_mode = "mixed" if len(set(link_modes)) > 1 else (link_modes[0] if link_modes else None)
    summary = {
        "output_dir": str(output_dir),
        "image_count": len(rows),
        "materialization_mode": materialization_mode,
        "rows": rows,
    }
    write_json(output_dir / "input_set_summary.json", summary)
    return summary


def build_expanded_pack(
    *,
    rep16_dir: Path,
    hard_case_dir: Path,
    manifest_payload: dict[str, Any],
    extra_count: int,
    output_root: Path,
    copy_mode: str,
) -> dict[str, Any]:
    rep16_paths = discover_images(rep16_dir)
    hard_case_paths = discover_images(hard_case_dir)
    if not rep16_paths:
        raise FileNotFoundError(f"No images found in rep16 dir: {rep16_dir}")
    if not hard_case_paths:
        raise FileNotFoundError(f"No images found in hard-case dir: {hard_case_dir}")

    used_names: set[str] = set()
    expanded_sources: list[Path] = []
    extra_sources: list[Path] = []

    def add_once(path: Path, *, count_as_extra: bool) -> None:
        if path.name in used_names:
            return
        used_names.add(path.name)
        expanded_sources.append(path)
        if count_as_extra:
            extra_sources.append(path)

    for path in rep16_paths:
        add_once(path, count_as_extra=False)
    for path in hard_case_paths:
        add_once(path, count_as_extra=False)

    for row in manifest_payload.get("ordered_samples") or []:
        if len(extra_sources) >= int(extra_count):
            break
        source_path = Path(str(row.get("source_path") or "")).expanduser().resolve()
        if not source_path.is_file():
            continue
        add_once(source_path, count_as_extra=True)

    input_sets_dir = output_root / "input_sets"
    rep16_local = materialize_input_set(rep16_paths, input_sets_dir / "rep16", copy_mode=copy_mode)
    hard_case_local = materialize_input_set(hard_case_paths, input_sets_dir / "hard_case_subset", copy_mode=copy_mode)
    expanded_local = materialize_input_set(expanded_sources, input_sets_dir / "expanded_eval", copy_mode=copy_mode)

    summary = {
        "rep16": rep16_local,
        "hard_case": hard_case_local,
        "expanded_eval": expanded_local,
        "rep16_source_dir": str(rep16_dir),
        "hard_case_source_dir": str(hard_case_dir),
        "expanded_extra_count": len(extra_sources),
        "expanded_extra_sources": [str(path) for path in extra_sources],
    }
    write_json(output_root / "expanded_eval_pack_summary.json", summary)
    return summary


def threshold_selection_score(metrics: dict[str, Any]) -> float:
    f1 = float(metrics.get("f1") or 0.0)
    precision = float(metrics.get("precision") or 0.0)
    no_person_fp_rate = float(metrics.get("no_person_fp_rate") or 0.0)
    return f1 + (0.25 * precision) - (0.75 * no_person_fp_rate)


def build_onnx_dataset_view(
    *,
    image_dir: Path,
    annotations: QuantAnnotationIndex,
    image_size: tuple[int, int],
    model_type: str,
    session: Any,
    onnx_input_name: str,
    onnx_output_name: str,
) -> dict[str, Any]:
    samples = build_eval_samples(
        image_dir,
        annotations,
        image_size=image_size,
        model_type=model_type,
    )
    if not samples:
        raise FileNotFoundError(f"No evaluation images found under {image_dir}")

    output_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    no_person_rows: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        raw_onnx = np.asarray(
            session.run(
                [onnx_output_name],
                {onnx_input_name: sample["staged_input"].detach().cpu().numpy()},
            )[0],
            dtype=np.float32,
        )
        output_rows.append(np.asarray(semantic_output(raw_onnx, "id"), dtype=np.float32).reshape(-1))
        target_rows.append(sample["follow_target"].detach().cpu().numpy().reshape(-1))
        no_person_rows.append(sample["true_no_person"].detach().cpu().numpy().reshape(-1))
        sample_rows.append(
            {
                "image_name": str(sample["image_name"]),
                "image_path": str(sample["image_path"]),
            }
        )

    return {
        "image_dir": str(image_dir),
        "image_count": len(sample_rows),
        "rows": sample_rows,
        "outputs": tensor_rows(output_rows),
        "targets": torch.tensor(np.asarray(target_rows, dtype=np.float32)),
        "no_person": torch.tensor(np.asarray(no_person_rows, dtype=np.int64)).view(-1, 1),
    }


def write_dory_semantic_input_bundle(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    model_type: str,
) -> None:
    payload = {
        # Deployment validation should stage uint8 inputs from the source image
        # using the same preprocess contract as the seeded DORY app path.
        "staging_mode": "runtime_preprocess_uint8",
        "model_type": str(model_type),
        "samples": [
            {
                "image_name": str(sample["image_name"]),
                "image_path": str(sample["image_path"]),
                # Keep the legacy staged_input around as a debug artifact and
                # fallback for older consumers, but prefer image-path restaging.
                "staged_input": [
                    int(value)
                    for value in sample["staged_input"].detach().cpu().numpy().reshape(-1).tolist()
                ],
            }
            for sample in samples
        ]
    }
    write_json(output_path, payload)


def run_dory_semantic_inference(
    *,
    context: dict[str, Any],
    dory_onnx_path: Path,
    bundle_path: Path,
    output_json: Path,
    log_path: Path,
) -> dict[str, Any]:
    helper_path = Path(context["dory_semantic_helper"])
    if not helper_path.is_file():
        raise FileNotFoundError(f"DORY semantic helper not found: {helper_path}")
    config_path = output_json.parent / "config_plain_follow_runtime.json"
    config_payload = read_json(Path(context["dory_config_template"]))
    config_payload["onnx_file"] = str(dory_onnx_path)
    write_json(config_path, config_payload)
    command = [
        str(context["dory_python"]),
        str(helper_path),
        "--onnx",
        str(dory_onnx_path),
        "--config",
        str(config_path),
        "--input-bundle",
        str(bundle_path),
        "--output-json",
        str(output_json),
    ]
    run_logged(command, log_path=log_path, cwd=PROJECT_DIR)
    return read_json(output_json)


def build_dory_semantic_dataset_view(
    *,
    context: dict[str, Any],
    image_dir: Path,
    annotations: QuantAnnotationIndex,
    image_size: tuple[int, int],
    model_type: str,
    dory_onnx_path: Path,
    work_dir: Path,
    command_log: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = build_eval_samples(
        image_dir,
        annotations,
        image_size=image_size,
        model_type=model_type,
    )
    if not samples:
        raise FileNotFoundError(f"No evaluation images found under {image_dir}")

    bundle_path = work_dir / "input_bundle.json"
    output_json = work_dir / "predictions.json"
    write_dory_semantic_input_bundle(samples, bundle_path, model_type=model_type)
    predictions = run_dory_semantic_inference(
        context=context,
        dory_onnx_path=dory_onnx_path,
        bundle_path=bundle_path,
        output_json=output_json,
        log_path=command_log,
    )

    sample_lookup = {
        (str(sample["image_name"]), str(sample["image_path"])): sample
        for sample in samples
    }
    output_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    no_person_rows: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    for row in predictions.get("samples") or []:
        key = (str(row.get("image_name") or ""), str(row.get("image_path") or ""))
        sample = sample_lookup.get(key)
        if sample is None:
            raise KeyError(f"Prediction row did not match a staged sample: {key}")
        raw_output = np.asarray(row.get("raw_output") or [], dtype=np.float32).reshape(1, -1)
        output_rows.append(np.asarray(semantic_output(raw_output, "id"), dtype=np.float32).reshape(-1))
        target_rows.append(sample["follow_target"].detach().cpu().numpy().reshape(-1))
        no_person_rows.append(sample["true_no_person"].detach().cpu().numpy().reshape(-1))
        sample_rows.append(
            {
                "image_name": str(sample["image_name"]),
                "image_path": str(sample["image_path"]),
            }
        )

    return (
        {
            "image_dir": str(image_dir),
            "image_count": len(sample_rows),
            "rows": sample_rows,
            "outputs": tensor_rows(output_rows),
            "targets": torch.tensor(np.asarray(target_rows, dtype=np.float32)),
            "no_person": torch.tensor(np.asarray(no_person_rows, dtype=np.int64)).view(-1, 1),
        },
        predictions,
    )


def build_prediction_row_lookup(predictions: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions.get("samples") or []:
        image_name = str(row.get("image_name") or "")
        image_path = str(row.get("image_path") or "")
        lookup[(image_name, image_path)] = row
    return lookup


def find_prediction_row_for_image(predictions: dict[str, Any], image_path: Path) -> dict[str, Any]:
    lookup = build_prediction_row_lookup(predictions)
    candidates = (
        (image_path.name, str(image_path)),
        (image_path.name, image_path.as_posix()),
    )
    for key in candidates:
        row = lookup.get(key)
        if row is not None:
            return row
    raise KeyError(f"Missing DORY-semantic prediction for {image_path}")


def prepare_dory_io_seed(
    *,
    context: dict[str, Any],
    image_path: Path,
    dory_onnx_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    helper_path = Path(context["dory_io_helper"])
    if not helper_path.is_file():
        raise FileNotFoundError(f"DORY IO helper not found: {helper_path}")
    manifest_path = output_dir / "nemo_dory_artifacts.json"
    weights_dir = output_dir / "weights_txt"
    command = [
        str(context["dory_python"]),
        str(helper_path),
        "--onnx",
        str(dory_onnx_path),
        "--config",
        str(context["dory_config_template"]),
        "--image",
        str(image_path),
        "--model-type",
        str(context["model_type"]),
        "--runtime-source",
        "dory_hw_graph",
        "--io-dir",
        str(output_dir),
        "--weights-dir",
        str(weights_dir),
        "--manifest",
        str(manifest_path),
    ]
    run_logged(command, log_path=log_path, cwd=PROJECT_DIR)
    manifest = read_json(manifest_path)
    runtime_source = str(manifest.get("runtime_source") or "dory_hw_graph")
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "input_txt": output_dir / "input.txt",
        "output_txt": output_dir / "output.txt",
        "golden_activations": [Path(path) for path in (manifest.get("golden_activations") or [])],
        "reference_source": f"{runtime_source}({dory_onnx_path.name}) via generate_dory_io_artifacts.py",
    }


def sync_dory_io_seed_into_model_dir(seed_dir: Path, model_dir: Path) -> list[str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    stale_patterns = ("input.txt", "output.txt", "out_layer*.txt")
    for pattern in stale_patterns:
        for candidate in model_dir.glob(pattern):
            candidate.unlink()

    copied: list[str] = []
    for pattern in stale_patterns:
        for src in sorted(seed_dir.glob(pattern)):
            dst = model_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def build_gap8_layer_manifest(
    *,
    context: dict[str, Any],
    dory_manifest_path: Path,
    network_header_path: Path,
    output_path: Path,
    log_path: Path,
) -> Path:
    builder_path = Path(context["gap8_layer_manifest_builder"])
    if not builder_path.is_file():
        raise FileNotFoundError(f"GAP8 layer manifest builder not found: {builder_path}")
    command = [
        str(context["python"]),
        str(builder_path),
        "--dory-manifest",
        str(dory_manifest_path),
        "--network-header",
        str(network_header_path),
        "--output-json",
        str(output_path),
    ]
    run_logged(command, log_path=log_path, cwd=PROJECT_DIR)
    return output_path


def patch_gap8_bn_quant_helpers(
    *,
    context: dict[str, Any],
    app_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    patch_script = Path(context["gap8_bn_quant_patch_script"])
    if not patch_script.is_file():
        raise FileNotFoundError(f"GAP8 BN quant patch script not found: {patch_script}")
    command = [
        str(context["python"]),
        str(patch_script),
        "--app-dir",
        str(app_dir),
    ]
    run_logged(command, log_path=log_path, cwd=PROJECT_DIR)
    return {
        "script": str(patch_script),
        "app_dir": str(app_dir),
        "pulp_nn_utils_c": str(app_dir / "src" / "pulp_nn_utils.c"),
        "status": "patched",
    }


def run_gap8_layer_compare(
    *,
    context: dict[str, Any],
    gvsoc_log_path: Path,
    layer_manifest_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    compare_script = Path(context["gap8_layer_compare_script"])
    if not compare_script.is_file():
        raise FileNotFoundError(f"GAP8 layer compare script not found: {compare_script}")
    command = [
        str(context["python"]),
        str(compare_script),
        "--gvsoc-log",
        str(gvsoc_log_path),
        "--layer-manifest",
        str(layer_manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    run_logged(command, log_path=log_path, cwd=PROJECT_DIR)
    return read_json(output_dir / "runtime_layer_compare.json")


def compare_raw_tensor_lists(expected: list[int], actual: list[int]) -> dict[str, Any]:
    pair_count = min(len(expected), len(actual))
    diffs = [int(actual[idx]) - int(expected[idx]) for idx in range(pair_count)]
    abs_diffs = [abs(value) for value in diffs]
    mismatch_indices = [idx for idx, value in enumerate(diffs) if value != 0]
    return {
        "match": expected == actual,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0,
        "l1_diff": int(sum(abs_diffs)) if abs_diffs else 0,
        "mismatch_count": len(mismatch_indices),
        "first_mismatch_index": (int(mismatch_indices[0]) if mismatch_indices else None),
    }


def run_threshold_sweep(
    *,
    dataset_views: dict[str, dict[str, Any]],
    model_type: str,
    head_type: str,
    thresholds: list[float],
    selection_dataset_label: str,
    secondary_dataset_label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for vis_thresh in thresholds:
        dataset_metrics: dict[str, dict[str, Any]] = {}
        for dataset_label, dataset_view in dataset_views.items():
            dataset_metrics[dataset_label] = stage_metrics_for_subset(
                dataset_view["outputs"],
                dataset_view["targets"],
                dataset_view["no_person"],
                model_type=model_type,
                head_type=head_type,
                vis_thresh=float(vis_thresh),
            )
        selection_metrics = dataset_metrics.get(selection_dataset_label) or {}
        secondary_metrics = dataset_metrics.get(secondary_dataset_label) or {}
        rows.append(
            {
                "vis_thresh": float(vis_thresh),
                "selection_score": float(threshold_selection_score(selection_metrics)),
                "datasets": dataset_metrics,
                "selection_dataset_metrics": selection_metrics,
                "secondary_dataset_metrics": secondary_metrics,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
        primary = row.get("selection_dataset_metrics") or {}
        secondary = row.get("secondary_dataset_metrics") or {}
        return (
            float(row.get("selection_score") or 0.0),
            float(primary.get("precision") or 0.0),
            -float(primary.get("no_person_fp_rate") or 0.0),
            float(primary.get("recall") or 0.0),
            float(secondary.get("precision") or 0.0),
            -float(secondary.get("no_person_fp_rate") or 0.0),
            float(secondary.get("recall") or 0.0),
        )

    best_row = max(rows, key=sort_key)
    selected_threshold = float(best_row["vis_thresh"])
    for row in rows:
        row["selected"] = bool(float(row["vis_thresh"]) == selected_threshold)

    return {
        "selection_dataset_label": selection_dataset_label,
        "secondary_dataset_label": secondary_dataset_label,
        "selected_vis_thresh": selected_threshold,
        "rows": rows,
    }


def build_threshold_sweep_markdown(
    sweep_summary: dict[str, Any],
    *,
    checkpoint_vis_thresh: float,
) -> str:
    selection_label = str(sweep_summary["selection_dataset_label"])
    secondary_label = str(sweep_summary["secondary_dataset_label"])
    selected_vis_thresh = float(sweep_summary["selected_vis_thresh"])

    lines = [
        "# plain_follow deployment threshold sweep",
        "",
        f"- checkpoint_vis_thresh: `{checkpoint_vis_thresh}`",
        f"- selected_deployment_vis_thresh: `{selected_vis_thresh}`",
        f"- deployment_onnx: `{sweep_summary.get('deployment_onnx_path')}`",
        f"- deployment_source: `{sweep_summary.get('deployment_semantic_source')}`",
        f"- selection_dataset: `{selection_label}`",
        f"- secondary_dataset: `{secondary_label}`",
        "",
        "| thresh | selected | score | expanded recall | expanded precision | expanded no_person_fp_rate | hard_case recall | hard_case no_person_fp_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in sweep_summary.get("rows") or []:
        primary = (row.get("datasets") or {}).get(selection_label) or {}
        secondary = (row.get("datasets") or {}).get(secondary_label) or {}
        lines.append(
            "| `{:.2f}` | `{}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{:.4f}` | `{:.4f}` |".format(
                float(row.get("vis_thresh") or 0.0),
                "yes" if row.get("selected") else "no",
                float(row.get("selection_score") or 0.0),
                float(primary.get("recall") or 0.0),
                float(primary.get("precision") or 0.0),
                float(primary.get("no_person_fp_rate") or 0.0),
                float(secondary.get("recall") or 0.0),
                float(secondary.get("no_person_fp_rate") or 0.0),
            )
        )
    return "\n".join(lines)


def parse_int_text_file(path: Path) -> list[int]:
    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    return values


def decode_id_tensor_summary(
    values: list[int],
    *,
    model_type: str,
    head_type: str,
    vis_thresh: float,
) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float32).reshape(1, -1)
    semantic = np.asarray(semantic_output(raw, "id"), dtype=np.float32).reshape(1, -1)
    summary = follow_runtime_decode_summary(
        torch.tensor(semantic[0], dtype=torch.float32),
        head_type=head_type,
        model_type=model_type,
        vis_thresh=float(vis_thresh),
    )
    summary["semantic_values"] = [float(value) for value in semantic.reshape(-1)]
    return summary


def select_gvsoc_image(rep16_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"GVSOC image not found: {requested}")
        return requested

    candidates = discover_images(rep16_dir)
    if not candidates:
        raise FileNotFoundError(f"No rep16 images found under {rep16_dir}")
    for candidate in candidates:
        if "visible" in candidate.stem.lower():
            return candidate
    return candidates[0]


def generate_dory_application(
    *,
    dory_python: str,
    dory_root: Path,
    dory_config_template: Path,
    dory_onnx_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    network_generate = dory_root / "network_generate.py"
    if not dory_root.is_dir() or not network_generate.is_file():
        raise FileNotFoundError(f"DORY network_generate.py not found under {dory_root}")
    if not dory_config_template.is_file():
        raise FileNotFoundError(f"DORY config template not found: {dory_config_template}")

    output_dir.mkdir(parents=True, exist_ok=True)
    app_dir = output_dir / "application"
    if app_dir.exists():
        shutil.rmtree(app_dir)

    config_payload = read_json(dory_config_template)
    config_payload["onnx_file"] = str(dory_onnx_path)
    generated_config_path = output_dir / dory_config_template.name
    write_json(generated_config_path, config_payload)

    command = [
        dory_python,
        "-u",
        str(network_generate),
        "NEMO",
        "PULP.GAP8",
        str(generated_config_path),
        "--app_dir",
        str(app_dir),
    ]
    run_logged(command, log_path=log_path, cwd=dory_root)

    return {
        "app_dir": str(app_dir),
        "config_path": str(generated_config_path),
        "network_generate": str(network_generate),
    }


def run_gvsoc_smoke(
    *,
    context: dict[str, Any],
    deployment_vis_thresh: float,
    dory_onnx_path: Path,
    rep16_dir: Path,
    rep16_predictions: dict[str, Any],
    output_dir: Path,
    command_dir: Path,
) -> dict[str, Any]:
    application_root = output_dir / "application_export"
    gvsoc_image = select_gvsoc_image(rep16_dir, context.get("gvsoc_image"))
    prediction_row = find_prediction_row_for_image(rep16_predictions, gvsoc_image)
    semantic_source = str(rep16_predictions.get("simulation_source") or "dory_graph_python")
    network_seed_dir = application_root / "network_generate_seed"
    network_seed = prepare_dory_io_seed(
        context=context,
        image_path=gvsoc_image,
        output_dir=network_seed_dir,
        dory_onnx_path=dory_onnx_path,
        log_path=command_dir / "10a_generate_dory_io_seed.log",
    )
    synced_seed_files = sync_dory_io_seed_into_model_dir(network_seed_dir, dory_onnx_path.parent)
    expected_output = parse_int_text_file(network_seed["output_txt"])
    semantic_output_values = [int(value) for value in prediction_row.get("raw_output") or []]

    generation = generate_dory_application(
        dory_python=str(context["dory_python"]),
        dory_root=Path(context["dory_root"]),
        dory_config_template=Path(context["dory_config_template"]),
        dory_onnx_path=dory_onnx_path,
        output_dir=application_root,
        log_path=command_dir / "10_generate_plain_follow_app.log",
    )
    app_dir = Path(generation["app_dir"])
    gap8_bn_quant_patch = patch_gap8_bn_quant_helpers(
        context=context,
        app_dir=app_dir,
        log_path=command_dir / "10b_patch_gap8_bn_quant_int64.log",
    )
    gap8_layer_manifest_path = build_gap8_layer_manifest(
        context=context,
        dory_manifest_path=Path(network_seed["manifest_path"]),
        network_header_path=app_dir / "inc" / "network.h",
        output_path=application_root / "gap8_layer_manifest.json",
        log_path=command_dir / "10c_build_gap8_layer_manifest.log",
    )
    staged_dir = application_root / "staged_input"
    artifacts = stage_image_artifacts(
        image_path=gvsoc_image,
        onnx_path=dory_onnx_path,
        output_dir=staged_dir,
        app_dir=app_dir,
        model_type=str(context["model_type"]),
        expected_output=expected_output,
        expected_output_name=DEFAULT_GVSOC_LABEL,
        expected_source=str(network_seed["reference_source"]),
    )

    final_tensor_json = application_root / "gvsoc_final_tensor.json"
    run_log_copy = application_root / "gvsoc_run.log"
    runtime_layer_compare_dir = application_root / "runtime_layer_compare"
    gvsoc_env = {
        "HOST_REPO_ROOT": str(REPO_DIR),
        "HOST_APP_DIR": str(app_dir),
        "HOST_VALIDATION_MAIN": str(Path(context["validation_main"])),
        "HOST_EXPECTED_OUTPUT": str(artifacts.output_txt),
        "HOST_INPUT_HEX": str(artifacts.input_hex),
        "HOST_RUN_LOG_COPY": str(run_log_copy),
        "HOST_FINAL_TENSOR_JSON": str(final_tensor_json),
        "HOST_TRACE_LAYER_OUTPUTS": "1",
        "HOST_TRACE_LAYER_OUTPUT_BYTES_PER_LINE": str(context["gvsoc_trace_layer_output_bytes_per_line"]),
        "HOST_PATCH_BN_QUANT_INT64": "1",
        "HOST_LAYER_MANIFEST": str(gap8_layer_manifest_path),
        "COMPARE_SCRIPT": str(Path(context["gvsoc_compare_script"])),
        "VERIFY_AFTER_RUN": "1",
        "EXPECTED_TENSOR_LABEL": DEFAULT_GVSOC_LABEL,
        "EXPECTED_TENSOR_COUNT": str(int(context["output_metadata"]["follow_output_dim"])),
        "RUN_STAGE_DRIFT_DEBUG": "0",
        "AUTO_REFRESH_APP": "0",
        "MODEL_SENTINEL": str(dory_onnx_path),
        "MODEL_MANIFEST": str(generation["config_path"]),
        "PLATFORM": DEFAULT_GVSOC_PLATFORM,
    }
    run_logged(
        ["bash", str(Path(context["gvsoc_script"]))],
        log_path=command_dir / "11_gvsoc_smoke.log",
        cwd=PROJECT_DIR,
        extra_env=gvsoc_env,
    )
    runtime_layer_compare = run_gap8_layer_compare(
        context=context,
        gvsoc_log_path=run_log_copy,
        layer_manifest_path=gap8_layer_manifest_path,
        output_dir=runtime_layer_compare_dir,
        log_path=command_dir / "11a_compare_gap8_layers.log",
    )

    expected_values = parse_int_text_file(Path(artifacts.output_txt))
    actual_payload = read_json(final_tensor_json)
    actual_values = [int(value) for value in actual_payload.get("values") or []]
    tensor_compare = compare_raw_tensor_lists(expected_values, actual_values)
    semantic_compare = compare_raw_tensor_lists(semantic_output_values, expected_values)

    summary = {
        "status": "pass" if expected_values == actual_values else "mismatch",
        "image_path": str(gvsoc_image),
        "expected_source": str(artifacts.expected_source or network_seed["reference_source"]),
        "app_seed_reference_source": str(network_seed["reference_source"]),
        "deployment_semantic_source": semantic_source,
        "semantic_parity_caveat": DORY_SEMANTIC_PARITY_CAVEAT,
        "artifacts": {
            "application_dir": str(app_dir),
            "generated_config": str(generation["config_path"]),
            "network_generate": str(generation["network_generate"]),
            "gap8_bn_quant_patch_script": str(gap8_bn_quant_patch["script"]),
            "gap8_bn_quant_patch_target": str(gap8_bn_quant_patch["pulp_nn_utils_c"]),
            "network_generate_seed_dir": str(network_seed_dir),
            "network_generate_seed_manifest": str(network_seed["manifest_path"]),
            "network_generate_seed_synced_files": synced_seed_files,
            "gap8_layer_manifest": str(gap8_layer_manifest_path),
            "runtime_layer_compare_dir": str(runtime_layer_compare_dir),
            "runtime_layer_compare_json": str(runtime_layer_compare_dir / "runtime_layer_compare.json"),
            "runtime_layer_compare_md": str(runtime_layer_compare_dir / "runtime_layer_compare.md"),
            "staged_dir": str(staged_dir),
            "expected_output": str(artifacts.output_txt),
            "input_hex": str(artifacts.input_hex),
            "run_log": str(run_log_copy),
            "final_tensor_json": str(final_tensor_json),
            "dory_onnx": str(dory_onnx_path),
        },
        "tensor_compare": {**tensor_compare, "expected_values": expected_values, "actual_values": actual_values},
        "semantic_to_app_seed_compare": {
            **semantic_compare,
            "semantic_expected_values": semantic_output_values,
            "app_seed_expected_values": expected_values,
        },
        "gap8_bn_quant_patch": gap8_bn_quant_patch,
        "runtime_layer_compare": runtime_layer_compare,
        "deployment_decode": {
            "expected": decode_id_tensor_summary(
                expected_values,
                model_type=str(context["model_type"]),
                head_type=str(context["follow_head_type"]),
                vis_thresh=float(deployment_vis_thresh),
            ),
            "gvsoc": decode_id_tensor_summary(
                actual_values,
                model_type=str(context["model_type"]),
                head_type=str(context["follow_head_type"]),
                vis_thresh=float(deployment_vis_thresh),
            ),
        },
    }
    write_json(application_root / "gvsoc_summary.json", summary)
    return summary


def build_release_summary_markdown(summary: dict[str, Any]) -> str:
    checkpoint_metrics = summary["metrics"]["checkpoint_val"]
    float_metrics = summary["metrics"]["float_validation"]
    quant_export_metrics = summary["metrics"]["quant_validation"]["export_summary"]
    deployment_metrics = summary["metrics"]["quant_validation"]["deployment_threshold"]
    compare_metrics = summary["metrics"]["pre_to_post_compare"]
    threshold_selection = summary["threshold_selection"]
    gvsoc_summary = summary.get("gvsoc_smoke") or {}
    dory_cleanup = summary.get("dory_cleanup") or {}
    weight_clamp = dory_cleanup.get("weight_clamp") or {}
    expanded_label = str(summary["expanded_dataset_label"])
    hard_case_label = str(summary["hard_case_dataset_label"])
    deployment_thresh = float(summary["deployment_vis_thresh"])

    lines = [
        "# plain_follow production validation",
        "",
        "## Contract",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- model_type: `{summary['model_type']}`",
        f"- follow_head_type: `{summary['follow_head_type']}`",
        f"- checkpoint_vis_thresh: `{summary['checkpoint_vis_thresh']}`",
        f"- quant_eval_vis_thresh: `{summary['quant_eval_vis_thresh']}`",
        f"- deployment_vis_thresh: `{deployment_thresh}`",
        f"- deployment_onnx: `{summary['artifacts']['dory_onnx']}`",
        f"- deployment_validation_source: `{summary['artifacts']['deployment_validation_source']}`",
        "",
        "## Checkpoint Validation",
        f"- follow_score: `{checkpoint_metrics.get('follow_score')}`",
        f"- x_mae: `{checkpoint_metrics.get('x_mae')}`",
        f"- size_mae: `{checkpoint_metrics.get('size_mae')}`",
        f"- recall: `{checkpoint_metrics.get('recall')}`",
        f"- no_person_fp_rate: `{checkpoint_metrics.get('no_person_fp_rate')}`",
        "",
        "## Float Overlay Validation",
        f"- rep16 follow_score: `{(float_metrics.get('rep16') or {}).get('follow_score')}`",
        f"- hard_case follow_score: `{(float_metrics.get('hard_case') or {}).get('follow_score')}`",
        f"- {expanded_label} follow_score: `{(float_metrics.get(expanded_label) or {}).get('follow_score')}`",
        f"- {expanded_label} recall: `{(float_metrics.get(expanded_label) or {}).get('recall')}`",
        f"- {expanded_label} no_person_fp_rate: `{(float_metrics.get(expanded_label) or {}).get('no_person_fp_rate')}`",
        "",
        "## Deployment Threshold Sweep",
        f"- selection_dataset: `{threshold_selection.get('selection_dataset_label')}`",
        f"- secondary_dataset: `{threshold_selection.get('secondary_dataset_label')}`",
        f"- selected_deployment_vis_thresh: `{deployment_thresh}`",
    ]

    for row in threshold_selection.get("rows") or []:
        primary = (row.get("datasets") or {}).get(expanded_label) or {}
        hard_case = (row.get("datasets") or {}).get(hard_case_label) or {}
        lines.append(
            "- thresh `{:.2f}`: score `{:.4f}`, {} recall `{:.4f}`, {} no_person_fp_rate `{:.4f}`, {} recall `{:.4f}`".format(
                float(row.get("vis_thresh") or 0.0),
                float(row.get("selection_score") or 0.0),
                expanded_label,
                float(primary.get("recall") or 0.0),
                expanded_label,
                float(primary.get("no_person_fp_rate") or 0.0),
                hard_case_label,
                float(hard_case.get("recall") or 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Quant Validation",
            f"- export summary {expanded_label} model_id.onnx follow_score: `{(quant_export_metrics.get(expanded_label) or {}).get('follow_score')}`",
            f"- deployment {expanded_label} DORY-semantic follow_score: `{(deployment_metrics.get(expanded_label) or {}).get('follow_score')}`",
            f"- deployment {expanded_label} recall: `{(deployment_metrics.get(expanded_label) or {}).get('recall')}`",
            f"- deployment {expanded_label} precision: `{(deployment_metrics.get(expanded_label) or {}).get('precision')}`",
            f"- deployment {expanded_label} no_person_fp_rate: `{(deployment_metrics.get(expanded_label) or {}).get('no_person_fp_rate')}`",
            f"- deployment {hard_case_label} recall: `{(deployment_metrics.get(hard_case_label) or {}).get('recall')}`",
            f"- deployment rep16 recall: `{(deployment_metrics.get('rep16') or {}).get('recall')}`",
            f"- export float_to_onnx visibility_gate_agreement: `{(quant_export_metrics.get('float_to_onnx_bin_preservation') or {}).get('visibility_gate_agreement')}`",
            "",
            "## DORY Deployment Artifact",
            f"- simplify fallback used: `{dory_cleanup.get('simplify_fallback_used')}`",
            f"- weight clamp applied: `{weight_clamp.get('applied')}`",
            f"- clipped initializer count: `{weight_clamp.get('initializer_count')}`",
            f"- clipped value count: `{weight_clamp.get('total_clipped_values')}`",
            f"- remaining out-of-range initializers: `{len(dory_cleanup.get('weight_range_audit_after') or [])}`",
            f"- runtime validation gate: `{dory_cleanup.get('runtime_validation_gate')}`",
            f"- semantic caveat: `{dory_cleanup.get('semantic_parity_caveat')}`",
            f"- deployment predictions source: `{summary['artifacts']['deployment_validation_source']}`",
            "",
            "## Visual Drift Checks",
            f"- rep16 visibility_gate_agreement: `{(compare_metrics.get('rep16') or {}).get('visibility_gate_agreement')}`",
            f"- {hard_case_label} visibility_gate_agreement: `{(compare_metrics.get(hard_case_label) or {}).get('visibility_gate_agreement')}`",
            f"- {expanded_label} visibility_gate_agreement: `{(compare_metrics.get(expanded_label) or {}).get('visibility_gate_agreement')}`",
        ]
    )

    if gvsoc_summary:
        gvsoc_decode = ((gvsoc_summary.get("deployment_decode") or {}).get("gvsoc") or {})
        semantic_compare = gvsoc_summary.get("semantic_to_app_seed_compare") or {}
        gap8_bn_quant_patch = gvsoc_summary.get("gap8_bn_quant_patch") or {}
        runtime_layer_compare = gvsoc_summary.get("runtime_layer_compare") or {}
        first_divergent_layer = runtime_layer_compare.get("first_divergent_layer") or {}
        lines.extend(
            [
                "",
                "## GVSOC Smoke",
                f"- status: `{gvsoc_summary.get('status')}`",
                f"- image: `{gvsoc_summary.get('image_path')}`",
                f"- expected source: `{gvsoc_summary.get('expected_source')}`",
                f"- deployment semantic source: `{gvsoc_summary.get('deployment_semantic_source')}`",
                f"- final tensor exact match: `{((gvsoc_summary.get('tensor_compare') or {}).get('match'))}`",
                f"- semantic_to_app_seed max_abs_diff: `{semantic_compare.get('max_abs_diff')}`",
                f"- semantic_to_app_seed mismatch_count: `{semantic_compare.get('mismatch_count')}`",
                f"- gap8_bn_quant_patch_status: `{gap8_bn_quant_patch.get('status')}`",
                f"- decoded visible: `{gvsoc_decode.get('target_visible')}`",
                f"- decoded vis_conf: `{gvsoc_decode.get('visibility_confidence')}`",
                f"- decoded x_value: `{gvsoc_decode.get('x_value')}`",
                f"- decoded size_value: `{gvsoc_decode.get('size_value')}`",
                f"- first divergent layer: `{first_divergent_layer.get('index')} {first_divergent_layer.get('layer_name')}`",
                f"- first divergent mean_abs_diff: `{first_divergent_layer.get('mean_abs_diff')}`",
                f"- semantic caveat: `{gvsoc_summary.get('semantic_parity_caveat')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- calibration_manifest: `{summary['artifacts']['calibration_manifest']}`",
            f"- validation_pack_manifest: `{summary['artifacts']['expanded_pack_manifest']}`",
            f"- expanded_eval_pack: `{summary['artifacts']['expanded_eval_dir']}`",
            f"- quant_summary: `{summary['artifacts']['quant_summary']}`",
            f"- threshold_sweep_json: `{summary['artifacts']['threshold_sweep_json']}`",
            f"- threshold_sweep_md: `{summary['artifacts']['threshold_sweep_md']}`",
            f"- dory_cleanup_report_json: `{summary['artifacts']['dory_cleanup_report_json']}`",
            f"- dory_cleanup_report_md: `{summary['artifacts']['dory_cleanup_report_md']}`",
            f"- dory_semantic_rep16_predictions: `{summary['artifacts']['dory_semantic_rep16_predictions']}`",
            f"- dory_semantic_hard_case_predictions: `{summary['artifacts']['dory_semantic_hard_case_predictions']}`",
            f"- dory_semantic_{expanded_label}_predictions: `{summary['artifacts'][f'dory_semantic_{expanded_label}_predictions']}`",
            f"- compare_expanded_eval_summary: `{summary['artifacts']['compare_expanded_eval_summary']}`",
        ]
    )

    if summary["artifacts"].get("gvsoc_summary_json"):
        lines.extend(
            [
                f"- gvsoc_summary_json: `{summary['artifacts']['gvsoc_summary_json']}`",
                f"- gvsoc_run_log: `{summary['artifacts']['gvsoc_run_log']}`",
                f"- gvsoc_app_seed_manifest: `{summary['artifacts']['gvsoc_app_seed_manifest']}`",
                f"- gvsoc_gap8_layer_manifest: `{summary['artifacts']['gvsoc_gap8_layer_manifest']}`",
                f"- gvsoc_gap8_bn_quant_patch_script: `{summary['artifacts']['gvsoc_gap8_bn_quant_patch_script']}`",
                f"- gvsoc_gap8_bn_quant_patch_target: `{summary['artifacts']['gvsoc_gap8_bn_quant_patch_target']}`",
                f"- gvsoc_runtime_layer_compare_json: `{summary['artifacts']['gvsoc_runtime_layer_compare_json']}`",
                f"- gvsoc_runtime_layer_compare_md: `{summary['artifacts']['gvsoc_runtime_layer_compare_md']}`",
            ]
        )

    lines.append(f"- release_summary_json: `{summary['artifacts']['release_summary_json']}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    ensure_output_dir(context["output_dir"], overwrite=bool(args.overwrite))

    python_bin = str(context["python"])
    ckpt_path = Path(context["ckpt_path"])
    follow_head_type = str(context["follow_head_type"])
    checkpoint_vis_thresh = float(context["checkpoint_vis_thresh"])
    quant_eval_vis_thresh = float(context["quant_eval_vis_thresh"])
    output_dir = Path(context["output_dir"])

    command_dir = output_dir / "commands"
    calibration_manifest_path = output_dir / "calibration_manifest_128.json"
    expanded_pack_manifest_path = output_dir / "expanded_eval_manifest.json"

    build_calibration_manifest_cmd = [
        python_bin,
        str(SCRIPT_DIR / "build_follow_calibration_manifest.py"),
        "--image-dir",
        str(context["val_image_dir"]),
        "--annotations",
        str(context["annotations"]),
        "--model-type",
        "plain_follow",
        "--follow-head-type",
        follow_head_type,
        "--target-count",
        str(int(args.calib_target_count)),
        "--output",
        str(calibration_manifest_path),
        "--overwrite",
    ]
    if int(args.calib_max_images) > 0:
        build_calibration_manifest_cmd.extend(["--max-images", str(int(args.calib_max_images))])
    run_logged(
        build_calibration_manifest_cmd,
        log_path=command_dir / "01_build_calibration_manifest.log",
    )

    build_expanded_pack_manifest_cmd = [
        python_bin,
        str(SCRIPT_DIR / "build_follow_calibration_manifest.py"),
        "--image-dir",
        str(context["val_image_dir"]),
        "--annotations",
        str(context["annotations"]),
        "--model-type",
        "plain_follow",
        "--follow-head-type",
        follow_head_type,
        "--target-count",
        str(max(int(args.expanded_pack_extra_count), 1)),
        "--output",
        str(expanded_pack_manifest_path),
        "--overwrite",
    ]
    if int(args.expanded_pack_max_images) > 0:
        build_expanded_pack_manifest_cmd.extend(["--max-images", str(int(args.expanded_pack_max_images))])
    run_logged(
        build_expanded_pack_manifest_cmd,
        log_path=command_dir / "02_build_expanded_eval_manifest.log",
    )

    expanded_pack = build_expanded_pack(
        rep16_dir=Path(context["rep16_dir"]),
        hard_case_dir=Path(context["hard_case_dir"]),
        manifest_payload=read_json(expanded_pack_manifest_path),
        extra_count=int(args.expanded_pack_extra_count),
        output_root=output_dir,
        copy_mode=str(args.copy_mode),
    )

    rep16_local_dir = Path((expanded_pack["rep16"] or {})["output_dir"])
    hard_case_local_dir = Path((expanded_pack["hard_case"] or {})["output_dir"])
    expanded_eval_local_dir = Path((expanded_pack["expanded_eval"] or {})["output_dir"])
    expanded_dataset_label = "expanded_eval"
    hard_case_dataset_label = "hard_case"

    float_jobs = [
        ("rep16", rep16_local_dir, output_dir / "float_rep16", "03_float_rep16.log"),
        ("hard_case", hard_case_local_dir, output_dir / "float_hard_case", "04_float_hard_case.log"),
        (
            expanded_dataset_label,
            expanded_eval_local_dir,
            output_dir / f"float_{expanded_dataset_label}",
            "05_float_expanded_eval.log",
        ),
    ]
    float_summaries: dict[str, dict[str, Any]] = {}
    for dataset_label, images_dir, dataset_output_dir, log_name in float_jobs:
        command = [
            python_bin,
            str(SCRIPT_DIR / "validate_follow_rep16_overlays.py"),
            "--ckpt",
            str(ckpt_path),
            "--output-dir",
            str(dataset_output_dir),
            "--images-dir",
            str(images_dir),
            "--annotations",
            str(context["annotations"]),
            "--dataset-label",
            dataset_label,
            "--vis-thresh",
            str(quant_eval_vis_thresh),
            "--overwrite",
        ]
        run_logged(command, log_path=command_dir / log_name)
        float_summaries[dataset_label] = read_json(dataset_output_dir / "summary.json")

    quant_output_dir = output_dir / "quant_eval"
    quant_command = [
        python_bin,
        str(SCRIPT_DIR / "evaluate_quant_native_follow.py"),
        "--ckpt",
        str(ckpt_path),
        "--output-dir",
        str(quant_output_dir),
        "--rep16-dir",
        str(expanded_eval_local_dir),
        "--hard-case-dir",
        str(hard_case_local_dir),
        "--primary-dataset-label",
        expanded_dataset_label,
        "--secondary-dataset-label",
        hard_case_dataset_label,
        "--annotations",
        str(context["annotations"]),
        "--calib-dir",
        str(context["val_image_dir"]),
        "--calib-manifest",
        str(calibration_manifest_path),
        "--vis-thresh",
        str(quant_eval_vis_thresh),
        "--overwrite",
    ]
    run_logged(quant_command, log_path=command_dir / "06_quant_eval.log")
    quant_summary = read_json(quant_output_dir / "summary.json")

    quant_onnx_path = Path(
        (quant_summary.get("artifacts") or {}).get("onnx") or (quant_output_dir / "model_id.onnx")
    )
    if not quant_onnx_path.is_file():
        raise FileNotFoundError(f"Quantized ONNX not found after quant eval: {quant_onnx_path}")

    dory_onnx_path = Path(
        (quant_summary.get("artifacts") or {}).get("dory_onnx") or (quant_output_dir / "model_id_dory.onnx")
    )
    if not dory_onnx_path.is_file():
        raise FileNotFoundError(f"DORY ONNX not found after quant eval: {dory_onnx_path}")

    annotations = QuantAnnotationIndex(Path(context["annotations"]))
    deployment_views: dict[str, dict[str, Any]] = {}
    deployment_prediction_artifacts: dict[str, Path] = {}
    deployment_raw_predictions: dict[str, dict[str, Any]] = {}
    deployment_jobs = [
        ("rep16", rep16_local_dir, output_dir / "dory_semantic_rep16", "07_dory_semantic_rep16.log"),
        (
            hard_case_dataset_label,
            hard_case_local_dir,
            output_dir / "dory_semantic_hard_case",
            "08_dory_semantic_hard_case.log",
        ),
        (
            expanded_dataset_label,
            expanded_eval_local_dir,
            output_dir / f"dory_semantic_{expanded_dataset_label}",
            "09_dory_semantic_expanded_eval.log",
        ),
    ]
    for dataset_label, images_dir, dataset_output_dir, log_name in deployment_jobs:
        deployment_view, deployment_predictions = build_dory_semantic_dataset_view(
            context=context,
            image_dir=images_dir,
            annotations=annotations,
            image_size=context["image_size"],
            model_type=str(context["model_type"]),
            dory_onnx_path=dory_onnx_path,
            work_dir=dataset_output_dir,
            command_log=command_dir / log_name,
        )
        deployment_views[dataset_label] = deployment_view
        deployment_raw_predictions[dataset_label] = deployment_predictions
        deployment_prediction_artifacts[dataset_label] = dataset_output_dir / "predictions.json"
    threshold_sweep = run_threshold_sweep(
        dataset_views=deployment_views,
        model_type=str(context["model_type"]),
        head_type=follow_head_type,
        thresholds=list(context["threshold_candidates"]),
        selection_dataset_label=expanded_dataset_label,
        secondary_dataset_label=hard_case_dataset_label,
    )
    threshold_sweep["deployment_onnx_path"] = str(dory_onnx_path)
    threshold_sweep["deployment_semantic_source"] = str(
        (deployment_raw_predictions.get(expanded_dataset_label) or {}).get("simulation_source") or "dory_graph_python"
    )
    deployment_vis_thresh = float(threshold_sweep["selected_vis_thresh"])
    threshold_sweep_path = output_dir / "deployment_threshold_sweep.json"
    threshold_sweep_md_path = output_dir / "deployment_threshold_sweep.md"
    write_json(threshold_sweep_path, threshold_sweep)
    write_markdown(
        threshold_sweep_md_path,
        build_threshold_sweep_markdown(
            threshold_sweep,
            checkpoint_vis_thresh=checkpoint_vis_thresh,
        ),
    )

    compare_jobs = [
        ("rep16", rep16_local_dir, output_dir / "compare_rep16", "10_compare_rep16.log"),
        (hard_case_dataset_label, hard_case_local_dir, output_dir / "compare_hard_case", "11_compare_hard_case.log"),
        (
            expanded_dataset_label,
            expanded_eval_local_dir,
            output_dir / "compare_expanded_eval",
            "12_compare_expanded_eval.log",
        ),
    ]
    compare_summaries: dict[str, dict[str, Any]] = {}
    for dataset_label, images_dir, dataset_output_dir, log_name in compare_jobs:
        command = [
            python_bin,
            str(SCRIPT_DIR / "compare_quant_native_follow_rep16_overlays.py"),
            "--ckpt",
            str(ckpt_path),
            "--onnx",
            str(dory_onnx_path),
            "--post-predictions-json",
            str(deployment_prediction_artifacts[dataset_label]),
            "--output-dir",
            str(dataset_output_dir),
            "--images-dir",
            str(images_dir),
            "--annotations",
            str(context["annotations"]),
            "--dataset-label",
            dataset_label,
            "--vis-thresh",
            str(deployment_vis_thresh),
            "--overwrite",
        ]
        run_logged(command, log_path=command_dir / log_name)
        compare_summaries[dataset_label] = read_json(dataset_output_dir / "comparison_summary.json")

    gvsoc_summary = None
    if not args.skip_gvsoc:
        gvsoc_summary = run_gvsoc_smoke(
            context=context,
            deployment_vis_thresh=deployment_vis_thresh,
            dory_onnx_path=dory_onnx_path,
            rep16_dir=rep16_local_dir,
            rep16_predictions=deployment_raw_predictions["rep16"],
            output_dir=output_dir,
            command_dir=command_dir,
        )

    deployment_metrics = {}
    selected_row = next(row for row in threshold_sweep["rows"] if row.get("selected"))
    for dataset_label in ("rep16", hard_case_dataset_label, expanded_dataset_label):
        deployment_metrics[dataset_label] = dict((selected_row.get("datasets") or {}).get(dataset_label) or {})

    release_summary = {
        "checkpoint_path": str(ckpt_path),
        "model_type": "plain_follow",
        "follow_head_type": follow_head_type,
        "checkpoint_vis_thresh": checkpoint_vis_thresh,
        "quant_eval_vis_thresh": quant_eval_vis_thresh,
        "deployment_vis_thresh": deployment_vis_thresh,
        "expanded_dataset_label": expanded_dataset_label,
        "hard_case_dataset_label": hard_case_dataset_label,
        "threshold_selection": {
            "selection_dataset_label": threshold_sweep["selection_dataset_label"],
            "secondary_dataset_label": threshold_sweep["secondary_dataset_label"],
            "selected_vis_thresh": deployment_vis_thresh,
            "rows": threshold_sweep["rows"],
        },
        "artifacts": {
            "calibration_manifest": str(calibration_manifest_path),
            "expanded_pack_manifest": str(expanded_pack_manifest_path),
            "expanded_eval_pack_summary": str(output_dir / "expanded_eval_pack_summary.json"),
            "rep16_dir": str(rep16_local_dir),
            "hard_case_dir": str(hard_case_local_dir),
            "expanded_eval_dir": str(expanded_eval_local_dir),
            "float_rep16_summary": str(output_dir / "float_rep16" / "summary.json"),
            "float_hard_case_summary": str(output_dir / "float_hard_case" / "summary.json"),
            f"float_{expanded_dataset_label}_summary": str(
                output_dir / f"float_{expanded_dataset_label}" / "summary.json"
            ),
            "quant_summary": str(quant_output_dir / "summary.json"),
            "quant_summary_md": str(quant_output_dir / "summary.md"),
            "quant_onnx": str(quant_onnx_path),
            "dory_onnx": str(dory_onnx_path),
            "deployment_validation_source": str(
                threshold_sweep.get("deployment_semantic_source") or "dory_graph_python"
            ),
            "dory_cleanup_report_json": str(quant_output_dir / "dory_cleanup_report.json"),
            "dory_cleanup_report_md": str(quant_output_dir / "dory_cleanup_report.md"),
            "threshold_sweep_json": str(threshold_sweep_path),
            "threshold_sweep_md": str(threshold_sweep_md_path),
            "dory_semantic_rep16_predictions": str(deployment_prediction_artifacts["rep16"]),
            "dory_semantic_hard_case_predictions": str(deployment_prediction_artifacts[hard_case_dataset_label]),
            f"dory_semantic_{expanded_dataset_label}_predictions": str(
                deployment_prediction_artifacts[expanded_dataset_label]
            ),
            "compare_rep16_summary": str(output_dir / "compare_rep16" / "comparison_summary.json"),
            "compare_hard_case_summary": str(output_dir / "compare_hard_case" / "comparison_summary.json"),
            "compare_expanded_eval_summary": str(output_dir / "compare_expanded_eval" / "comparison_summary.json"),
            "gvsoc_summary_json": (
                str(output_dir / "application_export" / "gvsoc_summary.json") if gvsoc_summary else None
            ),
            "gvsoc_run_log": (str(output_dir / "application_export" / "gvsoc_run.log") if gvsoc_summary else None),
            "gvsoc_app_seed_manifest": (
                str((output_dir / "application_export" / "network_generate_seed" / "nemo_dory_artifacts.json"))
                if gvsoc_summary
                else None
            ),
            "gvsoc_gap8_layer_manifest": (
                str(output_dir / "application_export" / "gap8_layer_manifest.json") if gvsoc_summary else None
            ),
            "gvsoc_gap8_bn_quant_patch_script": (
                str(context["gap8_bn_quant_patch_script"]) if gvsoc_summary else None
            ),
            "gvsoc_gap8_bn_quant_patch_target": (
                str(output_dir / "application_export" / "application" / "src" / "pulp_nn_utils.c")
                if gvsoc_summary
                else None
            ),
            "gvsoc_runtime_layer_compare_json": (
                str(output_dir / "application_export" / "runtime_layer_compare" / "runtime_layer_compare.json")
                if gvsoc_summary
                else None
            ),
            "gvsoc_runtime_layer_compare_md": (
                str(output_dir / "application_export" / "runtime_layer_compare" / "runtime_layer_compare.md")
                if gvsoc_summary
                else None
            ),
        },
        "metrics": {
            "checkpoint_val": dict((context["payload"].get("val_stats") or {})),
            "float_validation": {
                dataset_label: dict(summary.get("metrics") or {})
                for dataset_label, summary in float_summaries.items()
            },
            "quant_validation": {
                "export_summary": {
                    expanded_dataset_label: dict(
                        (((quant_summary.get("datasets") or {}).get(expanded_dataset_label) or {}).get("onnx") or {})
                    ),
                    hard_case_dataset_label: dict(
                        (((quant_summary.get("datasets") or {}).get(hard_case_dataset_label) or {}).get("onnx") or {})
                    ),
                    "float_to_onnx_bin_preservation": dict(
                        (quant_summary.get("quant_fidelity") or {}).get("float_to_onnx_bin_preservation") or {}
                    ),
                },
                "deployment_threshold": deployment_metrics,
            },
            "pre_to_post_compare": {
                dataset_label: dict(((summary.get("metrics") or {}).get("pre_to_post") or {}))
                for dataset_label, summary in compare_summaries.items()
            },
        },
        "dory_cleanup": dict(quant_summary.get("dory_cleanup") or {}),
        "gvsoc_smoke": gvsoc_summary,
    }
    release_summary_path = output_dir / "release_summary.json"
    release_summary["artifacts"]["release_summary_json"] = str(release_summary_path)
    write_json(release_summary_path, release_summary)
    write_markdown(output_dir / "release_summary.md", build_release_summary_markdown(release_summary))


if __name__ == "__main__":
    main()

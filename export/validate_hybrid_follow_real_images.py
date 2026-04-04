#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hybrid_follow_image_artifacts import PREPROCESS_DESCRIPTION, stage_image_artifacts
from visualize_hybrid_follow_prediction import (
    DEFAULT_ANN,
    DEFAULT_Q_SCALE,
    DEFAULT_VIS_THRESHOLD,
    generate_overlay_for_sample,
)


FINAL_BEGIN_RE = re.compile(r"^FINAL_TENSOR_I32_BEGIN\s+(\w+)\s+count=(\d+)$")
FINAL_LINE_RE = re.compile(r"^FINAL_TENSOR_I32\s+(\w+)(.*)$")
FINAL_END_RE = re.compile(r"^FINAL_TENSOR_I32_END\s+(\w+)$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid_follow GAP8 validation on a directory of real images."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory of validation images. Files are discovered recursively.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Directory for per-image artifacts and summaries. Defaults to "
            "export/hybrid_follow/real_image_validation/<timestamp>."
        ),
    )
    parser.add_argument(
        "--app-dir",
        default="application",
        help="Generated GAP8 app directory relative to pytorch_ssd.",
    )
    parser.add_argument(
        "--onnx",
        default="export/hybrid_follow/hybrid_follow_dory.onnx",
        help="DORY ONNX used for Python-side golden output generation.",
    )
    parser.add_argument(
        "--run-script",
        default="tools/run_aideck_val_impl.sh",
        help="Path to the AI-Deck validation script.",
    )
    parser.add_argument("--platform", default="gvsoc", help="Validation platform passed to the AI-Deck validation script.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to run.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the results directory first if it already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch on the first failure instead of continuing.",
    )
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Do not generate annotated prediction overlay images for each sample.",
    )
    parser.add_argument(
        "--overlay-annotations",
        default=str(DEFAULT_ANN),
        help="Optional COCO annotations json used for GT person overlays.",
    )
    parser.add_argument(
        "--overlay-q-scale",
        type=float,
        default=DEFAULT_Q_SCALE,
        help="Quantization scale used to decode the 3-value hybrid_follow tensor.",
    )
    parser.add_argument(
        "--overlay-vis-threshold",
        type=float,
        default=DEFAULT_VIS_THRESHOLD,
        help="Visibility threshold used in overlay generation.",
    )
    parser.set_defaults(stage_drift=True)
    parser.add_argument(
        "--stage-drift",
        dest="stage_drift",
        action="store_true",
        help="Run compare_hybrid_follow_stages.py for each sample after validation.",
    )
    parser.add_argument(
        "--no-stage-drift",
        dest="stage_drift",
        action="store_false",
        help="Disable the per-sample stage-drift comparison.",
    )
    parser.add_argument(
        "--stage-drift-ckpt",
        default="training/hybrid_follow/hybrid_follow_best_follow_score.pth",
        help="Checkpoint passed to the stage-drift comparison tool.",
    )
    parser.add_argument(
        "--stage-drift-python",
        default=None,
        help="Optional Python interpreter for the stage-drift tool. Defaults to ../nemoenv/bin/python3 when available.",
    )
    parser.add_argument(
        "--stage-drift-nemo-stage",
        default="skip",
        choices=["auto", "skip", "fq", "qd", "id"],
        help="Optional in-memory NEMO stage used by the stage-drift tool.",
    )
    parser.add_argument(
        "--trace-layer-outputs",
        action="store_true",
        help="Enable per-layer GVSOC byte tracing for targeted debug runs.",
    )
    parser.add_argument(
        "--trace-layer-output-bytes-per-line",
        type=int,
        default=64,
        help="Chunk size for per-layer GVSOC byte trace lines.",
    )
    parser.add_argument(
        "--layer-manifest",
        default="export/hybrid_follow/gap8_layer_manifest.json",
        help="Combined GAP8/DORY layer manifest used to decode runtime traces.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "image"


def discover_images(images_dir: Path) -> list[Path]:
    images = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def ensure_results_dir(results_dir: Path, overwrite: bool) -> None:
    if results_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Results directory already exists: {results_dir}. "
                "Use --overwrite or choose a different --results-dir."
            )
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)


def parse_int_file(path: Path) -> list[int]:
    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    return values


def parse_tensor_from_log(path: Path, label: str = "final") -> tuple[int, list[int]]:
    values: list[int] = []
    expected_count: int | None = None
    collecting = False

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        begin_match = FINAL_BEGIN_RE.match(line)
        if begin_match and begin_match.group(1) == label:
            expected_count = int(begin_match.group(2))
            values = []
            collecting = True
            continue

        end_match = FINAL_END_RE.match(line)
        if end_match and end_match.group(1) == label:
            collecting = False
            continue

        line_match = FINAL_LINE_RE.match(line)
        if line_match and line_match.group(1) == label and collecting:
            payload = line_match.group(2).strip()
            if payload:
                values.extend(int(token) for token in payload.split())

    if expected_count is None:
        raise RuntimeError(f"No FINAL_TENSOR_I32 block found for label 'final' in {path}")
    return expected_count, values


def run_validation_case(
    run_script: Path,
    app_dir: Path,
    expected_output: Path,
    input_hex: Path,
    gvsoc_log_copy: Path,
    platform: str,
    trace_layer_outputs: bool,
    trace_layer_output_bytes_per_line: int,
    layer_manifest: Path | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOST_APP_DIR"] = str(app_dir)
    env["HOST_EXPECTED_OUTPUT"] = str(expected_output)
    env["HOST_INPUT_HEX"] = str(input_hex)
    env["HOST_RUN_LOG_COPY"] = str(gvsoc_log_copy)
    env["RUN_LOG_NAME"] = "run_gvsoc.log"
    env["PLATFORM"] = platform
    env["VERIFY_AFTER_RUN"] = "1"
    env["HOST_TRACE_LAYER_OUTPUTS"] = "1" if trace_layer_outputs else "0"
    env["HOST_TRACE_LAYER_OUTPUT_BYTES_PER_LINE"] = str(trace_layer_output_bytes_per_line)
    if layer_manifest is not None:
        env["HOST_LAYER_MANIFEST"] = str(layer_manifest)
    shell_script = run_script.name if os.name == "nt" else str(run_script)

    return subprocess.run(
        ["bash", shell_script],
        cwd=str(PROJECT_DIR),
        env=env,
        capture_output=True,
        text=True,
    )


def resolve_stage_drift_python(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.extend(
        [
            (PROJECT_DIR.parent / "nemoenv" / "bin" / "python3"),
            (PROJECT_DIR.parent / "nemoenv" / "Scripts" / "python.exe"),
            Path(sys.executable),
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return Path(sys.executable).resolve()


def run_stage_drift_case(
    stage_drift_python: Path,
    image_path: Path,
    ckpt_path: Path,
    onnx_path: Path,
    golden_output: Path,
    output_dir: Path,
    nemo_stage: str,
    gvsoc_json: Path | None = None,
    gvsoc_log: Path | None = None,
    layer_manifest: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(stage_drift_python),
        str((PROJECT_DIR / "export" / "compare_hybrid_follow_stages.py").resolve()),
        "--image",
        str(image_path),
        "--ckpt",
        str(ckpt_path),
        "--onnx",
        str(onnx_path),
        "--golden",
        str(golden_output),
        "--output-dir",
        str(output_dir),
        "--overwrite",
        "--nemo-stage",
        nemo_stage,
    ]
    if gvsoc_json is not None and gvsoc_json.is_file():
        cmd.extend(["--gvsoc-json", str(gvsoc_json)])
    if gvsoc_log is not None and gvsoc_log.is_file():
        cmd.extend(["--gvsoc-log", str(gvsoc_log)])
    if layer_manifest is not None and layer_manifest.is_file():
        cmd.extend(["--layer-manifest", str(layer_manifest)])
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "index",
        "image_path",
        "sample_dir",
        "status",
        "returncode",
        "expected_tensor",
        "actual_tensor",
        "mismatch",
        "elapsed_seconds",
        "gvsoc_log",
        "overlay_path",
        "overlay_status",
        "overlay_error",
        "stage_drift_dir",
        "stage_drift_status",
        "stage_drift_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "image_path": row["image_path"],
                    "sample_dir": row["sample_dir"],
                    "status": row["status"],
                    "returncode": row["returncode"],
                    "expected_tensor": json.dumps(row["expected_tensor"]),
                    "actual_tensor": json.dumps(row.get("actual_tensor")),
                    "mismatch": row.get("mismatch", ""),
                    "elapsed_seconds": f"{row['elapsed_seconds']:.3f}",
                    "gvsoc_log": row.get("gvsoc_log", ""),
                    "overlay_path": row.get("overlay_path", ""),
                    "overlay_status": row.get("overlay_status", ""),
                    "overlay_error": row.get("overlay_error", ""),
                    "stage_drift_dir": row.get("stage_drift_dir", ""),
                    "stage_drift_status": row.get("stage_drift_status", ""),
                    "stage_drift_error": row.get("stage_drift_error", ""),
                }
            )


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else (PROJECT_DIR / "export" / "hybrid_follow" / "real_image_validation" / timestamp).resolve()
    )
    if PROJECT_DIR not in results_dir.parents and results_dir != PROJECT_DIR:
        raise RuntimeError(
            f"Results dir must live under {PROJECT_DIR} so Docker can access staged files: {results_dir}"
        )
    ensure_results_dir(results_dir, overwrite=args.overwrite)

    app_dir = (PROJECT_DIR / args.app_dir).resolve()
    run_script = (PROJECT_DIR / args.run_script).resolve()
    onnx_path = (PROJECT_DIR / args.onnx).resolve()
    layer_manifest = (PROJECT_DIR / args.layer_manifest).resolve() if args.layer_manifest else None
    stage_drift_python = resolve_stage_drift_python(args.stage_drift_python) if args.stage_drift else None
    stage_drift_ckpt = (PROJECT_DIR / args.stage_drift_ckpt).resolve() if args.stage_drift else None

    if not app_dir.is_dir():
        raise FileNotFoundError(f"Application directory not found: {app_dir}")
    if not run_script.is_file():
        raise FileNotFoundError(f"Validation script not found: {run_script}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"DORY ONNX not found: {onnx_path}")
    if args.stage_drift and stage_drift_ckpt is not None and not stage_drift_ckpt.is_file():
        raise FileNotFoundError(f"Stage-drift checkpoint not found: {stage_drift_ckpt}")

    images = discover_images(images_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No images found under {images_dir}")

    summary_rows: list[dict] = []
    start_time = time.time()

    for index, image_path in enumerate(images, start=1):
        sample_name = f"{index:04d}_{sanitize_name(image_path.stem)}"
        sample_dir = results_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        case_start = time.time()
        artifacts = stage_image_artifacts(
            image_path=image_path,
            onnx_path=onnx_path,
            output_dir=sample_dir,
        )

        gvsoc_log_path = sample_dir / "gvsoc.log"
        runner_log_path = sample_dir / "run_aideck_val.log"
        comparison_path = sample_dir / "comparison.json"
        actual_tensor_path = sample_dir / "gvsoc_final_tensor.json"

        result = run_validation_case(
            run_script=run_script,
            app_dir=app_dir,
            expected_output=Path(artifacts.output_txt),
            input_hex=Path(artifacts.input_hex),
            gvsoc_log_copy=gvsoc_log_path,
            platform=args.platform,
            trace_layer_outputs=bool(args.trace_layer_outputs),
            trace_layer_output_bytes_per_line=int(args.trace_layer_output_bytes_per_line),
            layer_manifest=layer_manifest,
        )
        runner_log_path.write_text(result.stdout + result.stderr, encoding="utf-8")

        actual_tensor: list[int] | None = None
        mismatch = ""
        status = "pass"
        overlay_path: str | None = None
        overlay_status = "skipped" if args.skip_overlays else "pending"
        overlay_error = ""
        overlay_gt_x_offset: float | None = None
        overlay_gt_size_proxy: float | None = None
        stage_drift_dir = sample_dir / "stage_drift"
        stage_drift_status = "skipped" if args.stage_drift else "disabled"
        stage_drift_error = ""

        if gvsoc_log_path.is_file():
            try:
                logged_count, actual_tensor = parse_tensor_from_log(gvsoc_log_path)
                write_json(
                    actual_tensor_path,
                    {
                        "label": "final",
                        "count": logged_count,
                        "values": actual_tensor,
                    },
                )
            except Exception as exc:
                mismatch = f"Failed to parse GVSOC log: {exc}"
                status = "fail"
        else:
            mismatch = "GVSOC log was not produced."
            status = "fail"

        expected_tensor = parse_int_file(Path(artifacts.output_txt))
        if actual_tensor is not None and actual_tensor != expected_tensor:
            mismatch = (
                f"Expected {expected_tensor} but got {actual_tensor}"
                if not mismatch
                else mismatch
            )
            status = "fail"

        if result.returncode != 0 and status == "pass":
            mismatch = "The AI-Deck validation script returned a non-zero exit code."
            status = "fail"

        elapsed_seconds = time.time() - case_start
        comparison = {
            "index": index,
            "image_path": str(image_path),
            "sample_dir": str(sample_dir),
            "status": status,
            "returncode": result.returncode,
            "expected_tensor": expected_tensor,
            "actual_tensor": actual_tensor,
            "mismatch": mismatch,
            "elapsed_seconds": elapsed_seconds,
            "gvsoc_log": str(gvsoc_log_path) if gvsoc_log_path.is_file() else None,
            "runner_log": str(runner_log_path),
            "preprocess": PREPROCESS_DESCRIPTION,
            "overlay_path": overlay_path,
            "overlay_status": overlay_status,
            "overlay_error": overlay_error,
            "overlay_gt_x_offset": overlay_gt_x_offset,
            "overlay_gt_size_proxy": overlay_gt_size_proxy,
            "stage_drift_dir": str(stage_drift_dir) if args.stage_drift else None,
            "stage_drift_status": stage_drift_status,
            "stage_drift_error": stage_drift_error,
        }

        write_json(comparison_path, comparison)

        if args.stage_drift:
            drift_result = run_stage_drift_case(
                stage_drift_python=stage_drift_python,
                image_path=image_path,
                ckpt_path=stage_drift_ckpt,
                onnx_path=onnx_path,
                golden_output=Path(artifacts.output_txt),
                output_dir=stage_drift_dir,
                nemo_stage=args.stage_drift_nemo_stage,
                gvsoc_json=actual_tensor_path if actual_tensor_path.is_file() else None,
                gvsoc_log=gvsoc_log_path if gvsoc_log_path.is_file() else None,
                layer_manifest=layer_manifest if layer_manifest is not None and layer_manifest.is_file() else None,
            )
            stage_drift_status = "ok" if drift_result.returncode == 0 else "fail"
            if drift_result.returncode != 0:
                stage_drift_error = (drift_result.stdout + drift_result.stderr).strip()
            stage_drift_log_path = stage_drift_dir / "runner.log"
            stage_drift_log_path.parent.mkdir(parents=True, exist_ok=True)
            stage_drift_log_path.write_text(
                drift_result.stdout + drift_result.stderr,
                encoding="utf-8",
            )

        if not args.skip_overlays and actual_tensor is not None:
            try:
                overlay_result = generate_overlay_for_sample(
                    sample_dir=sample_dir,
                    annotations_path=args.overlay_annotations,
                    q_scale=args.overlay_q_scale,
                    vis_threshold=args.overlay_vis_threshold,
                )
                overlay_path = overlay_result.overlay_path
                overlay_status = "ok"
                overlay_gt_x_offset = overlay_result.gt_x_offset
                overlay_gt_size_proxy = overlay_result.gt_size_proxy
            except Exception as exc:
                overlay_status = "fail"
                overlay_error = str(exc)
        elif not args.skip_overlays:
            overlay_status = "unavailable"
            overlay_error = "Overlay skipped because actual tensor output was unavailable."

        comparison["overlay_path"] = overlay_path
        comparison["overlay_status"] = overlay_status
        comparison["overlay_error"] = overlay_error
        comparison["overlay_gt_x_offset"] = overlay_gt_x_offset
        comparison["overlay_gt_size_proxy"] = overlay_gt_size_proxy
        comparison["stage_drift_dir"] = str(stage_drift_dir) if args.stage_drift else None
        comparison["stage_drift_status"] = stage_drift_status
        comparison["stage_drift_error"] = stage_drift_error
        write_json(comparison_path, comparison)
        summary_rows.append(comparison)

        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"{status.upper()} expected={expected_tensor} actual={actual_tensor} "
            f"overlay={overlay_status} stage_drift={stage_drift_status}"
        )

        if status != "pass" and args.stop_on_error:
            break

    total_elapsed = time.time() - start_time
    summary_payload = {
        "images_dir": str(images_dir),
        "results_dir": str(results_dir),
        "app_dir": str(app_dir),
        "onnx": str(onnx_path),
        "platform": args.platform,
        "overlays_enabled": not args.skip_overlays,
        "overlay_annotations": str(Path(args.overlay_annotations).expanduser()),
        "count": len(summary_rows),
        "passed": sum(1 for row in summary_rows if row["status"] == "pass"),
        "failed": sum(1 for row in summary_rows if row["status"] != "pass"),
        "elapsed_seconds": total_elapsed,
        "results": summary_rows,
    }

    summary_json_path = results_dir / "summary.json"
    summary_csv_path = results_dir / "summary.csv"
    write_json(summary_json_path, summary_payload)
    write_summary_csv(summary_csv_path, summary_rows)

    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV:  {summary_csv_path}")

    return 0 if summary_payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
EXPORTER_DIR = PROJECT_DIR / "nemo"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

from nemo.quant.pact import PACT_IntegerAdd


DEFAULT_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_x.pth"
DEFAULT_CALIB_DIR = PROJECT_DIR / "data" / "coco" / "images" / "val2017"
DEFAULT_PROBE_IMAGE = PROJECT_DIR / "training" / "hybrid_follow" / "eval_epoch_015" / "top_fn" / "01_p0.0114_000000132408.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "logs" / "hybrid_follow_val" / "determinism_checks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exporter determinism and no-debug-side-effects checks for hybrid_follow "
            "baseline and preset exports."
        )
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--probe-image", default=str(DEFAULT_PROBE_IMAGE))
    parser.add_argument("--stage", default="id", choices=["fp", "fq", "qd", "id"])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["baseline", "microblock_add_only"],
        help="Hybrid-follow export presets to validate.",
    )
    parser.add_argument(
        "--dory-manifest",
        default=None,
        help="Optional DORY artifact manifest path to checksum when available after export.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def probe_onnx(onnx_path: Path, probe_image: Path, *, height: int, width: int) -> dict[str, Any]:
    from compare_hybrid_follow_stages import preprocess_image_once, run_onnx_stage

    x_float, x_uint8 = preprocess_image_once(probe_image, height, width)
    args = SimpleNamespace(
        height=height,
        width=width,
        onnx_stage="auto",
        integer_output_scale=32768.0,
    )
    stage = run_onnx_stage(onnx_path, x_float, x_uint8, args)
    return stage.to_dict()


def summarize_dory_manifest(manifest_path: Path | None) -> dict[str, Any] | None:
    if manifest_path is None or not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = payload.get("layers") or []
    layer_entries = []
    for layer in layers:
        layer_path = Path(layer.get("path") or layer.get("golden_path") or "")
        if not layer_path.is_absolute():
            layer_path = (PROJECT_DIR / layer_path).resolve()
        layer_entries.append(
            {
                "index": int(layer.get("index", len(layer_entries))),
                "path": str(layer_path),
                "sha256": sha256_file(layer_path) if layer_path.is_file() else None,
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "layer_count": len(layers),
        "layer_entries": layer_entries,
        "digest": stable_digest(layer_entries),
    }


def run_export_variant(
    *,
    preset: str,
    variant_name: str,
    base_output_dir: Path,
    ckpt: Path,
    calib_dir: Path,
    probe_image: Path,
    args: argparse.Namespace,
    dory_manifest: Path | None,
    enable_debug: bool,
) -> dict[str, Any]:
    from export_nemo_quant_core import main as export_main

    variant_dir = base_output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = variant_dir / "hybrid_follow_dory.onnx"
    stage_report_path = variant_dir / "final_stage.txt"
    debug_dir = variant_dir / "debug_quant_drift" if enable_debug else None

    argv = [
        "--model-type",
        "hybrid_follow",
        "--ckpt",
        str(ckpt),
        "--out",
        str(onnx_path),
        "--stage",
        args.stage,
        "--stage-report",
        str(stage_report_path),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--input-channels",
        str(args.input_channels),
        "--num-classes",
        str(args.num_classes),
        "--width-mult",
        str(args.width_mult),
        "--bits",
        str(args.bits),
        "--eps-in",
        str(args.eps_in),
        "--calib-dir",
        str(calib_dir),
        "--calib-batches",
        str(args.calib_batches),
        "--hybrid-follow-export-preset",
        preset,
    ]
    if enable_debug:
        argv.extend(["--debug-quant-drift-dir", str(debug_dir)])

    return_code = export_main(argv)
    if return_code is None:
        return_code = 0
    if return_code != 0:
        raise RuntimeError(f"Export failed for {preset}/{variant_name} with return code {return_code}")

    probe = probe_onnx(onnx_path, probe_image, height=args.height, width=args.width)
    debug_report = load_json_if_exists((debug_dir or Path("__missing__")) / "debug_report.json")
    summary = {
        "variant_name": variant_name,
        "return_code": int(return_code),
        "preset": preset,
        "onnx_path": str(onnx_path),
        "stage_report_path": str(stage_report_path),
        "stage_report_value": stage_report_path.read_text(encoding="utf-8").strip(),
        "onnx_sha256": sha256_file(onnx_path),
        "probe": probe,
        "probe_digest": stable_digest(
            {
                "raw_native": probe.get("raw_native"),
                "raw_semantic": probe.get("raw_semantic"),
                "decoded": probe.get("decoded"),
                "stage_tag": probe.get("stage_tag"),
                "representation": probe.get("representation"),
            }
        ),
        "debug_dir": str(debug_dir) if debug_dir is not None else None,
        "debug_report_path": str((debug_dir / "debug_report.json")) if debug_dir is not None else None,
        "debug_report_digest": stable_digest(debug_report) if debug_report is not None else None,
        "preset_report_digest": (
            stable_digest(debug_report.get("hybrid_follow_export_preset_report"))
            if isinstance(debug_report, dict) and debug_report.get("hybrid_follow_export_preset_report") is not None
            else None
        ),
        "integerized_conv_biases_digest": (
            stable_digest(debug_report.get("integerized_conv_biases"))
            if isinstance(debug_report, dict) and debug_report.get("integerized_conv_biases") is not None
            else None
        ),
        "dory_manifest": summarize_dory_manifest(dory_manifest),
    }
    write_json(variant_dir / "run_summary.json", summary)
    return summary


def compare_fields(left: dict[str, Any], right: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    comparisons = {}
    for field_name in fields:
        comparisons[field_name] = {
            "left": left.get(field_name),
            "right": right.get(field_name),
            "match": left.get(field_name) == right.get(field_name),
        }
    return comparisons


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ckpt = Path(args.ckpt).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    probe_image = Path(args.probe_image).resolve()
    dory_manifest = Path(args.dory_manifest).resolve() if args.dory_manifest else None

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
    if not probe_image.is_file():
        raise FileNotFoundError(f"Probe image not found: {probe_image}")

    ensure_clean_dir(output_dir, overwrite=args.overwrite)

    original_get_output_eps = PACT_IntegerAdd.get_output_eps
    import export_nemo_quant  # noqa: F401

    import_side_effect_check = {
        "before_id": id(original_get_output_eps),
        "after_id": id(PACT_IntegerAdd.get_output_eps),
        "match": PACT_IntegerAdd.get_output_eps is original_get_output_eps,
    }

    from export_nemo_quant_scopes import integer_add_scale_selection_scope

    scope_check: dict[str, Any] = {}
    before_scope = PACT_IntegerAdd.get_output_eps
    with integer_add_scale_selection_scope("legacy", {"stage4.1.add": "legacy"}):
        scope_check["inside_id"] = id(PACT_IntegerAdd.get_output_eps)
        scope_check["inside_changed"] = PACT_IntegerAdd.get_output_eps is not before_scope
    after_scope = PACT_IntegerAdd.get_output_eps
    scope_check["before_id"] = id(before_scope)
    scope_check["after_id"] = id(after_scope)
    scope_check["restored"] = after_scope is before_scope

    presets_report = {}
    for preset in args.presets:
        preset_dir = output_dir / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        clean = run_export_variant(
            preset=preset,
            variant_name="clean",
            base_output_dir=preset_dir,
            ckpt=ckpt,
            calib_dir=calib_dir,
            probe_image=probe_image,
            args=args,
            dory_manifest=dory_manifest,
            enable_debug=False,
        )
        debug_a = run_export_variant(
            preset=preset,
            variant_name="debug_a",
            base_output_dir=preset_dir,
            ckpt=ckpt,
            calib_dir=calib_dir,
            probe_image=probe_image,
            args=args,
            dory_manifest=dory_manifest,
            enable_debug=True,
        )
        debug_b = run_export_variant(
            preset=preset,
            variant_name="debug_b",
            base_output_dir=preset_dir,
            ckpt=ckpt,
            calib_dir=calib_dir,
            probe_image=probe_image,
            args=args,
            dory_manifest=dory_manifest,
            enable_debug=True,
        )

        repeated_debug = compare_fields(
            debug_a,
            debug_b,
            [
                "stage_report_value",
                "onnx_sha256",
                "probe_digest",
                "preset_report_digest",
                "integerized_conv_biases_digest",
                "dory_manifest",
            ],
        )
        clean_vs_debug = compare_fields(
            clean,
            debug_a,
            [
                "stage_report_value",
                "onnx_sha256",
                "probe_digest",
                "dory_manifest",
            ],
        )
        presets_report[preset] = {
            "clean": clean,
            "debug_a": debug_a,
            "debug_b": debug_b,
            "repeated_debug_checks": repeated_debug,
            "clean_vs_debug_checks": clean_vs_debug,
            "repeated_debug_ok": all(item["match"] for item in repeated_debug.values()),
            "clean_vs_debug_ok": all(item["match"] for item in clean_vs_debug.values()),
        }

    summary = {
        "output_dir": str(output_dir),
        "import_side_effect_check": import_side_effect_check,
        "scope_restore_check": scope_check,
        "presets": presets_report,
        "all_checks_ok": bool(
            import_side_effect_check["match"]
            and scope_check["restored"]
            and all(
                report["repeated_debug_ok"] and report["clean_vs_debug_ok"]
                for report in presets_report.values()
            )
        ),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Export Determinism Checks",
                "",
                f"- Import side effects: `{import_side_effect_check['match']}`",
                f"- Scope restore: `{scope_check['restored']}`",
                *[
                    f"- {preset}: repeated_debug_ok=`{report['repeated_debug_ok']}` clean_vs_debug_ok=`{report['clean_vs_debug_ok']}`"
                    for preset, report in presets_report.items()
                ],
                f"- Overall: `{summary['all_checks_ok']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Determinism summary: {output_dir / 'summary.json'}")
    print(f"Overall checks ok: {summary['all_checks_ok']}")
    return 0 if summary["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

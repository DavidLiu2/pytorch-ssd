#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    (parent for parent in SCRIPT_DIR.parents if (parent / "models").is_dir() and (parent / "export").is_dir()),
    SCRIPT_DIR.parent,
)
EXPORT_DIR = PROJECT_DIR / "export"
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_BASE_CKPT = PROJECT_DIR / "training" / "hybrid_follow" / "hybrid_follow_best_follow_score.pth"
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
DEFAULT_BASELINE_SUMMARY = (
    PROJECT_DIR
    / "export"
    / "hybrid_follow"
    / "earliest_bad_op_loop"
    / "run4"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "export" / "hybrid_follow" / "stage4_variant_study"

VARIANT_SPECS = {
    "variant_a": {
        "stage4_variant": "plain_non_residual",
        "label": "Variant A",
        "description": "Remove the final residual add and keep stage4.1 as a plain two-conv block.",
    },
    "variant_b": {
        "stage4_variant": "single_conv_non_residual",
        "label": "Variant B",
        "description": "Replace the late residual microblock with one conv + ReLU.",
    },
    "variant_c": {
        "stage4_variant": "narrow_stage4",
        "label": "Variant C",
        "description": "Keep the late-stage topology but reduce stage4 channel width only.",
    },
}
BOUNDARY_ORDER = ["fp_to_fq", "fq_to_id", "id_to_onnx", "onnx_to_golden", "golden_to_gvsoc", None]
ALIAS_ORDER = [
    "stage4_1_conv1",
    "stage4_1_conv2",
    "stage4_1_add_pre_requant",
    "stage4_1_add_post_requant",
    "global_pool_post_requant",
    "head_input",
    "model_output",
    None,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/export/evaluate hybrid_follow stage4 architecture variants and "
            "collect float plus strict earliest-bad-op reports."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["variant_a", "variant_b", "variant_c"],
        choices=sorted(VARIANT_SPECS),
        help="Variant subset to run.",
    )
    parser.add_argument("--base-ckpt", default=str(DEFAULT_BASE_CKPT))
    parser.add_argument("--rep16-dir", default=str(DEFAULT_REP16_DIR))
    parser.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR))
    parser.add_argument(
        "--baseline-earliest-summary",
        default=str(DEFAULT_BASELINE_SUMMARY),
        help="Reference baseline summary.json used to judge whether the earliest bad op moved later.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--calib-batches", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def project_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_DIR))


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode not in allowed_returncodes:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {shlex.join(command)}. "
            f"See {log_path}."
        )
    return process.returncode


def checkpoint_metadata(train_python: Path, ckpt_path: Path) -> dict[str, Any]:
    command = [
        str(train_python),
        "-c",
        (
            "import json, torch, sys; "
            "obj=torch.load(sys.argv[1], map_location='cpu'); "
            "out={k:v for k,v in obj.items() if k!='state_dict'} if isinstance(obj, dict) else {}; "
            "print(json.dumps(out))"
        ),
        str(ckpt_path),
    ]
    result = subprocess.run(command, cwd=str(PROJECT_DIR), capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip() or "{}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_hard_case_list(hard_case_dir: Path, destination: Path) -> list[str]:
    names = sorted(path.name for path in hard_case_dir.iterdir() if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def boundary_position(boundary_key: str | None) -> int:
    return BOUNDARY_ORDER.index(boundary_key if boundary_key in BOUNDARY_ORDER else None)


def alias_position(alias: str | None) -> int:
    return ALIAS_ORDER.index(alias if alias in ALIAS_ORDER else None)


def earliest_bad_moved_later(
    baseline_earliest: dict[str, Any],
    variant_earliest: dict[str, Any],
) -> bool:
    base_boundary = boundary_position(baseline_earliest.get("boundary_key"))
    variant_boundary = boundary_position(variant_earliest.get("boundary_key"))
    if variant_boundary > base_boundary:
        return True
    if variant_boundary < base_boundary:
        return False
    return alias_position(variant_earliest.get("alias")) > alias_position(baseline_earliest.get("alias"))


def bottleneck_status(
    baseline_earliest: dict[str, Any],
    variant_earliest: dict[str, Any],
) -> str:
    if variant_earliest.get("operator_name") != "stage4.1.conv1":
        return "removed"
    baseline_drift = float((((baseline_earliest.get("local_metrics") or {}).get("drift") or {}).get("mean_abs_diff")) or 0.0)
    variant_drift = float((((variant_earliest.get("local_metrics") or {}).get("drift") or {}).get("mean_abs_diff")) or 0.0)
    if baseline_drift > 0.0 and variant_drift <= baseline_drift * 0.85:
        return "reduced"
    return "unchanged"


def summarize_result(
    *,
    variant_key: str,
    spec: dict[str, Any],
    ckpt_meta: dict[str, Any],
    earliest_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    paths: dict[str, str],
    durations_s: dict[str, float],
) -> dict[str, Any]:
    baseline = baseline_summary["baseline"]
    current = earliest_summary["baseline"]
    baseline_earliest = baseline.get("earliest_bad") or {}
    current_earliest = current.get("earliest_bad") or {}
    rep16 = ((current.get("datasets") or {}).get("rep16") or {}).get("aggregate") or {}
    hard_case = ((current.get("datasets") or {}).get("hard_case") or {}).get("aggregate") or {}
    anti = ((current.get("anti_collapse") or {}).get("onnx") or {})
    float_val = ckpt_meta.get("val_stats") or {}
    runtime_mismatch_count = int(rep16.get("runtime_mismatch_count") or 0)
    export_runtime_regression = runtime_mismatch_count == 0
    return {
        "variant": variant_key,
        "label": spec["label"],
        "description": spec["description"],
        "stage4_variant": spec["stage4_variant"],
        "checkpoint": paths["ckpt"],
        "float_val_metrics": float_val,
        "earliest_bad": current_earliest,
        "first_bad_local_drift": ((current_earliest.get("local_metrics") or {}).get("drift") or {}),
        "x_anti_collapse": anti,
        "rep16": rep16,
        "hard_case": hard_case,
        "moved_later_or_disappeared_vs_baseline": earliest_bad_moved_later(baseline_earliest, current_earliest),
        "stage4_1_conv1_bottleneck_status": bottleneck_status(baseline_earliest, current_earliest),
        "no_new_exporter_runtime_regressions": export_runtime_regression,
        "durations_s": durations_s,
        "paths": paths,
    }


def summary_markdown(
    *,
    baseline_summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    baseline = baseline_summary["baseline"]
    baseline_earliest = baseline.get("earliest_bad") or {}
    baseline_rep16 = (((baseline.get("datasets") or {}).get("rep16") or {}).get("aggregate") or {})
    baseline_hard = (((baseline.get("datasets") or {}).get("hard_case") or {}).get("aggregate") or {})
    lines = [
        "# Hybrid Follow Stage4 Variant Study",
        "",
        "## Baseline Reference",
        "",
        f"- earliest bad boundary: `{baseline_earliest.get('boundary_label')}`",
        f"- earliest bad operator: `{baseline_earliest.get('operator_name')}`",
        f"- rep16 primary score mean: `{baseline_rep16.get('primary_score_mean')}`",
        f"- hard-case primary score mean: `{baseline_hard.get('primary_score_mean')}`",
        "",
    ]
    for result in results:
        float_val = result.get("float_val_metrics") or {}
        earliest = result.get("earliest_bad") or {}
        drift = result.get("first_bad_local_drift") or {}
        anti = result.get("x_anti_collapse") or {}
        rep16 = result.get("rep16") or {}
        hard_case = result.get("hard_case") or {}
        lines.extend(
            [
                f"## {result['label']}",
                "",
                f"- stage4 variant: `{result['stage4_variant']}`",
                f"- float x_mae: `{float_val.get('x_mae')}`",
                f"- float size_mae: `{float_val.get('size_mae')}`",
                f"- float follow_score: `{float_val.get('follow_score')}`",
                f"- float control_score: `{float_val.get('control_score')}`",
                f"- earliest bad boundary: `{earliest.get('boundary_label')}`",
                f"- earliest bad operator: `{earliest.get('operator_name')}`",
                f"- first bad local mean_abs_diff: `{drift.get('mean_abs_diff')}`",
                f"- x anti-collapse sign_flip_rate: `{anti.get('sign_flip_rate')}`",
                f"- x anti-collapse corr: `{anti.get('correlation')}`",
                f"- x anti-collapse slope: `{anti.get('slope')}`",
                f"- x anti-collapse collapsed_fraction: `{anti.get('collapsed_fraction')}`",
                f"- rep16 primary score mean: `{rep16.get('primary_score_mean')}`",
                f"- hard-case primary score mean: `{hard_case.get('primary_score_mean')}`",
                f"- stage4.1.conv1 bottleneck: `{result.get('stage4_1_conv1_bottleneck_status')}`",
                f"- earliest bad moved later or disappeared vs baseline: `{result.get('moved_later_or_disappeared_vs_baseline')}`",
                f"- no new exporter/runtime regressions: `{result.get('no_new_exporter_runtime_regressions')}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    base_ckpt = Path(args.base_ckpt).expanduser().resolve()
    rep16_dir = Path(args.rep16_dir).expanduser().resolve()
    hard_case_dir = Path(args.hard_case_dir).expanduser().resolve()
    baseline_summary_path = Path(args.baseline_earliest_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_python = REPO_ROOT / "trainenv" / "bin" / "python"
    nemo_python = REPO_ROOT / "nemoenv" / "bin" / "python"
    dory_python = REPO_ROOT / "doryenv" / "bin" / "python3"

    for path in (base_ckpt, rep16_dir, hard_case_dir, baseline_summary_path, train_python, nemo_python, dory_python):
        if not path.exists():
            raise FileNotFoundError(path)

    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_case_names = make_hard_case_list(hard_case_dir, output_dir / "hard_case_names.txt")
    baseline_summary = load_json(baseline_summary_path)

    results = []
    runs_payload = []

    for variant_key in args.variants:
        spec = VARIANT_SPECS[variant_key]
        variant_dir = output_dir / variant_key
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_dir = variant_dir / "train"
        export_dir = variant_dir / "export_artifacts"
        app_dir = variant_dir / "application"
        rep16_results_dir = variant_dir / "rep16_runtime"
        earliest_dir = variant_dir / "earliest_bad"

        best_ckpt = train_dir / "hybrid_follow_best_follow_score.pth"
        run_all_log = variant_dir / "run_all.log"
        train_log = variant_dir / "train.log"
        validate_log = variant_dir / "validate_rep16.log"
        earliest_log = variant_dir / "earliest_bad.log"

        durations_s: dict[str, float] = {}

        if not args.reuse_existing or not best_ckpt.is_file():
            train_cmd = [
                str(train_python),
                str(PROJECT_DIR / "train.py"),
                "--model-type",
                "hybrid_follow",
                "--stage4-variant",
                spec["stage4_variant"],
                "--init-ckpt",
                str(base_ckpt),
                "--stage4-heads-only",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--num_workers",
                str(args.num_workers),
                "--output_dir",
                str(train_dir),
            ]
            if args.max_train_batches is not None:
                train_cmd.extend(["--max-train-batches", str(args.max_train_batches)])
            if args.max_val_batches is not None:
                train_cmd.extend(["--max-val-batches", str(args.max_val_batches)])
            started = time.time()
            run_logged(train_cmd, cwd=REPO_ROOT, log_path=train_log)
            durations_s["train"] = round(time.time() - started, 3)

        if not best_ckpt.is_file():
            raise FileNotFoundError(best_ckpt)

        dory_onnx = export_dir / "hybrid_follow_dory.onnx"
        if not args.reuse_existing or not dory_onnx.is_file():
            run_all_env = {
                "MODEL_TYPE": "hybrid_follow",
                "CKPT": str(best_ckpt),
                "CALIB_DIR": project_rel(PROJECT_DIR / "data" / "coco" / "images" / "val2017"),
                "CALIB_BATCHES": str(args.calib_batches),
                "OUT_ONNX": project_rel(export_dir / "hybrid_follow_quant.onnx"),
                "SIM_ONNX": project_rel(export_dir / "hybrid_follow_quant_sim.onnx"),
                "STAGE_REPORT": project_rel(export_dir / "hybrid_follow_final_stage.txt"),
                "DORY_CONFIG_GEN": project_rel(export_dir / "config_hybrid_follow_runtime.json"),
                "DORY_ONNX": project_rel(export_dir / "hybrid_follow_dory.onnx"),
                "DORY_NO_AFFINE_ONNX": project_rel(export_dir / "hybrid_follow_noaffine.onnx"),
                "DORY_NO_TRANSPOSE_ONNX": project_rel(export_dir / "hybrid_follow_notranspose.onnx"),
                "DORY_NO_MIN_ONNX": project_rel(export_dir / "hybrid_follow_nomin.onnx"),
                "DORY_WEIGHTS_TXT_DIR": project_rel(export_dir / "weights_txt"),
                "DORY_ARTIFACT_MANIFEST": project_rel(export_dir / "nemo_dory_artifacts.json"),
                "COMPAT_PY_REPORT": project_rel(export_dir / "model_compat_python.json"),
                "COMPAT_ONNX_REPORT": project_rel(export_dir / "model_compat_onnx.json"),
                "RAW_RESIDUAL_PATCH_REPORT": project_rel(export_dir / "gap8_raw_residual_patch_report.json"),
                "GAP8_LAYER_MANIFEST": project_rel(export_dir / "gap8_layer_manifest.json"),
                "DORY_APP_DIR": project_rel(app_dir),
                "RUN_STAGE_DRIFT": "0",
                "RUN_QUANT_POLICY_SWEEP": "0",
            }
            started = time.time()
            run_logged(["bash", str(PROJECT_DIR / "run_all.sh")], cwd=PROJECT_DIR, log_path=run_all_log, env=run_all_env)
            durations_s["export"] = round(time.time() - started, 3)

        rep16_summary = rep16_results_dir / "summary.json"
        if not args.reuse_existing or not rep16_summary.is_file():
            validate_cmd = [
                str(dory_python),
                str(PROJECT_DIR / "export" / "validate_hybrid_follow_real_images.py"),
                "--images-dir",
                str(rep16_dir),
                "--results-dir",
                project_rel(rep16_results_dir),
                "--app-dir",
                project_rel(app_dir),
                "--onnx",
                project_rel(dory_onnx),
                "--overwrite",
                "--no-stage-drift",
            ]
            started = time.time()
            validate_returncode = run_logged(
                validate_cmd,
                cwd=PROJECT_DIR,
                log_path=validate_log,
                allowed_returncodes=(0, 1),
            )
            durations_s["validate_rep16"] = round(time.time() - started, 3)
            durations_s["validate_rep16_exit_code"] = validate_returncode

        if not rep16_summary.is_file():
            raise FileNotFoundError(rep16_summary)

        earliest_summary_path = earliest_dir / "summary.json"
        if not args.reuse_existing or not earliest_summary_path.is_file():
            earliest_cmd = [
                str(nemo_python),
                str(SCRIPT_DIR / "run_hybrid_follow_earliest_bad_op.py"),
                "--ckpt",
                str(best_ckpt),
                "--calib-dir",
                str(PROJECT_DIR / "data" / "coco" / "images" / "val2017"),
                "--rep16-dir",
                str(rep16_dir),
                "--application-summary",
                str(rep16_summary),
                "--layer-manifest",
                str(export_dir / "gap8_layer_manifest.json"),
                "--hard-case-list",
                str(output_dir / "hard_case_names.txt"),
                "--output-dir",
                str(earliest_dir),
                "--overwrite",
                "--skip-focused-qat",
                "--report-only",
            ]
            started = time.time()
            run_logged(earliest_cmd, cwd=PROJECT_DIR, log_path=earliest_log)
            durations_s["earliest_bad"] = round(time.time() - started, 3)

        ckpt_meta = checkpoint_metadata(train_python, best_ckpt)
        earliest_summary = load_json(earliest_summary_path)
        paths = {
            "ckpt": str(best_ckpt),
            "dory_onnx": str(dory_onnx),
            "application_dir": str(app_dir),
            "rep16_summary": str(rep16_summary),
            "earliest_summary": str(earliest_summary_path),
            "train_log": str(train_log),
            "run_all_log": str(run_all_log),
            "validate_log": str(validate_log),
            "earliest_log": str(earliest_log),
        }
        result = summarize_result(
            variant_key=variant_key,
            spec=spec,
            ckpt_meta=ckpt_meta,
            earliest_summary=earliest_summary,
            baseline_summary=baseline_summary,
            paths=paths,
            durations_s=durations_s,
        )
        results.append(result)
        runs_payload.append(
            {
                "variant": variant_key,
                "stage4_variant": spec["stage4_variant"],
                "durations_s": durations_s,
                "hard_case_names": hard_case_names,
                "paths": paths,
            }
        )

    summary_payload = {
        "args": vars(args),
        "baseline_reference_summary": str(baseline_summary_path),
        "hard_case_names": hard_case_names,
        "runs": runs_payload,
        "results": results,
    }
    write_json(output_dir / "summary.json", summary_payload)
    write_markdown(
        output_dir / "summary.md",
        summary_markdown(
            baseline_summary=baseline_summary,
            results=results,
        ),
    )
    print(f"Wrote variant study summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

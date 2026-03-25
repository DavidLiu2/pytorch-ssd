#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


TRACKED_PATCH_FILES = (
    "application/inc/pulp_nn_kernels.h",
    "application/src/network.c",
    "application/src/pulp_nn_add.c",
    "application/src/pulp_nn_conv_Ho_parallel.c",
    "application/src/pulp_nn_linear_out_32.c",
    "application/src/pulp_nn_matmul.c",
    "application/src/pulp_nn_utils.c",
    "application/src/Convolution1.c",
    "application/src/Convolution3.c",
    "application/src/Convolution6.c",
    "application/src/Convolution8.c",
    "application/src/Convolution10.c",
    "application/src/Convolution13.c",
    "application/src/Convolution15.c",
    "application/src/Convolution17.c",
    "application/src/Convolution20.c",
    "application/src/Convolution22.c",
    "application/src/Convolution24.c",
    "application/src/Convolution27.c",
    "application/src/ReluQAddition4.c",
    "application/src/ReluQAddition7.c",
    "application/src/ReluQAddition11.c",
    "application/src/ReluQAddition14.c",
    "application/src/ReluQAddition18.c",
    "application/src/ReluQAddition21.c",
    "application/src/ReluQAddition25.c",
    "application/src/ReluQAddition28.c",
)


RAW_WRAPPERS = {
    4: ("pulp_nn_add_raw_i32_u8", "const int32_t *", "24576u"),
    11: ("pulp_nn_add_raw_i32_u8", "const int32_t *", "8192u"),
    18: ("pulp_nn_add_raw_i32_u8", "const int32_t *", "4096u"),
    25: ("pulp_nn_add_raw_i32_u8", "const int32_t *", "1280u"),
    7: ("pulp_nn_add_raw_i32_u8_mixed", "const uint8_t *", "24576u"),
    14: ("pulp_nn_add_raw_i32_u8_mixed", "const uint8_t *", "8192u"),
    21: ("pulp_nn_add_raw_i32_u8_mixed", "const uint8_t *", "4096u"),
    28: ("pulp_nn_add_raw_i32_u8_mixed", "const uint8_t *", "1280u"),
}


def wrapper_source(layer_id: int, helper_name: str, x2_type: str, num_elements: str) -> str:
    x2_cast = "const int32_t *" if x2_type == "const int32_t *" else "const uint8_t *"
    x2_decl = "const int32_t *x2" if x2_type == "const int32_t *" else "const uint8_t *x2"
    return (
        f'#include "ReluQAddition{layer_id}.h"\n'
        '#include "pulp.h"\n'
        '#include "pmsis.h"\n'
        '#include "dory_get_tile.h"\n'
        '#include "dory_dma.h"\n'
        '#include "pulp_nn_kernels.h"\n'
        '\n'
        f"void ReluQAddition{layer_id}(void *args) {{\n"
        "  unsigned int *real_arg = (unsigned int *) args;\n"
        "  const int32_t *x = (const int32_t *) real_arg[3];\n"
        f"  {x2_decl} = ({x2_cast}) real_arg[4];\n"
        "  uint8_t *y = (uint8_t *) real_arg[5];\n"
        "  unsigned int out_mult_in = (unsigned int) real_arg[9];\n"
        "  unsigned int out_shift_in = (unsigned int) real_arg[10];\n"
        "\n"
        f'  {helper_name}(x, x2, y, 32, 32, 6, out_mult_in, out_shift_in, {num_elements}, "ReluQAddition{layer_id}");\n'
        "}\n"
    )


def repo_root_from_application_dir(app_dir: Path) -> Path:
    repo_root = app_dir.parent.resolve()
    if not (repo_root / ".git").exists():
        raise FileNotFoundError(f"Could not locate repo root from application dir: {app_dir}")
    return repo_root


def _git_show_head(repo_root: Path, rel_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def restore_tracked_patch_files(app_dir: Path) -> list[str]:
    repo_root = repo_root_from_application_dir(app_dir)
    changed = []
    for rel_path in TRACKED_PATCH_FILES:
        path = repo_root / rel_path
        expected = _git_show_head(repo_root, rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing generated runtime file: {path}")
        current = path.read_text(encoding="utf-8")
        if current != expected:
            path.write_text(expected, encoding="utf-8")
            changed.append(str(path))
    return changed


def rewrite_add_wrappers(app_dir: Path) -> list[str]:
    changed = []
    for layer_id, (helper_name, x2_type, num_elements) in sorted(RAW_WRAPPERS.items()):
        path = app_dir / "src" / f"ReluQAddition{layer_id}.c"
        expected = wrapper_source(layer_id, helper_name, x2_type, num_elements)
        if not path.exists():
            raise FileNotFoundError(f"Missing generated wrapper: {path}")
        current = path.read_text(encoding="utf-8")
        if current != expected:
            path.write_bytes(expected.replace("\n", "\r\n").encode("utf-8"))
            changed.append(str(path))
    return changed


def verify_file_contains(path: Path, needles: list[str]) -> list[str]:
    if not path.exists():
        return [f"missing:{path}"]
    text = path.read_text(encoding="utf-8")
    errors = []
    for needle in needles:
        if needle not in text:
            errors.append(f"{path}:{needle}")
    return errors


def verify_runtime_patch_set(app_dir: Path) -> dict:
    src_dir = app_dir / "src"
    inc_dir = app_dir / "inc"

    errors: list[str] = []

    errors.extend(
        verify_file_contains(
            src_dir / "network.c",
            ["unsigned int args[5];"],
        )
    )
    errors.extend(
        verify_file_contains(
            src_dir / "pulp_nn_add.c",
            [
                "void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8(",
                "void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8_mixed(",
            ],
        )
    )
    errors.extend(
        verify_file_contains(
            inc_dir / "pulp_nn_kernels.h",
            [
                "void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8(",
                "void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8_mixed(",
            ],
        )
    )
    errors.extend(
        verify_file_contains(
            src_dir / "pulp_nn_utils.c",
            ["int32_t x = (m * phi) >> d;"],
        )
    )
    errors.extend(
        verify_file_contains(
            src_dir / "pulp_nn_matmul.c",
            ["const int32_t * bias,", "int output_is_i32 = (flag_relu == 0 && flag_batch_norm == 0);"],
        )
    )
    errors.extend(
        verify_file_contains(
            src_dir / "pulp_nn_linear_out_32.c",
            ["const int32_t *bias,"],
        )
    )
    errors.extend(
        verify_file_contains(
            src_dir / "pulp_nn_conv_Ho_parallel.c",
            ["const int32_t * bias,", "int output_is_i32 = (flag_relu == 0 && flag_batch_norm == 0);"],
        )
    )

    for layer_id, (helper_name, _x2_type, _num_elements) in sorted(RAW_WRAPPERS.items()):
        errors.extend(
            verify_file_contains(
                src_dir / f"ReluQAddition{layer_id}.c",
                [helper_name, f"ReluQAddition{layer_id}"],
            )
        )

    for path in sorted(src_dir.glob("Convolution*.c")):
        errors.extend(
            verify_file_contains(
                path,
                ["(const int32_t *) b", "pulp_nn_conv_Ho_parallel("],
            )
        )

    return {
        "application_dir": str(app_dir),
        "raw_wrapper_layers": sorted(RAW_WRAPPERS.keys()),
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reapply and verify the hybrid_follow GAP8 raw-residual generated-code patch set."
    )
    parser.add_argument(
        "--application-dir",
        type=Path,
        default=Path("application"),
        help="Generated application directory to patch and verify.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify the raw-residual patch set without rewriting the ReluQAddition wrappers.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    app_dir = args.application_dir.resolve()
    changed_files: list[str] = []
    if not args.check_only:
        changed_files.extend(restore_tracked_patch_files(app_dir))
        changed_files.extend(rewrite_add_wrappers(app_dir))
        changed_files = sorted(set(changed_files))

    report = verify_runtime_patch_set(app_dir)
    report["changed_files"] = changed_files

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if report["ok"]:
        print(
            "[reapply_gap8_raw_residual_patches] OK: raw-residual GAP8 patch set is present."
        )
        if changed_files:
            print(
                "[reapply_gap8_raw_residual_patches] Rewrote wrappers: "
                + ", ".join(Path(path).name for path in changed_files)
            )
        else:
            print("[reapply_gap8_raw_residual_patches] Wrapper sources were already up to date.")
        return 0

    print("[reapply_gap8_raw_residual_patches] ERROR: required runtime patch fragments are missing:")
    for item in report["errors"]:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

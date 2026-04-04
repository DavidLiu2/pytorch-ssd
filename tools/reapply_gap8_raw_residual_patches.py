#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TEMPLATE_APPLICATION_DIR = PROJECT_DIR / "application"


PATCH_TEMPLATE_FILES = (
    "inc/app_config.h",
    "inc/pulp_nn_kernels.h",
    "inc/printf.h",
    "src/pulp_nn_add.c",
    "src/pulp_nn_conv_Ho_parallel.c",
    "src/pulp_nn_linear_out_32.c",
    "src/pulp_nn_matmul.c",
    "src/pulp_nn_utils.c",
)

OPTIONAL_GENERATED_TEMPLATE_FILES = (
    "src/Convolution1.c",
    "src/Convolution3.c",
    "src/Convolution6.c",
    "src/Convolution8.c",
    "src/Convolution10.c",
    "src/Convolution13.c",
    "src/Convolution15.c",
    "src/Convolution17.c",
    "src/Convolution20.c",
    "src/Convolution22.c",
    "src/Convolution24.c",
    "src/Convolution27.c",
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


def restore_template_patch_files(app_dir: Path, template_app_dir: Path) -> list[str]:
    if not template_app_dir.is_dir():
        raise FileNotFoundError(f"Template application directory not found: {template_app_dir}")
    changed = []
    for rel_path in PATCH_TEMPLATE_FILES:
        source_path = template_app_dir / rel_path
        target_path = app_dir / rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing template runtime file: {source_path}")
        expected = source_path.read_bytes()
        current = target_path.read_bytes() if target_path.exists() else None
        if current != expected:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(expected)
            changed.append(str(target_path))
    for rel_path in OPTIONAL_GENERATED_TEMPLATE_FILES:
        source_path = template_app_dir / rel_path
        target_path = app_dir / rel_path
        if not source_path.exists() or not target_path.exists():
            continue
        expected = source_path.read_bytes()
        current = target_path.read_bytes()
        if current != expected:
            target_path.write_bytes(expected)
            changed.append(str(target_path))
    return changed


def patch_network_source(app_dir: Path) -> list[str]:
    path = app_dir / "src" / "network.c"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated network source: {path}")
    current = path.read_text(encoding="utf-8")
    updated = current.replace("unsigned int args[4];", "unsigned int args[5];")
    if updated == current:
        return []
    path.write_bytes(updated.replace("\n", "\r\n").encode("utf-8"))
    return [str(path)]


def rewrite_add_wrappers(app_dir: Path) -> list[str]:
    changed = []
    for layer_id, (helper_name, x2_type, num_elements) in sorted(RAW_WRAPPERS.items()):
        path = app_dir / "src" / f"ReluQAddition{layer_id}.c"
        if not path.exists():
            continue
        expected = wrapper_source(layer_id, helper_name, x2_type, num_elements)
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
    active_raw_wrapper_layers = sorted(
        layer_id
        for layer_id in RAW_WRAPPERS
        if (src_dir / f"ReluQAddition{layer_id}.c").exists()
    )

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

    for layer_id in active_raw_wrapper_layers:
        helper_name, _x2_type, _num_elements = RAW_WRAPPERS[layer_id]
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
        "raw_wrapper_layers": active_raw_wrapper_layers,
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
        "--template-application-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_APPLICATION_DIR,
        help="Patched application directory used as the source template for runtime files.",
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
    template_app_dir = args.template_application_dir.resolve()
    changed_files: list[str] = []
    if not args.check_only:
        changed_files.extend(restore_template_patch_files(app_dir, template_app_dir))
        changed_files.extend(patch_network_source(app_dir))
        changed_files.extend(rewrite_add_wrappers(app_dir))
        changed_files = sorted(set(changed_files))

    report = verify_runtime_patch_set(app_dir)
    report["changed_files"] = changed_files
    report["template_application_dir"] = str(template_app_dir)

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

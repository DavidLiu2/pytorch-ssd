#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "/* GAP8_INT64_REQUANT_PATCH */"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a generated GAP8 app so pulp_nn quant helpers use int64 "
            "intermediates before the requant right-shift."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app-dir", help="Path to the generated GAP8 app directory.")
    group.add_argument("--pulp-nn-utils-c", help="Path to generated src/pulp_nn_utils.c.")
    return parser.parse_args()


def resolve_target(args: argparse.Namespace) -> Path:
    if args.app_dir:
        path = Path(args.app_dir).expanduser().resolve() / "src" / "pulp_nn_utils.c"
    else:
        path = Path(args.pulp_nn_utils_c).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pulp_nn_utils.c not found: {path}")
    return path


def build_replacement() -> str:
    return f"""
{MARKER}
static inline uint8_t gap8_clip_u8_from_i64(int64_t value) {{
  if (value <= 0) {{
    return 0;
  }}
  if (value >= 255) {{
    return 255;
  }}
  return (uint8_t)value;
}}

static inline uint8_t gap8_clip_u4_from_i64(int64_t value) {{
  if (value <= 0) {{
    return 0;
  }}
  if (value >= 15) {{
    return 15;
  }}
  return (uint8_t)value;
}}

static inline uint8_t gap8_clip_u2_from_i64(int64_t value) {{
  if (value <= 0) {{
    return 0;
  }}
  if (value >= 3) {{
    return 3;
  }}
  return (uint8_t)value;
}}
"""


def apply_patch_to_text(text: str) -> str:
    if MARKER in text:
        return text

    start_anchor = '#define clip8(x) __builtin_pulp_clipu_r(x, 255)\n'
    start = text.find(start_anchor)
    if start < 0:
        raise RuntimeError("Could not locate the clip8 anchor in pulp_nn_utils.c")
    text = text[: start + len(start_anchor)] + build_replacement() + text[start + len(start_anchor) :]

    replacements = {
        "pulp_nn_bn_quant_u8": """
uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u8 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = ((int64_t)k * (int64_t)phi) + (int64_t)lambda;
  int64_t x = integer_image_phi >> d;
  return gap8_clip_u8_from_i64(x);
}
""",
        "pulp_nn_add_quant_u8": """
uint8_t pulp_nn_add_quant_u8 (
  uint8_t pix1,
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  int8_t  d
) {
  int64_t integer_image = ((int64_t)pix1 * (int64_t)m1) + ((int64_t)pix2 * (int64_t)m2);
  int64_t x = integer_image >> d;
  return gap8_clip_u8_from_i64(x);
}
""",
        "pulp_nn_quant_u8": """
uint8_t __attribute__((always_inline)) pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int64_t x = ((int64_t)m * (int64_t)phi) >> d;
  return gap8_clip_u8_from_i64(x);
}
""",
        "pulp_nn_bn_quant_u4": """
uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u4 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = ((int64_t)k * (int64_t)phi) + (int64_t)lambda;
  int64_t x = integer_image_phi >> d;
  return gap8_clip_u4_from_i64(x);
}
""",
        "pulp_nn_quant_u4": """
uint8_t __attribute__((always_inline)) pulp_nn_quant_u4(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int64_t x = ((int64_t)m * (int64_t)phi) >> d;
  return gap8_clip_u4_from_i64(x);
}
""",
        "pulp_nn_bn_quant_u2": """
uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u2 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int64_t integer_image_phi = ((int64_t)k * (int64_t)phi) + (int64_t)lambda;
  int64_t x = integer_image_phi >> d;
  return gap8_clip_u2_from_i64(x);
}
""",
        "pulp_nn_quant_u2": """
uint8_t __attribute__((always_inline)) pulp_nn_quant_u2(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int64_t x = ((int64_t)m * (int64_t)phi) >> d;
  return gap8_clip_u2_from_i64(x);
}
""",
    }

    for name, replacement in replacements.items():
        pattern = re.compile(
            rf"uint8_t(?:\s+__attribute__\(\(always_inline\)\))?\s+{re.escape(name)}\s*\([^)]*\)\s*\{{.*?\n\}}",
            re.S,
        )
        new_text, count = pattern.subn(replacement.strip(), text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch function {name} in pulp_nn_utils.c")
        text = new_text
    return text


def main() -> int:
    target = resolve_target(parse_args())
    original = target.read_text(encoding="utf-8")
    patched = apply_patch_to_text(original)
    if patched != original:
        target.write_text(patched, encoding="utf-8")
        print(f"Patched int64 GAP8 requant helpers into {target}")
    else:
        print(f"Int64 GAP8 requant helpers already present in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

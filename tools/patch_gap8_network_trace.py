#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch a generated GAP8 network.c file to emit LAYER_BYTES runtime traces."
    )
    parser.add_argument("--network-c", required=True, help="Path to generated network.c.")
    parser.add_argument("--bytes-per-line", type=int, default=64)
    return parser.parse_args()


def build_helper(bytes_per_line: int) -> str:
    return f"""
#ifndef APP_TRACE_LAYER_OUTPUT_BYTES_PER_LINE
#define APP_TRACE_LAYER_OUTPUT_BYTES_PER_LINE ({bytes_per_line}u)
#endif

static void app_trace_layer_bytes(int layer_index, const char *layer_name, const uint8_t *data, size_t size) {{
  uint32_t sum = 0;
  uint32_t hash = 2166136261u;
  for (size_t idx = 0; idx < size; ++idx) {{
    uint32_t byte = (uint32_t)data[idx];
    sum += byte;
    hash ^= byte;
    hash *= 16777619u;
  }}
  printf("LAYER_BYTES_BEGIN %d %s bytes=%u sum_mod32=%u hash32=%u\\n",
         layer_index, layer_name, (unsigned int) size, sum, hash);
  for (size_t offset = 0; offset < size; offset += APP_TRACE_LAYER_OUTPUT_BYTES_PER_LINE) {{
    size_t line_end = offset + APP_TRACE_LAYER_OUTPUT_BYTES_PER_LINE;
    if (line_end > size) {{
      line_end = size;
    }}
    printf("LAYER_BYTES %d %s offset=%u", layer_index, layer_name, (unsigned int) offset);
    for (size_t pos = offset; pos < line_end; ++pos) {{
      printf(" %u", (unsigned int) data[pos]);
    }}
    printf("\\n");
  }}
  printf("LAYER_BYTES_END %d %s\\n", layer_index, layer_name);
}}
"""


def main() -> int:
    args = parse_args()
    path = Path(args.network_c).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"network.c not found: {path}")

    text = path.read_text(encoding="utf-8")
    helper_anchor = '#include <string.h>\n'
    if "LAYER_BYTES_BEGIN" not in text:
        if helper_anchor not in text:
            raise RuntimeError(f"Could not find helper anchor in {path}")
        text = text.replace(helper_anchor, helper_anchor + build_helper(int(args.bytes_per_line)), 1)

    trace_call = (
        '    app_trace_layer_bytes(i, Layers_name[i],\n'
        '                          (const uint8_t *) (L3_output_layers[i] == 1 ? L3_output : L2_output),\n'
        '                          activations_out_size[i]);\n'
    )
    checksum_block = (
        '    if (L3_output_layers[i]==1) {\n'
        '      printf("Output in L3. Expected checksum: %d\\n", activations_out_checksum[i][exec]);\n'
        '    } else {\n'
        '      checksum(i + 1 < 9 ? "L2 output" : "final output",\n'
        '               L2_output, activations_out_size[i], activations_out_checksum[i][exec]);\n'
        '    }\n'
    )
    if trace_call not in text:
        if checksum_block not in text:
            raise RuntimeError(f"Could not find checksum block in {path}")
        text = text.replace(checksum_block, checksum_block + trace_call, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Patched runtime layer trace into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

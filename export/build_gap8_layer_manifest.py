#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join DORY golden layer artifacts with generated GAP8 network layer order "
            "so host-side runtime traces can be decoded deterministically."
        )
    )
    parser.add_argument("--dory-manifest", required=True, help="Path to nemo_dory_artifacts.json.")
    parser.add_argument("--network-header", required=True, help="Path to generated application/inc/network.h.")
    parser.add_argument("--output-json", required=True, help="Output combined layer manifest JSON.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_string_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}\[\d+\]\s*=\s*\{{(.*?)\}};", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not find array '{name}' in network header.")
    return re.findall(r'"([^"]+)"', match.group(1))


def parse_int_array(text: str, name: str) -> list[int]:
    match = re.search(rf"{re.escape(name)}\[\d+\]\s*=\s*\{{(.*?)\}};", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not find array '{name}' in network header.")
    return [int(token) for token in re.findall(r"-?\d+", match.group(1))]


def infer_element_count(path_value: str | None) -> int:
    if not path_value:
        return 0
    raw = str(path_value)
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        win_tail = raw[7:].replace("/", "\\")
        raw = f"{drive}:\\{win_tail}"
    path = Path(raw)
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for _token in line.replace(",", " ").split():
            count += 1
    return count


def main() -> int:
    args = parse_args()
    dory_manifest_path = Path(args.dory_manifest).resolve()
    network_header_path = Path(args.network_header).resolve()
    output_path = Path(args.output_json).resolve()

    dory_manifest = read_json(dory_manifest_path)
    header_text = network_header_path.read_text(encoding="utf-8")

    layer_names = parse_string_array(header_text, "Layers_name")
    activation_out_sizes = parse_int_array(header_text, "activations_out_size")
    dory_layers = dory_manifest.get("layers") or [
        {
            "index": idx,
            "path": path,
        }
        for idx, path in enumerate(dory_manifest.get("golden_activations") or [])
    ]

    if len(layer_names) != len(activation_out_sizes):
        raise RuntimeError(
            f"network.h layer metadata mismatch: names={len(layer_names)} bytes={len(activation_out_sizes)}"
        )
    if len(dory_layers) != len(layer_names):
        raise RuntimeError(
            f"Layer count mismatch between DORY manifest ({len(dory_layers)}) and network.h ({len(layer_names)})"
        )

    layers = []
    for idx, (layer_name, runtime_byte_count, dory_layer) in enumerate(
        zip(layer_names, activation_out_sizes, dory_layers)
    ):
        golden_path = dory_layer.get("path")
        element_count = int(dory_layer.get("element_count") or 0)
        if element_count <= 0:
            element_count = infer_element_count(golden_path)
        if element_count <= 0:
            raise RuntimeError(f"Invalid DORY element_count for layer {idx}: {dory_layer}")
        if runtime_byte_count % element_count != 0:
            raise RuntimeError(
                f"Cannot derive element width for layer {idx} ({layer_name}): "
                f"runtime bytes={runtime_byte_count} element_count={element_count}"
            )
        element_width_bytes = int(runtime_byte_count // element_count)
        if element_width_bytes == 4:
            runtime_dtype = "int32_le"
        elif element_width_bytes == 1:
            runtime_dtype = "uint8"
        elif element_width_bytes == 2:
            runtime_dtype = "int16_le"
        else:
            runtime_dtype = f"bytes_{element_width_bytes}"
        layers.append(
            {
                "index": idx,
                "layer_name": layer_name,
                "runtime_byte_count": int(runtime_byte_count),
                "element_count": element_count,
                "element_width_bytes": element_width_bytes,
                "runtime_dtype": runtime_dtype,
                "golden_path": golden_path,
                "tensor_name": dory_layer.get("tensor_name"),
                "shape": dory_layer.get("shape"),
                "original_dtype": dory_layer.get("original_dtype"),
                "original_itemsize_bytes": dory_layer.get("original_itemsize_bytes"),
            }
        )

    payload = {
        "dory_manifest": str(dory_manifest_path),
        "network_header": str(network_header_path),
        "num_layers": len(layers),
        "layers": layers,
    }
    write_json(output_path, payload)
    print(f"GAP8 layer manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

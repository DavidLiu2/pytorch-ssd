#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GVSOC LAYER_BYTES dumps against a GAP8 layer manifest and "
            "write a layer-by-layer runtime drift report."
        )
    )
    parser.add_argument("--gvsoc-log", required=True, help="Path to the GVSOC run log with LAYER_BYTES markers.")
    parser.add_argument("--layer-manifest", required=True, help="Path to the GAP8 layer manifest JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory where the comparison report should be written.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_numeric_artifact(path: Path) -> list[int]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "values" in payload:
            payload = payload["values"]
        return [int(value) for value in payload]

    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    return values


LAYER_BEGIN_RE = re.compile(
    r"^LAYER_BYTES_BEGIN\s+(\d+)\s+(\S+)\s+bytes=(\d+)\s+sum_mod32=(\d+)\s+hash32=(\d+)$"
)
LAYER_LINE_RE = re.compile(r"^LAYER_BYTES\s+(\d+)\s+(\S+)\s+offset=(\d+)(.*)$")
LAYER_END_RE = re.compile(r"^LAYER_BYTES_END\s+(\d+)\s+(\S+)$")


def parse_gvsoc_layer_bytes(log_path: Path) -> dict[int, dict[str, Any]]:
    layers: dict[int, dict[str, Any]] = {}
    current_index: int | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        begin_match = LAYER_BEGIN_RE.match(line)
        if begin_match:
            current_index = int(begin_match.group(1))
            layers[current_index] = {
                "index": current_index,
                "layer_name": begin_match.group(2),
                "byte_count": int(begin_match.group(3)),
                "sum_mod32": int(begin_match.group(4)),
                "hash32": int(begin_match.group(5)),
                "raw_bytes": [],
            }
            continue

        end_match = LAYER_END_RE.match(line)
        if end_match:
            current_index = None
            continue

        line_match = LAYER_LINE_RE.match(line)
        if line_match:
            index = int(line_match.group(1))
            payload = line_match.group(4).strip()
            if index not in layers:
                raise RuntimeError(f"Found LAYER_BYTES payload for missing layer index {index} in {log_path}")
            if payload:
                layers[index]["raw_bytes"].extend(int(token) for token in payload.split())

    for index, payload in layers.items():
        expected = int(payload["byte_count"])
        actual = len(payload["raw_bytes"])
        if actual != expected:
            raise RuntimeError(
                f"GVSOC layer dump size mismatch for layer {index}: expected {expected}, got {actual}"
            )
    return layers


def decode_runtime_layer_values(raw_bytes: list[int], element_width_bytes: int) -> np.ndarray:
    data = bytes(int(value) & 0xFF for value in raw_bytes)
    if element_width_bytes == 1:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if element_width_bytes == 2:
        return np.frombuffer(data, dtype="<i2").astype(np.int64)
    if element_width_bytes == 4:
        return np.frombuffer(data, dtype="<i4").astype(np.int64)
    raise ValueError(f"Unsupported runtime element width: {element_width_bytes}")


def compare_runtime_layers(
    *,
    gvsoc_log_path: Path,
    layer_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(layer_manifest_path.read_text(encoding="utf-8"))
    gvsoc_layers = parse_gvsoc_layer_bytes(gvsoc_log_path)
    reports: list[dict[str, Any]] = []
    first_divergent: dict[str, Any] | None = None

    for layer in manifest.get("layers") or []:
        index = int(layer["index"])
        gvsoc_payload = gvsoc_layers.get(index)
        if gvsoc_payload is None:
            reports.append(
                {
                    "index": index,
                    "layer_name": layer.get("layer_name"),
                    "status": "missing",
                    "reason": "No GVSOC layer dump found for this layer index.",
                }
            )
            continue

        golden_values = np.asarray(parse_numeric_artifact(Path(layer["golden_path"])), dtype=np.int64).reshape(-1)
        runtime_values = decode_runtime_layer_values(
            gvsoc_payload["raw_bytes"],
            int(layer["element_width_bytes"]),
        ).reshape(-1)

        if runtime_values.shape != golden_values.shape:
            status = "shape_mismatch"
            diff_report: dict[str, Any] = {
                "expected_shape": [int(golden_values.size)],
                "actual_shape": [int(runtime_values.size)],
            }
        else:
            diff = np.abs(runtime_values.astype(np.float64) - golden_values.astype(np.float64))
            cosine = None
            runtime_norm = float(np.linalg.norm(runtime_values.astype(np.float64)))
            golden_norm = float(np.linalg.norm(golden_values.astype(np.float64)))
            if runtime_norm > 0.0 and golden_norm > 0.0:
                cosine = float(
                    np.dot(runtime_values.astype(np.float64), golden_values.astype(np.float64))
                    / (runtime_norm * golden_norm)
                )
            diff_report = {
                "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                "cosine_similarity": cosine,
                "exact_match": bool(np.array_equal(runtime_values, golden_values)),
            }
            status = "ok" if diff_report["exact_match"] else "warn"
            if first_divergent is None and status == "warn":
                first_divergent = {
                    "index": index,
                    "layer_name": layer.get("layer_name"),
                    "tensor_name": layer.get("tensor_name"),
                    **diff_report,
                }

        layer_report = {
            "index": index,
            "layer_name": layer.get("layer_name"),
            "tensor_name": layer.get("tensor_name"),
            "status": status,
            "element_count": int(layer["element_count"]),
            "element_width_bytes": int(layer["element_width_bytes"]),
            "golden_path": layer.get("golden_path"),
            "gvsoc_byte_count": int(gvsoc_payload["byte_count"]),
            "gvsoc_sum_mod32": int(gvsoc_payload["sum_mod32"]),
            "gvsoc_hash32": int(gvsoc_payload["hash32"]),
            "compare": diff_report,
        }
        reports.append(layer_report)

        layer_dir = output_dir / "runtime_layers" / f"{index:02d}_{layer.get('layer_name')}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        write_json(layer_dir / "report.json", layer_report)
        write_json(
            layer_dir / "gvsoc_values.json",
            {
                "index": index,
                "layer_name": layer.get("layer_name"),
                "values": runtime_values.tolist(),
            },
        )

    summary = {
        "gvsoc_log": str(gvsoc_log_path),
        "layer_manifest": str(layer_manifest_path),
        "layer_count": len(reports),
        "first_divergent_layer": first_divergent,
        "layers": reports,
    }
    write_json(output_dir / "runtime_layer_compare.json", summary)
    lines = [
        "# Runtime Layer Compare",
        "",
        f"- GVSOC log: `{gvsoc_log_path}`",
        f"- Layer manifest: `{layer_manifest_path}`",
        "",
    ]
    if first_divergent is None:
        lines.append("- First divergent layer: none")
    else:
        lines.append(
            "- First divergent layer: `{} {}` mean_abs_diff=`{:.6f}` max_abs_diff=`{:.6f}`".format(
                first_divergent["index"],
                first_divergent["layer_name"],
                float(first_divergent["mean_abs_diff"]),
                float(first_divergent["max_abs_diff"]),
            )
        )
    lines.extend(["", "| idx | layer | status | mean_abs | max_abs |", "| --- | --- | --- | ---: | ---: |"])
    for report in reports:
        compare = report.get("compare") or {}
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                report["index"],
                report.get("layer_name"),
                report["status"],
                "{:.6f}".format(float(compare.get("mean_abs_diff"))) if compare.get("mean_abs_diff") is not None else "n/a",
                "{:.6f}".format(float(compare.get("max_abs_diff"))) if compare.get("max_abs_diff") is not None else "n/a",
            )
        )
    (output_dir / "runtime_layer_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    gvsoc_log_path = Path(args.gvsoc_log).expanduser().resolve()
    layer_manifest_path = Path(args.layer_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not gvsoc_log_path.is_file():
        raise FileNotFoundError(f"GVSOC log not found: {gvsoc_log_path}")
    if not layer_manifest_path.is_file():
        raise FileNotFoundError(f"Layer manifest not found: {layer_manifest_path}")

    summary = compare_runtime_layers(
        gvsoc_log_path=gvsoc_log_path,
        layer_manifest_path=layer_manifest_path,
        output_dir=output_dir,
    )
    first = summary.get("first_divergent_layer")
    if first is None:
        print("Runtime layer compare: no divergent layer found.")
    else:
        print(
            "Runtime layer compare: first divergent layer "
            f"{first['index']} {first['layer_name']} "
            f"(mean_abs_diff={float(first['mean_abs_diff']):.6f}, "
            f"max_abs_diff={float(first['max_abs_diff']):.6f})"
        )
    print(f"Runtime layer compare report: {output_dir / 'runtime_layer_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

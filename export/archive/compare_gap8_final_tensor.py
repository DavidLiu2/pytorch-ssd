#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


FINAL_BEGIN_RE = re.compile(r"^FINAL_TENSOR_I32_BEGIN\s+(\w+)\s+count=(\d+)$")
FINAL_LINE_RE = re.compile(r"^FINAL_TENSOR_I32\s+(\w+)(.*)$")
FINAL_END_RE = re.compile(r"^FINAL_TENSOR_I32_END\s+(\w+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a GVSOC FINAL_TENSOR_I32 dump against a golden output.txt."
    )
    parser.add_argument("--gvsoc-log", required=True, help="Path to the run log.")
    parser.add_argument("--expected-output", required=True, help="Path to the golden output.txt.")
    parser.add_argument("--label", default="final", help="Tensor label to compare.")
    parser.add_argument("--count", type=int, default=3, help="Expected int32 element count.")
    return parser.parse_args()


def parse_int_file(path: Path) -> list[int]:
    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    return values


def parse_tensor_from_log(path: Path, label: str) -> tuple[int, list[int]]:
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
        raise RuntimeError(f"No FINAL_TENSOR_I32 block found for label '{label}' in {path}")
    return expected_count, values


def summarize(name: str, values: list[int]) -> str:
    if not values:
        return f"{name}: count=0"
    return (
        f"{name}: count={len(values)} sum={sum(values)} "
        f"min={min(values)} max={max(values)} values={values}"
    )


def main() -> int:
    args = parse_args()
    log_path = Path(args.gvsoc_log).resolve()
    expected_path = Path(args.expected_output).resolve()

    expected = parse_int_file(expected_path)
    logged_count, actual = parse_tensor_from_log(log_path, args.label)

    if logged_count != args.count:
        raise RuntimeError(
            f"Log reported count={logged_count} for '{args.label}', expected {args.count}"
        )
    if len(actual) != logged_count:
        raise RuntimeError(
            f"Log dump size mismatch for '{args.label}': expected {logged_count}, got {len(actual)}"
        )
    if len(expected) != args.count:
        raise RuntimeError(
            f"Golden output size mismatch: expected {args.count}, got {len(expected)} from {expected_path}"
        )

    print(summarize("golden", expected))
    print(summarize("gvsoc", actual))

    if actual != expected:
        mismatch_lines = []
        for index, (exp_value, act_value) in enumerate(zip(expected, actual)):
            if exp_value != act_value:
                mismatch_lines.append(f"idx={index} expected={exp_value} actual={act_value}")
        details = ", ".join(mismatch_lines[:8]) if mismatch_lines else "size mismatch"
        raise RuntimeError(f"Tensor mismatch for '{args.label}': {details}")

    print(f"PASS: '{args.label}' matches exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def default_candidates(repo_root: Path) -> list[Path]:
    build_dir = repo_root / "aideck-gap8-examples" / "examples" / "other" / "dory_examples" / "application" / "BUILD" / "GAP8_V2" / "GCC_RISCV_PULPOS"
    app_dir = repo_root / "aideck-gap8-examples" / "examples" / "other" / "dory_examples" / "application"
    return [
        build_dir / "network_progress.txt",
        app_dir / "network_progress.txt",
    ]


def locate_progress_file(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)

    for candidate in default_candidates(repo_root):
        if candidate.exists():
            return candidate

    return default_candidates(repo_root)[0]


def parse_progress(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def get_value(data: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        if key in data:
            return data[key]
    return default


def as_int(data: dict[str, str], *keys: str, default: int = 0) -> int:
    try:
        return int(str(get_value(data, *keys, default=str(default))), 0)
    except ValueError:
        return default


def parse_csv_ints(raw: str, count: int, default: int = 0) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part, 0))
        except ValueError:
            values.append(default)
    if len(values) < count:
        values.extend([default] * (count - len(values)))
    return values[:count]


def format_report(path: Path, data: dict[str, str], previous: tuple[float, int, int] | None) -> tuple[str, tuple[float, int, int]]:
    now = time.time()
    mtime = path.stat().st_mtime
    age = now - mtime

    latest_layer = as_int(data, "latest_completed_layer", "l", default=-1)
    layer_name = get_value(data, "latest_completed_layer_name", "ln", default="n/a")
    heartbeat = as_int(data, "heartbeat", "hb", default=0)
    cycle_count = as_int(data, "cycle_count", "cy", default=0)
    status = get_value(data, "status", "st", default="missing")
    final_status = as_int(data, "final_status", "fs", default=-1)
    bbox_bytes = as_int(data, "bbox_bytes_written", "bb", default=0)
    cls_bytes = as_int(data, "cls_bytes_written", "cb", default=0)
    marker_epoch = as_int(data, "marker_epoch", "me", default=0)
    marker_layer = as_int(data, "marker_layer", "ml", default=-1)
    marker_layer_name = get_value(data, "marker_layer_name", default="n/a")
    marker = get_value(data, "latest_marker", "mn", default="")
    error = get_value(data, "latest_error", "er", default="")
    allocator_begin_addr = as_int(data, "allocator_begin_addr", "ab", default=0)
    allocator_end_addr = as_int(data, "allocator_end_addr", "ae", default=0)
    allocator_fields = parse_csv_ints(get_value(data, "alc"), 11, default=0)
    allocator_debug_base_addr = allocator_fields[0]
    allocator_debug_limit_addr = allocator_fields[1]
    allocator_last_before_begin_addr = allocator_fields[2]
    allocator_last_before_end_addr = allocator_fields[3]
    allocator_last_after_begin_addr = allocator_fields[4]
    allocator_last_after_end_addr = allocator_fields[5]
    allocator_last_return_addr = allocator_fields[6]
    allocator_last_request_size = allocator_fields[7]
    allocator_last_alignment = allocator_fields[8]
    allocator_last_direction = allocator_fields[9]
    allocator_last_operation = allocator_fields[10]
    l2_input_addr = as_int(data, "l2_input_addr", "ib", default=0)
    l2_input_size = as_int(data, "l2_input_size", "is", default=0)
    l2_output_addr = as_int(data, "l2_output_addr", "ob", default=0)
    l2_output_size = as_int(data, "l2_output_size", "os", default=0)
    l2_weights_addr = as_int(data, "l2_weights_addr", "wb", default=0)
    l2_weights_size = as_int(data, "l2_weights_size", "ws", default=0)
    bbox_output_addr = as_int(data, "bbox_output_addr", "bo", default=0)
    cls_output_addr = as_int(data, "cls_output_addr", "co", default=0)
    l3_copy_src_addr = as_int(data, "l3_copy_src_addr", "ls", default=0)
    l3_copy_dst_addr = as_int(data, "l3_copy_dst_addr", "ld", default=0)
    l3_copy_size = as_int(data, "l3_copy_size", "lz", default=0)
    capture_dest_addr = as_int(data, "capture_dest_addr", "cd", default=0)
    capture_size = as_int(data, "capture_size", "cz", default=0)
    probe66_fields = parse_csv_ints(get_value(data, "p66"), 13, default=0)
    probe66_valid = probe66_fields[0]
    probe66_layer = probe66_fields[1] if probe66_valid else -1
    probe66_base = probe66_fields[2]
    probe66_limit = probe66_fields[3]
    probe66_before_begin = probe66_fields[4]
    probe66_before_end = probe66_fields[5]
    probe66_after_begin = probe66_fields[6]
    probe66_after_end = probe66_fields[7]
    probe66_return = probe66_fields[8]
    probe66_request = probe66_fields[9]
    probe66_alignment = probe66_fields[10]
    probe66_direction = probe66_fields[11]
    probe66_operation = probe66_fields[12]
    probe67_fields = parse_csv_ints(get_value(data, "p67"), 13, default=0)
    probe67_valid = probe67_fields[0]
    probe67_layer = probe67_fields[1] if probe67_valid else -1
    probe67_base = probe67_fields[2]
    probe67_limit = probe67_fields[3]
    probe67_before_begin = probe67_fields[4]
    probe67_before_end = probe67_fields[5]
    probe67_after_begin = probe67_fields[6]
    probe67_after_end = probe67_fields[7]
    probe67_return = probe67_fields[8]
    probe67_request = probe67_fields[9]
    probe67_alignment = probe67_fields[10]
    probe67_direction = probe67_fields[11]
    probe67_operation = probe67_fields[12]

    def format_probe(
        valid: int,
        layer: int,
        base: int,
        limit: int,
        before_begin: int,
        before_end: int,
        after_begin: int,
        after_end: int,
        returned: int,
        request: int,
        alignment: int,
        direction: int,
        operation: int,
    ) -> str:
        if not valid:
            return "n/a"
        return (
            f"layer={layer} req={request} align={alignment} dir={direction} op={operation} "
            f"base=0x{base:08x} limit=0x{limit:08x} "
            f"before=0x{before_begin:08x}->0x{before_end:08x} "
            f"after=0x{after_begin:08x}->0x{after_end:08x} "
            f"ret=0x{returned:08x}"
        )

    throughput = "throughput=n/a"
    if previous is not None:
        prev_wall, prev_cycles, prev_layer = previous
        delta_wall = now - prev_wall
        delta_cycles = cycle_count - prev_cycles
        delta_layer = latest_layer - prev_layer
        if delta_wall > 0 and delta_cycles >= 0:
            throughput = (
                f"throughput={delta_cycles / delta_wall:,.0f} cyc/s "
                f"delta_layers={delta_layer:+d}"
            )

    report = (
        f"path={path} "
        f"status={status} "
        f"layer={latest_layer} "
        f"name={layer_name} "
        f"heartbeat={heartbeat} "
        f"cycles={cycle_count} "
        f"bbox_bytes={bbox_bytes} "
        f"cls_bytes={cls_bytes} "
        f"final_status={final_status} "
        f"marker_epoch={marker_epoch} "
        f"marker_layer={marker_layer} "
        f"marker_layer_name={marker_layer_name} "
        f"marker={marker or 'n/a'} "
        f"error={error or 'n/a'} "
        f"alloc=0x{allocator_begin_addr:08x}->0x{allocator_end_addr:08x} "
        f"alloc_dbg=base:0x{allocator_debug_base_addr:08x} limit:0x{allocator_debug_limit_addr:08x} "
        f"before:0x{allocator_last_before_begin_addr:08x}->0x{allocator_last_before_end_addr:08x} "
        f"after:0x{allocator_last_after_begin_addr:08x}->0x{allocator_last_after_end_addr:08x} "
        f"req:{allocator_last_request_size} align:{allocator_last_alignment} dir:{allocator_last_direction} "
        f"op:{allocator_last_operation} ret:0x{allocator_last_return_addr:08x} "
        f"l2_in=0x{l2_input_addr:08x}/{l2_input_size} "
        f"l2_out=0x{l2_output_addr:08x}/{l2_output_size} "
        f"l2_w=0x{l2_weights_addr:08x}/{l2_weights_size} "
        f"bbox_ptr=0x{bbox_output_addr:08x} "
        f"cls_ptr=0x{cls_output_addr:08x} "
        f"l3=0x{l3_copy_src_addr:08x}->0x{l3_copy_dst_addr:08x}/{l3_copy_size} "
        f"capture=0x{capture_dest_addr:08x}/{capture_size} "
        f"probe66=[{format_probe(probe66_valid, probe66_layer, probe66_base, probe66_limit, probe66_before_begin, probe66_before_end, probe66_after_begin, probe66_after_end, probe66_return, probe66_request, probe66_alignment, probe66_direction, probe66_operation)}] "
        f"probe67=[{format_probe(probe67_valid, probe67_layer, probe67_base, probe67_limit, probe67_before_begin, probe67_before_end, probe67_after_begin, probe67_after_end, probe67_return, probe67_request, probe67_alignment, probe67_direction, probe67_operation)}] "
        f"age={age:.1f}s "
        f"{throughput}"
    )
    return report, (now, cycle_count, latest_layer)


def describe_outputs(path: Path) -> list[str]:
    output_lines: list[str] = []
    for name in ("bbox.bin", "cls.bin"):
        candidate = path.parent / name
        if candidate.exists():
            output_lines.append(f"{candidate.name}={candidate.stat().st_size} bytes")
    return output_lines


def probe_valid_for_layer(data: dict[str, str], layer: int) -> bool:
    probe_fields = parse_csv_ints(get_value(data, f"p{layer}"), 13, default=0)
    return bool(probe_fields and probe_fields[0])


def conditions_met(
    data: dict[str, str],
    until_layer: int | None,
    until_marker: str | None,
    until_probe_layer: int | None,
) -> bool:
    if until_layer is not None and as_int(data, "latest_completed_layer", "l", default=-1) < until_layer:
        return False
    if until_marker is not None and get_value(data, "latest_marker", "mn", default="") != until_marker:
        return False
    if until_probe_layer is not None and not probe_valid_for_layer(data, until_probe_layer):
        return False
    return any(value is not None for value in (until_layer, until_marker, until_probe_layer))


def emit_lines(lines: list[str], log_handle) -> None:
    for line in lines:
        print(line)
        if log_handle is not None:
            log_handle.write(line + "\n")
            log_handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll GAP8 GVSOC progress exported by the SSD app.")
    parser.add_argument("--progress-file", help="Path to network_progress.txt")
    parser.add_argument("--watch", action="store_true", help="Keep polling until interrupted.")
    parser.add_argument("--interval", type=float, default=5.0, help="Watch interval in seconds.")
    parser.add_argument("--timeout", type=float, default=0.0, help="Stop watching after this many seconds (0 disables timeout).")
    parser.add_argument("--until-layer", type=int, help="Stop when latest_completed_layer reaches this value.")
    parser.add_argument("--until-marker", help="Stop when latest_marker matches this value exactly.")
    parser.add_argument("--until-probe-layer", type=int, choices=(66, 67), help="Stop when the requested layer allocation probe becomes valid.")
    parser.add_argument("--log-file", help="Append watch output to this file.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    progress_path = locate_progress_file(repo_root, args.progress_file)

    if not args.watch:
        if not progress_path.exists():
            print(f"progress file not found: {progress_path}", file=sys.stderr)
            return 1
        data = parse_progress(progress_path)
        report, _ = format_report(progress_path, data, None)
        print(report)
        for line in describe_outputs(progress_path):
            print(line)
        return 0

    start = time.time()
    log_handle = open(args.log_file, "a", encoding="utf-8") if args.log_file else None
    try:
        emit_lines([f"watching {progress_path}"], log_handle)
        previous: tuple[float, int, int] | None = None
        last_signature: tuple[int, int, int, int] | None = None
        while True:
            if args.timeout > 0 and (time.time() - start) > args.timeout:
                emit_lines([f"timeout after {args.timeout:.1f}s waiting on {progress_path}"], log_handle)
                return 124

            if progress_path.exists():
                data = parse_progress(progress_path)
                signature = (
                    as_int(data, "latest_completed_layer", "l", default=-1),
                    as_int(data, "heartbeat", "hb", default=0),
                    as_int(data, "cycle_count", "cy", default=0),
                    as_int(data, "marker_epoch", "me", default=0),
                )
                if signature != last_signature:
                    report, previous = format_report(progress_path, data, previous)
                    emit_lines([report, *describe_outputs(progress_path)], log_handle)
                    last_signature = signature

                if conditions_met(data, args.until_layer, args.until_marker, args.until_probe_layer):
                    emit_lines(["target condition reached"], log_handle)
                    return 0

                finished = bool(as_int(data, "finished", "fn", default=0))
                if finished:
                    if any(value is not None for value in (args.until_layer, args.until_marker, args.until_probe_layer)):
                        emit_lines(["run finished before requested target condition was reached"], log_handle)
                        return 2
                    emit_lines(["run finished"], log_handle)
                    return 0
            else:
                emit_lines([f"waiting for {progress_path}"], log_handle)
            time.sleep(args.interval)
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

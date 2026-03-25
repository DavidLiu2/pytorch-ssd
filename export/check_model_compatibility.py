#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import operator
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Finding:
    scope: str
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopeReport:
    scope: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self.findings.append(
            Finding(
                scope=self.scope,
                severity=severity,
                code=code,
                message=message,
                details=details,
            )
        )

    @property
    def status(self) -> str:
        severities = {finding.severity for finding in self.findings}
        if "error" in severities:
            return "fail"
        if "warning" in severities:
            return "warn"
        return "pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check PyTorch and ONNX model compatibility with the current NEMO + DORY "
            "deployment path."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["python", "onnx", "both"],
        default="both",
        help="Which checks to run.",
    )
    parser.add_argument(
        "--model-type",
        choices=["ssd", "hybrid_follow"],
        default="hybrid_follow",
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--eps-in", type=float, default=1.0 / 255.0)
    parser.add_argument("--stage", choices=["fp", "fq", "qd", "id"], default="id")
    parser.add_argument("--calib-dir", type=str, default=None)
    parser.add_argument("--calib-tensor", type=str, default=None)
    parser.add_argument("--calib-batches", type=int, default=32)
    parser.add_argument(
        "--compat-calib-batches",
        type=int,
        default=8,
        help="How many calibration samples to use in the dry-run stage test.",
    )
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--mean", type=str, default=None)
    parser.add_argument("--std", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument(
        "--dory-onnx",
        type=str,
        default=None,
        help="Optional DORY-clean ONNX to inspect with stricter rules.",
    )
    parser.add_argument("--report-json", type=str, default=None)
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--skip-stage-dry-run", action="store_true")
    return parser.parse_args()


def summarize_findings(findings: list[Finding]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def print_scope_report(report: ScopeReport) -> None:
    counts = summarize_findings(report.findings)
    print(
        f"[compat] {report.scope}: status={report.status.upper()} "
        f"errors={counts['error']} warnings={counts['warning']} infos={counts['info']}"
    )
    for finding in report.findings:
        print(
            f"[compat] {finding.scope} {finding.severity.upper()} "
            f"{finding.code}: {finding.message}"
        )


def build_export_stage_args(args: argparse.Namespace) -> SimpleNamespace:
    compat_batches = min(args.calib_batches, args.compat_calib_batches)
    return SimpleNamespace(
        model_type=args.model_type,
        calib_dir=args.calib_dir,
        calib_tensor=args.calib_tensor,
        calib_batches=compat_batches,
        calib_seed=args.calib_seed,
        mean=args.mean,
        std=args.std,
        input_channels=args.input_channels,
        eps_in=args.eps_in,
    )


def analyze_python_model(args: argparse.Namespace) -> ScopeReport:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.fx import symbolic_trace

    import nemo

    from export_nemo_quant import (
        build_model,
        iter_calib_batches,
        load_checkpoint,
        maybe_convert_hybrid_follow_to_export_head,
        maybe_fuse_hybrid_follow_for_export,
        normalize_integer_requant_tensors,
        patch_model_to_graph_compat,
        repair_hybrid_follow_fused_quant_graph,
    )

    report = ScopeReport(scope="python")
    image_size = (args.height, args.width)
    device = torch.device("cpu")

    model = build_model(
        args.model_type,
        args.num_classes,
        args.width_mult,
        image_size,
        args.input_channels,
    )
    if args.ckpt:
        model = load_checkpoint(model, args.ckpt, device)
    model.to(device).eval()

    named_modules = list(model.named_modules())
    leaf_modules = [
        (name, module)
        for name, module in named_modules
        if name and len(list(module.children())) == 0
    ]
    report.metrics["leaf_module_classes"] = Counter(
        module.__class__.__name__ for _, module in leaf_modules
    )

    sequential_modules = [
        (name, module)
        for name, module in named_modules
        if isinstance(module, nn.Sequential) and name
    ]
    numeric_sequentials = [
        name
        for name, module in sequential_modules
        if module._modules and all(key.isdigit() for key in module._modules.keys())
    ]
    if numeric_sequentials:
        report.add(
            "warning",
            "PY.SEQUENTIAL_NUMERIC_NAMES",
            "The model uses nn.Sequential blocks with numeric child names. This is workable, but it makes eps mapping and graph debugging harder.",
            examples=numeric_sequentials[:8],
        )
    else:
        report.add(
            "info",
            "PY.SEQUENTIAL_NAMES",
            "Sequential containers use explicit child names or are absent.",
        )

    numeric_paths = [
        name
        for name, _ in named_modules
        if name and any(part.isdigit() for part in name.split("."))
    ]
    if numeric_paths:
        report.add(
            "warning",
            "PY.NUMERIC_PATH_SEGMENTS",
            "The model exposes numeric path segments such as stage1.0.conv1. This is not a deployment blocker, but it is less readable than explicit block names.",
            examples=numeric_paths[:10],
        )

    head_names = [
        name
        for name, module in model.named_children()
        if name.startswith("head_") and isinstance(module, nn.Linear)
    ]
    if head_names:
        report.add(
            "info",
            "PY.EXPLICIT_HEADS",
            "The output heads are explicitly named, which is helpful for export and debugging.",
            heads=head_names,
        )

    add_modules = [
        name
        for name, module in named_modules
        if name and "Add" in module.__class__.__name__
    ]
    if add_modules:
        report.add(
            "info",
            "PY.EXPLICIT_ADD_MODULES",
            "Residual additions are represented as explicit modules instead of plain functional adds.",
            add_modules=add_modules[:16],
        )

    allowed_leaf_classes = {
        "AvgPool1d",
        "AvgPool2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "Conv1d",
        "Conv2d",
        "DefaultBoxGenerator",
        "Dropout",
        "Flatten",
        "GeneralizedRCNNTransform",
        "Identity",
        "Linear",
        "MaxPool1d",
        "MaxPool2d",
        "PACT_IntegerAdd",
        "ReLU",
        "ReLU6",
    }
    unusual_leafs = sorted(
        {
            module.__class__.__name__
            for _, module in leaf_modules
            if module.__class__.__name__ not in allowed_leaf_classes
        }
    )
    if unusual_leafs:
        report.add(
            "warning",
            "PY.UNUSUAL_LEAF_MODULES",
            "The model contains leaf modules outside the small set proven in the current GAP8 flow. They may still work, but they deserve closer export review.",
            leaf_classes=unusual_leafs,
        )

    try:
        traced = symbolic_trace(model)
        report.add(
            "info",
            "PY.FX_TRACE_OK",
            "torch.fx symbolic_trace() succeeded on the PyTorch model.",
        )
        functional_relu_nodes = []
        functional_add_nodes = []
        unexpected_function_nodes = []
        allowed_function_targets = {
            getattr(operator, "getitem", None),
            torch.add,
            torch.cat,
            torch.flatten,
        }
        allowed_function_targets.discard(None)
        allowed_function_names = {"add", "cat", "flatten", "getitem"}
        for node in traced.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            target_name = getattr(target, "__name__", repr(target))
            if target in {F.relu, torch.relu} or target_name == "relu":
                functional_relu_nodes.append(node.name)
            elif target in {operator.add, torch.add} or target_name == "add":
                functional_add_nodes.append(node.name)
            elif target not in allowed_function_targets and target_name not in allowed_function_names:
                unexpected_function_nodes.append(target_name)

        if functional_relu_nodes:
            report.add(
                "error",
                "PY.FUNCTIONAL_RELU",
                "The model uses functional ReLU calls instead of explicit nn.ReLU modules.",
                fx_nodes=functional_relu_nodes[:12],
            )
        else:
            report.add(
                "info",
                "PY.EXPLICIT_RELU_MODULES",
                "No functional ReLU calls were detected in the FX graph.",
            )

        if functional_add_nodes and not add_modules:
            report.add(
                "error",
                "PY.FUNCTIONAL_ADD",
                "Residual or merge adds appear as functional graph ops without matching explicit Add modules.",
                fx_nodes=functional_add_nodes[:12],
            )
        elif functional_add_nodes:
            report.add(
                "info",
                "PY.EXPLICIT_ADD_TRACE",
                "FX lowers the explicit Add modules into add nodes, but the model still exposes named Add modules for export/debugging.",
                fx_nodes=functional_add_nodes[:12],
            )
        else:
            report.add(
                "info",
                "PY.EXPLICIT_MERGES",
                "No functional add nodes were detected in the FX graph.",
            )

        if unexpected_function_nodes:
            report.add(
                "warning",
                "PY.UNUSUAL_FUNCTION_CALLS",
                "The FX graph contains call_function targets outside the basic Conv/BN/ReLU/Add path.",
                targets=sorted(set(unexpected_function_nodes))[:16],
            )
    except Exception as exc:
        report.add(
            "error",
            "PY.FX_TRACE_FAILED",
            "torch.fx symbolic_trace() failed. Data-dependent control flow or unsupported graph patterns will make export harder.",
            error=str(exc),
        )

    if args.skip_stage_dry_run:
        report.add(
            "info",
            "PY.STAGE_DRY_RUN_SKIPPED",
            "Skipped the NEMO QD/ID dry-run at user request.",
        )
        return report

    try:
        export_model = maybe_fuse_hybrid_follow_for_export(model)
        export_model = maybe_convert_hybrid_follow_to_export_head(export_model)
        export_model.to(device).eval()
        dummy_input = torch.randn(
            1,
            args.input_channels,
            args.height,
            args.width,
            device=device,
        )
        patch_model_to_graph_compat()
        model_q = nemo.transform.quantize_pact(export_model, dummy_input=dummy_input)
        model_q.to(device).eval()
        if args.model_type == "hybrid_follow":
            repair_hybrid_follow_fused_quant_graph(model_q)

        model_q.change_precision(
            bits=args.bits,
            scale_weights=True,
            scale_activations=True,
        )

        compat_args = build_export_stage_args(args)
        with torch.no_grad():
            with model_q.statistics_act():
                for tensor in iter_calib_batches(compat_args, image_size, device):
                    _ = model_q(tensor)
        model_q.reset_alpha_act()
        try:
            model_q.reset_alpha_weights()
        except Exception:
            pass

        if args.stage in {"qd", "id"}:
            if args.model_type == "hybrid_follow":
                repair_hybrid_follow_fused_quant_graph(model_q)
            model_q.qd_stage(eps_in=args.eps_in)
            report.add(
                "info",
                "PY.QD_STAGE_OK",
                "The export-ready model completed qd_stage() successfully during the dry-run.",
            )
            if args.model_type == "hybrid_follow":
                repair_hybrid_follow_fused_quant_graph(model_q)

        if args.stage == "id":
            model_q.id_stage()
            normalize_integer_requant_tensors(model_q)
            report.add(
                "info",
                "PY.ID_STAGE_OK",
                "The export-ready model completed id_stage() successfully during the dry-run.",
            )
    except Exception as exc:
        report.add(
            "error",
            "PY.STAGE_DRY_RUN_FAILED",
            "The export-ready model failed the NEMO stage dry-run before ONNX export.",
            stage=args.stage,
            error=str(exc),
        )

    return report


KNOWN_DORY_OPS = {
    "Add",
    "AveragePool",
    "Cast",
    "Clip",
    "Constant",
    "Conv",
    "Div",
    "Flatten",
    "Floor",
    "Gemm",
    "MatMul",
    "Mul",
    "Pad",
}


def analyze_onnx_graph(path: Path, role: str) -> ScopeReport:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    report = ScopeReport(scope=f"onnx:{role}")
    if not path.exists():
        report.add(
            "error",
            "ONNX.MISSING",
            "The requested ONNX file does not exist.",
            path=str(path),
        )
        return report

    model = onnx.load(str(path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    report.metrics["path"] = str(path)
    report.metrics["op_counts"] = dict(op_counts)
    report.metrics["initializer_count"] = len(model.graph.initializer)

    opset_versions = {
        entry.domain or "ai.onnx": entry.version
        for entry in model.opset_import
    }
    report.metrics["opset_versions"] = opset_versions
    report.add(
        "info",
        "ONNX.LOADED",
        "Loaded the ONNX graph successfully.",
        path=str(path),
        opset=opset_versions,
    )

    unexpected_ops = sorted(set(op_counts) - KNOWN_DORY_OPS)
    if unexpected_ops:
        severity = "error" if role == "dory" else "warning"
        report.add(
            severity,
            "ONNX.UNEXPECTED_OPS",
            "The ONNX graph contains ops outside the small DORY-friendly set used by the current GAP8 flow.",
            ops=unexpected_ops,
        )
    else:
        report.add(
            "info",
            "ONNX.DORY_FRIENDLY_OP_SET",
            "The ONNX op set matches the currently validated DORY-friendly path.",
        )

    if "BatchNormalization" in op_counts:
        report.add(
            "error",
            "ONNX.BATCHNORM_PRESENT",
            "BatchNormalization nodes remain in the deploy ONNX. They should be folded or quantized away before DORY export.",
            count=op_counts["BatchNormalization"],
        )

    if "QuantizeLinear" in op_counts or "DequantizeLinear" in op_counts:
        report.add(
            "error",
            "ONNX.QDQ_PRESENT",
            "QuantizeLinear / DequantizeLinear nodes remain in the deploy ONNX.",
            counts={
                "QuantizeLinear": op_counts.get("QuantizeLinear", 0),
                "DequantizeLinear": op_counts.get("DequantizeLinear", 0),
            },
        )

    suspicious_cleanup_ops = {
        op_name: op_counts[op_name]
        for op_name in ("Min", "Transpose")
        if op_counts.get(op_name, 0)
    }
    if suspicious_cleanup_ops:
        severity = "error" if role == "dory" else "warning"
        report.add(
            severity,
            "ONNX.CLEANUP_OPS_PRESENT",
            "The ONNX graph still contains ops that the current cleanup flow usually strips before DORY frontend parsing.",
            counts=suspicious_cleanup_ops,
        )

    kappa_inits = [
        init.name for init in model.graph.initializer if init.name.endswith(".kappa")
    ]
    lamda_inits = [
        init.name for init in model.graph.initializer if init.name.endswith(".lamda")
    ]
    if kappa_inits or lamda_inits:
        severity = "error" if role == "dory" else "warning"
        report.add(
            severity,
            "ONNX.BN_AFFINE_INITIALIZERS",
            "The ONNX graph still contains explicit BN affine initializers (.kappa / .lamda).",
            kappa_count=len(kappa_inits),
            lamda_count=len(lamda_inits),
            examples=(kappa_inits + lamda_inits)[:10],
        )
    else:
        report.add(
            "info",
            "ONNX.NO_BN_AFFINE_INITIALIZERS",
            "No .kappa or .lamda initializers remain in the ONNX graph.",
        )

    weight_initializer_names = set()
    for node in model.graph.node:
        if node.op_type in {"Conv", "Gemm", "MatMul"} and len(node.input) >= 2:
            weight_initializer_names.add(node.input[1])

    weight_range_issues = []
    for initializer in model.graph.initializer:
        if initializer.name not in weight_initializer_names:
            continue
        array = numpy_helper.to_array(initializer)
        if array.dtype.kind not in {"f", "i", "u"}:
            continue
        rounded = np.rint(array).astype(np.int64, copy=False)
        mask = (rounded < -128) | (rounded > 127)
        if not np.any(mask):
            continue
        weight_range_issues.append(
            {
                "name": initializer.name,
                "count": int(np.count_nonzero(mask)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        )

    if weight_range_issues:
        report.add(
            "warning",
            "ONNX.WEIGHT_RANGE_DIAGNOSTIC",
            "Some Conv/Gemm/MatMul initializers round outside signed int8. Treat this as a diagnostic and confirm DORY frontend acceptance.",
            examples=weight_range_issues[:8],
            total=len(weight_range_issues),
        )
    else:
        report.add(
            "info",
            "ONNX.WEIGHT_RANGE_OK",
            "No Conv/Gemm/MatMul initializers round outside signed int8.",
        )

    return report


def write_report(
    args: argparse.Namespace,
    scope_reports: list[ScopeReport],
) -> dict[str, Any]:
    findings = [
        asdict(finding)
        for report in scope_reports
        for finding in report.findings
    ]
    counts = summarize_findings(
        [finding for report in scope_reports for finding in report.findings]
    )
    overall_status = "pass"
    if counts["error"]:
        overall_status = "fail"
    elif counts["warning"]:
        overall_status = "warn"

    report_data = {
        "status": overall_status,
        "counts": counts,
        "reports": [
            {
                "scope": report.scope,
                "status": report.status,
                "metrics": report.metrics,
                "findings": [asdict(finding) for finding in report.findings],
            }
            for report in scope_reports
        ],
        "findings": findings,
    }

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
        print(f"[compat] Wrote JSON report: {report_path}")

    return report_data


def main() -> None:
    args = parse_args()
    scope_reports: list[ScopeReport] = []

    if args.mode in {"python", "both"}:
        scope_reports.append(analyze_python_model(args))

    if args.mode in {"onnx", "both"}:
        if not args.onnx and not args.dory_onnx:
            raise SystemExit("ONNX mode requires --onnx and/or --dory-onnx.")
        if args.onnx:
            scope_reports.append(analyze_onnx_graph(Path(args.onnx), role="raw"))
        if args.dory_onnx:
            scope_reports.append(analyze_onnx_graph(Path(args.dory_onnx), role="dory"))

    for report in scope_reports:
        print_scope_report(report)

    report_data = write_report(args, scope_reports)
    if args.fail_on_errors and report_data["counts"]["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

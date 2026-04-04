from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Optional

import nemo
import torch


def _compute_integer_add_eps_out(eps_in_list, policy: str):
    max_eps = max(eps_in_list)
    fan_in = len(eps_in_list)
    if policy in {"legacy", "max_only"}:
        return max_eps
    if policy in {"max_branch", "max_branch_scale"}:
        return max_eps
    if policy in {"joint_balanced", "joint_balanced_scale", "geometric_mean", "geomean"}:
        min_eps = min(eps_in_list)
        return torch.sqrt(max_eps * min_eps)
    if policy == "sqrt_fanin":
        return max_eps * math.sqrt(fan_in)
    if policy == "midpoint":
        return max_eps * ((1.0 + math.sqrt(fan_in)) / 2.0)
    if policy == "fanin":
        return max_eps * fan_in
    raise ValueError(f"Unsupported integer-add scale policy: {policy}")


def _normalize_integer_add_override_spec(value: Any):
    if value is None:
        return None
    if isinstance(value, dict):
        override = dict(value)
        if override.get("eps_out") is not None:
            return {
                "mode": "explicit_eps_out",
                "eps_out": float(override["eps_out"]),
                "policy_name": (
                    str(override.get("policy_name")).strip().lower()
                    if override.get("policy_name") is not None
                    else "explicit_eps_out"
                ),
                "metadata": deepcopy(override.get("metadata") or {}),
            }
        policy = (
            str(
                override.get("policy")
                or override.get("policy_name")
                or override.get("mode")
                or ""
            )
            .strip()
            .lower()
        )
        if not policy:
            raise ValueError(f"Unsupported integer-add scale override: {value}")
        return {
            "mode": "policy",
            "policy": policy,
        }
    return {
        "mode": "policy",
        "policy": str(value).strip().lower(),
    }


def _integer_add_cls():
    return nemo.quant.pact.PACT_IntegerAdd


def _ensure_original_get_output_eps():
    integer_add_cls = _integer_add_cls()
    original = getattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps", None)
    if original is None:
        original = integer_add_cls.get_output_eps
        integer_add_cls._export_nemo_quant_original_get_output_eps = original
    return integer_add_cls, original


def _normalized_integer_add_overrides(operator_policies: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    normalized_overrides = {}
    for module_name, module_policy in (operator_policies or {}).items():
        if module_policy is None:
            continue
        normalized_overrides[str(module_name)] = _normalize_integer_add_override_spec(module_policy)
    return normalized_overrides


def _patched_integer_add_method(original, policy: str, normalized_overrides: dict[str, Any]):
    def patched_get_output_eps(self, eps_in_list):
        if type(eps_in_list) is list:
            self.eps_in_list = eps_in_list
        module_name = getattr(self, "_export_nemo_quant_module_name", None)
        effective_override = normalized_overrides.get(module_name)
        if effective_override is None:
            effective_override = {"mode": "policy", "policy": policy}
        effective_policy = (
            effective_override.get("policy")
            if isinstance(effective_override, dict)
            else str(effective_override)
        )
        if effective_policy == "legacy" and effective_override.get("mode") == "policy":
            return original(self, eps_in_list)
        if effective_override.get("mode") == "explicit_eps_out":
            eps_reference = None
            for value in self.eps_in_list:
                if torch.is_tensor(value):
                    eps_reference = value
                    break
            if eps_reference is None:
                self.eps_out = torch.as_tensor(
                    float(effective_override["eps_out"]),
                    dtype=torch.float32,
                )
            else:
                self.eps_out = torch.as_tensor(
                    float(effective_override["eps_out"]),
                    dtype=eps_reference.dtype,
                    device=eps_reference.device,
                )
        else:
            self.eps_out = _compute_integer_add_eps_out(self.eps_in_list, effective_policy)
        self.alpha_out = 2.0 ** (self.precision.get_bits()) - 1
        self.D = 2 ** torch.as_tensor(
            torch.ceil(
                torch.log2(
                    self.requantization_factor * self.eps_out / min(self.eps_in_list)
                )
            ),
            dtype=torch.int64,
        )
        self._export_nemo_quant_scale_override = deepcopy(effective_override)
        return self.eps_out

    return patched_get_output_eps


def patch_integer_add_scale_selection(
    policy: str = "legacy",
    operator_policies: Optional[dict[str, Any]] = None,
):
    integer_add_cls, original = _ensure_original_get_output_eps()
    normalized_policy = (policy or "legacy").strip().lower() or "legacy"
    normalized_overrides = _normalized_integer_add_overrides(operator_policies)

    if normalized_policy == "legacy" and not normalized_overrides:
        integer_add_cls.get_output_eps = original
    else:
        integer_add_cls.get_output_eps = _patched_integer_add_method(
            original,
            normalized_policy,
            normalized_overrides,
        )
    integer_add_cls._export_nemo_quant_scale_policy = normalized_policy
    integer_add_cls._export_nemo_quant_scale_policy_overrides = deepcopy(normalized_overrides)
    return normalized_policy


def restore_integer_add_scale_selection() -> None:
    integer_add_cls = _integer_add_cls()
    original = getattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps", None)
    if original is not None:
        integer_add_cls.get_output_eps = original
    if hasattr(integer_add_cls, "_export_nemo_quant_scale_policy"):
        delattr(integer_add_cls, "_export_nemo_quant_scale_policy")
    if hasattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides"):
        delattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides")


@contextmanager
def integer_add_scale_selection_scope(
    policy: str = "legacy",
    operator_policies: Optional[dict[str, Any]] = None,
):
    integer_add_cls = _integer_add_cls()
    previous_method = integer_add_cls.get_output_eps
    had_original = hasattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps")
    previous_original = getattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps", None)
    had_policy = hasattr(integer_add_cls, "_export_nemo_quant_scale_policy")
    previous_policy = getattr(integer_add_cls, "_export_nemo_quant_scale_policy", None)
    had_overrides = hasattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides")
    previous_overrides = deepcopy(
        getattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides", {})
    )

    normalized_policy = patch_integer_add_scale_selection(
        policy=policy,
        operator_policies=operator_policies,
    )
    normalized_overrides = deepcopy(
        getattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides", {})
    )

    try:
        yield {
            "policy": normalized_policy,
            "operator_overrides": normalized_overrides,
        }
    finally:
        integer_add_cls.get_output_eps = previous_method
        if had_original:
            integer_add_cls._export_nemo_quant_original_get_output_eps = previous_original
        elif hasattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps"):
            delattr(integer_add_cls, "_export_nemo_quant_original_get_output_eps")

        if had_policy:
            integer_add_cls._export_nemo_quant_scale_policy = previous_policy
        elif hasattr(integer_add_cls, "_export_nemo_quant_scale_policy"):
            delattr(integer_add_cls, "_export_nemo_quant_scale_policy")

        if had_overrides:
            integer_add_cls._export_nemo_quant_scale_policy_overrides = previous_overrides
        elif hasattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides"):
            delattr(integer_add_cls, "_export_nemo_quant_scale_policy_overrides")


def run_with_integer_add_scale_selection(
    runner: Callable[..., Any],
    *args,
    policy: str = "legacy",
    operator_policies: Optional[dict[str, Any]] = None,
    **kwargs,
):
    with integer_add_scale_selection_scope(policy, operator_policies):
        return runner(*args, **kwargs)


def clone_model_and_run_with_integer_add_scale_selection(
    model,
    runner: Callable[..., Any],
    *args,
    policy: str = "legacy",
    operator_policies: Optional[dict[str, Any]] = None,
    **kwargs,
):
    with integer_add_scale_selection_scope(policy, operator_policies):
        return runner(deepcopy(model), *args, **kwargs)

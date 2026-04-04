from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

try:
    from nemo.quant.pact import PACT_IntegerAdd
except ImportError:
    class PACT_IntegerAdd(nn.Module):
        def forward(self, *x: torch.Tensor) -> torch.Tensor:
            if len(x) != 2:
                raise ValueError("PACT_IntegerAdd fallback expects exactly two input tensors.")
            return x[0] + x[1]


HYBRID_FOLLOW_BASE_STAGE_CHANNELS: Tuple[int, int, int, int] = (24, 32, 64, 80)
HYBRID_FOLLOW_STAGE4_VARIANTS = (
    "baseline",
    "single_conv_residual",
    "plain_non_residual",
    "single_conv_non_residual",
    "narrow_stage4",
)
HYBRID_FOLLOW_LEGACY_STAGE4_ALIASES = {
    "none": "baseline",
    "single_conv": "single_conv_residual",
}
HYBRID_FOLLOW_VARIANT_CHECKPOINT_ALIASES = {
    "baseline": "none",
    "single_conv_residual": "single_conv",
    "plain_non_residual": "plain_non_residual",
    "single_conv_non_residual": "single_conv_non_residual",
    "narrow_stage4": "narrow_stage4",
}


def normalize_stage4_variant(
    stage4_variant: str | None = None,
    stage4_1_ablation: str | None = None,
) -> str:
    candidate = (stage4_variant or stage4_1_ablation or "baseline").strip().lower()
    candidate = HYBRID_FOLLOW_LEGACY_STAGE4_ALIASES.get(candidate, candidate)
    alias_map = {
        "variant_a": "plain_non_residual",
        "variant_b": "single_conv_non_residual",
        "variant_c": "narrow_stage4",
    }
    candidate = alias_map.get(candidate, candidate)
    if candidate not in HYBRID_FOLLOW_STAGE4_VARIANTS:
        raise ValueError(
            "Unsupported stage4 variant '{}'; expected one of {}.".format(
                candidate,
                HYBRID_FOLLOW_STAGE4_VARIANTS,
            )
        )
    return candidate


def checkpoint_stage4_ablation_value(stage4_variant: str) -> str:
    return HYBRID_FOLLOW_VARIANT_CHECKPOINT_ALIASES[normalize_stage4_variant(stage4_variant=stage4_variant)]


def stage_channels_for_variant(
    stage_channels: Tuple[int, int, int, int],
    stage4_variant: str,
) -> Tuple[int, int, int, int]:
    normalized = normalize_stage4_variant(stage4_variant=stage4_variant)
    if len(stage_channels) != 4:
        raise ValueError(
            "HybridFollowNet expects 4 stage channel values, got {}.".format(stage_channels)
        )
    if normalized != "narrow_stage4":
        return tuple(int(value) for value in stage_channels)
    return (
        int(stage_channels[0]),
        int(stage_channels[1]),
        int(stage_channels[2]),
        int(min(stage_channels[3], stage_channels[2])),
    )


def hybrid_follow_config_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    payload = dict(metadata or {})
    stage4_variant = normalize_stage4_variant(
        stage4_variant=payload.get("stage4_variant"),
        stage4_1_ablation=payload.get("stage4_1_ablation"),
    )
    raw_stage_channels = payload.get("stage_channels")
    if raw_stage_channels is None:
        raw_stage_channels = stage_channels_for_variant(
            HYBRID_FOLLOW_BASE_STAGE_CHANNELS,
            stage4_variant,
        )
    stage_channels = tuple(int(value) for value in raw_stage_channels)
    return {
        "stage4_variant": stage4_variant,
        "stage_channels": stage_channels_for_variant(stage_channels, stage4_variant),
    }


def _slice_tensor_to_shape(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if tuple(value.shape) == tuple(target_shape):
        return value
    if value.ndim != len(target_shape):
        return None
    if any(source_dim < target_dim for source_dim, target_dim in zip(value.shape, target_shape)):
        return None
    slices = tuple(slice(0, int(target_dim)) for target_dim in target_shape)
    sliced = value[slices]
    if tuple(sliced.shape) != tuple(target_shape):
        return None
    return sliced


def adapt_hybrid_follow_state_dict_to_model(
    state_dict: Mapping[str, Any],
    model: nn.Module,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapted = dict(state_dict)
    target_state = model.state_dict()
    adaptation_report = []
    allowed_prefixes = (
        "stage4.",
        "head.",
        "head_x.",
        "head_size.",
        "head_vis.",
    )

    for key, target_tensor in target_state.items():
        value = adapted.get(key)
        if not torch.is_tensor(value) or not torch.is_tensor(target_tensor):
            continue
        if tuple(value.shape) == tuple(target_tensor.shape):
            continue
        if not key.startswith(allowed_prefixes):
            continue
        sliced = _slice_tensor_to_shape(value, target_tensor.shape)
        if sliced is None:
            continue
        adapted[key] = sliced.to(dtype=value.dtype)
        adaptation_report.append(
            {
                "key": key,
                "source_shape": tuple(value.shape),
                "target_shape": tuple(target_tensor.shape),
            }
        )

    return adapted, adaptation_report


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class PassthroughAdd(nn.Module):
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        if not x:
            raise ValueError("PassthroughAdd expects at least one tensor input.")
        return x[0]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        single_conv_mode: bool = False,
        use_residual: bool = True,
        use_relu1: bool = True,
        use_out_relu: bool = True,
    ) -> None:
        super().__init__()
        self.single_conv_mode = bool(single_conv_mode)
        self.use_residual = bool(use_residual)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False) if use_relu1 else nn.Identity()
        if self.single_conv_mode:
            self.conv2 = nn.Identity()
            self.bn2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = None

        if self.use_residual:
            # NEMO needs an explicit module here for branched residual adds.
            self.add = PACT_IntegerAdd()
        else:
            self.add = PassthroughAdd()
        self.out_relu = nn.ReLU(inplace=False) if use_out_relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = None
        if self.use_residual:
            identity = x if self.proj is None else self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        if not self.single_conv_mode:
            out = self.conv2(out)
            out = self.bn2(out)
        if self.use_residual:
            out = self.add(out, identity)
        else:
            out = self.add(out)
        out = self.out_relu(out)
        return out


class HybridFollowNet(nn.Module):
    """
    Compact DroNet-style follow network.

    Input:
      - grayscale tensor shaped [B, 1, 128, 128]

    Output:
      - raw linear outputs shaped [B, 3]
        0: x_offset logit/regression value
        1: size_proxy regression value
        2: visibility_confidence logit
    """

    def __init__(
        self,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        stem_channels: int = 16,
        stage_channels: Tuple[int, int, int, int] = HYBRID_FOLLOW_BASE_STAGE_CHANNELS,
        stage4_variant: str | None = None,
        stage4_1_ablation: str | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError("HybridFollowNet expects true 1-channel grayscale input.")
        if image_size != (128, 128):
            raise ValueError(
                "HybridFollowNet assumes fixed 128x128 input so stage4 ends at 4x4."
            )

        self.input_channels = input_channels
        self.image_size = image_size
        self.stage4_variant = normalize_stage4_variant(
            stage4_variant=stage4_variant,
            stage4_1_ablation=stage4_1_ablation,
        )
        self.stage_channels = stage_channels_for_variant(stage_channels, self.stage4_variant)
        self.stage4_1_ablation = checkpoint_stage4_ablation_value(self.stage4_variant)

        # 128 -> 64
        self.stem = ConvBNReLU(
            input_channels,
            stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # 64 -> 32
        self.stage1 = nn.Sequential(
            ResidualBlock(stem_channels, self.stage_channels[0], stride=2),
            ResidualBlock(self.stage_channels[0], self.stage_channels[0], stride=1),
        )
        # 32 -> 16
        self.stage2 = nn.Sequential(
            ResidualBlock(self.stage_channels[0], self.stage_channels[1], stride=2),
            ResidualBlock(self.stage_channels[1], self.stage_channels[1], stride=1),
        )
        # 16 -> 8
        self.stage3 = nn.Sequential(
            ResidualBlock(self.stage_channels[1], self.stage_channels[2], stride=2),
            ResidualBlock(self.stage_channels[2], self.stage_channels[2], stride=1),
        )
        stage4_1_kwargs = {
            "single_conv_mode": False,
            "use_residual": True,
            "use_relu1": True,
            "use_out_relu": True,
        }
        if self.stage4_variant == "single_conv_residual":
            stage4_1_kwargs["single_conv_mode"] = True
        elif self.stage4_variant == "plain_non_residual":
            stage4_1_kwargs["use_residual"] = False
        elif self.stage4_variant == "single_conv_non_residual":
            stage4_1_kwargs["single_conv_mode"] = True
            stage4_1_kwargs["use_residual"] = False
            stage4_1_kwargs["use_out_relu"] = False
        # 8 -> 4
        self.stage4 = nn.Sequential(
            ResidualBlock(self.stage_channels[2], self.stage_channels[3], stride=2),
            ResidualBlock(
                self.stage_channels[3],
                self.stage_channels[3],
                stride=1,
                **stage4_1_kwargs,
            ),
        )

        # Fixed 4x4 -> 1x1 pool for the expected 128x128 input path.
        self.global_pool = nn.AvgPool2d(kernel_size=4, stride=4, count_include_pad=False)
        self.head_x = nn.Linear(self.stage_channels[3], 1)
        self.head_size = nn.Linear(self.stage_channels[3], 1)
        self.head_vis = nn.Linear(self.stage_channels[3], 1)

    def _forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        x = self.stage4(x)
        features["stage4"] = x
        return features

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_backbone(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_head_weight_key = prefix + "head.weight"
        old_head_bias_key = prefix + "head.bias"
        if old_head_weight_key in state_dict and old_head_bias_key in state_dict:
            weight = state_dict.pop(old_head_weight_key)
            bias = state_dict.pop(old_head_bias_key)
            split_heads = (
                ("head_x", 0),
                ("head_size", 1),
                ("head_vis", 2),
            )
            for head_name, row_idx in split_heads:
                weight_key = prefix + f"{head_name}.weight"
                bias_key = prefix + f"{head_name}.bias"
                if weight_key not in state_dict:
                    state_dict[weight_key] = weight[row_idx:row_idx + 1]
                if bias_key not in state_dict:
                    state_dict[bias_key] = bias[row_idx:row_idx + 1]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x_out = self.head_x(x)
        size_out = self.head_size(x)
        vis_out = self.head_vis(x)
        return torch.cat((x_out, size_out, vis_out), dim=1)

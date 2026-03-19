from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
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

        self.out_relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
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
        stage_channels: Tuple[int, int, int, int] = (24, 32, 64, 80),
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
            ResidualBlock(stem_channels, stage_channels[0], stride=2),
            ResidualBlock(stage_channels[0], stage_channels[0], stride=1),
        )
        # 32 -> 16
        self.stage2 = nn.Sequential(
            ResidualBlock(stage_channels[0], stage_channels[1], stride=2),
            ResidualBlock(stage_channels[1], stage_channels[1], stride=1),
        )
        # 16 -> 8
        self.stage3 = nn.Sequential(
            ResidualBlock(stage_channels[1], stage_channels[2], stride=2),
            ResidualBlock(stage_channels[2], stage_channels[2], stride=1),
        )
        # 8 -> 4
        self.stage4 = nn.Sequential(
            ResidualBlock(stage_channels[2], stage_channels[3], stride=2),
            ResidualBlock(stage_channels[3], stage_channels[3], stride=1),
        )

        # Fixed 4x4 -> 1x1 pool for the expected 128x128 input path.
        self.global_pool = nn.AvgPool2d(kernel_size=4, stride=4, count_include_pad=False)
        self.head_x = nn.Linear(stage_channels[3], 1)
        self.head_size = nn.Linear(stage_channels[3], 1)
        self.head_vis = nn.Linear(stage_channels[3], 1)

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

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from utils.follow_task import (
    follow_head_output_dim,
    follow_output_metadata,
    resolve_follow_head_type,
)


DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE = "conv_bn_relu"
QUANT_NATIVE_FOLLOW_STEM_MODES = (
    DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    "delayed_relu",
)


def normalize_quant_native_follow_stem_mode(stem_mode: str | None) -> str:
    mode = str(stem_mode or DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE).strip().lower()
    if mode not in QUANT_NATIVE_FOLLOW_STEM_MODES:
        raise ValueError(
            f"Unsupported quant-native follow stem_mode: {stem_mode}. "
            f"Expected one of {QUANT_NATIVE_FOLLOW_STEM_MODES}."
        )
    return mode


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DelayedActivationStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.pre = ConvBN(in_channels, out_channels, stride=stride)
        self.post = ConvBNReLU(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.post(x)
        return x


class PassthroughAdd(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


class StraightStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = ConvBNReLU(in_channels, out_channels, stride=2)
        self.refine = ConvBNReLU(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.refine(x)
        return x


class SingleConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualDownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main_conv1 = ConvBNReLU(in_channels, out_channels, stride=2)
        self.main_conv2 = ConvBN(out_channels, out_channels, stride=1)
        self.skip_proj = ConvBN(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.add = PassthroughAdd()
        self.out_relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_conv1(x)
        main = self.main_conv2(main)
        skip = self.skip_proj(x)
        return self.out_relu(self.add(main, skip))


class QuantNativeFollowBase(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        follow_head_type: str | None = None,
        pool_kernel: int = 8,
    ) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError(f"{model_type} expects true 1-channel grayscale input.")
        if image_size != (128, 128):
            raise ValueError(f"{model_type} assumes fixed 128x128 input.")

        self.model_type = model_type
        self.input_channels = int(input_channels)
        self.image_size = tuple(int(value) for value in image_size)
        self.follow_head_type = resolve_follow_head_type(
            follow_head_type,
            model_type=model_type,
        )
        self.pool_kernel = int(pool_kernel)

    def _init_output_head(self, in_features: int) -> None:
        self.output_head = nn.Linear(
            in_features,
            follow_head_output_dim(self.follow_head_type, model_type=self.model_type),
        )
        self.output_metadata = follow_output_metadata(
            model_type=self.model_type,
            head_type=self.follow_head_type,
        )

    def _pool_and_project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.output_head(x)


class PlainFollowNet(QuantNativeFollowBase):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        follow_head_type: str | None = None,
        stem_channels: int = 16,
        stage_channels: Tuple[int, int, int] = (24, 32, 48),
        stem_mode: str = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    ) -> None:
        super().__init__(
            model_type="plain_follow",
            input_channels=input_channels,
            image_size=image_size,
            follow_head_type=follow_head_type,
            pool_kernel=8,
        )
        self.stem_channels = int(stem_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stem_mode = normalize_quant_native_follow_stem_mode(stem_mode)

        if self.stem_mode == DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE:
            self.stem = ConvBNReLU(self.input_channels, self.stem_channels, stride=2)
        elif self.stem_mode == "delayed_relu":
            self.stem = DelayedActivationStem(self.input_channels, self.stem_channels, stride=2)
        else:
            raise AssertionError(f"Unhandled stem_mode: {self.stem_mode}")
        self.stage1 = StraightStage(self.stem_channels, self.stage_channels[0])
        self.stage2 = StraightStage(self.stage_channels[0], self.stage_channels[1])
        self.stage3 = StraightStage(self.stage_channels[1], self.stage_channels[2])
        self.global_pool = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, count_include_pad=False)
        self._init_output_head(self.stage_channels[-1])

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self._pool_and_project(x)


class PlainFollowV2Net(QuantNativeFollowBase):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        follow_head_type: str | None = None,
        stem_channels: int = 16,
        stage_channels: Tuple[int, int, int] = (24, 28, 40),
        stem_mode: str = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    ) -> None:
        super().__init__(
            model_type="plain_follow_v2",
            input_channels=input_channels,
            image_size=image_size,
            follow_head_type=follow_head_type,
            pool_kernel=8,
        )
        self.stem_channels = int(stem_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stem_mode = normalize_quant_native_follow_stem_mode(stem_mode)

        if self.stem_mode == DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE:
            self.stem = ConvBNReLU(self.input_channels, self.stem_channels, stride=2)
        elif self.stem_mode == "delayed_relu":
            self.stem = DelayedActivationStem(self.input_channels, self.stem_channels, stride=2)
        else:
            raise AssertionError(f"Unhandled stem_mode: {self.stem_mode}")
        self.stage1 = StraightStage(self.stem_channels, self.stage_channels[0])
        self.stage2 = StraightStage(self.stage_channels[0], self.stage_channels[1])
        self.stage3 = StraightStage(self.stage_channels[1], self.stage_channels[2])
        self.global_pool = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, count_include_pad=False)
        self._init_output_head(self.stage_channels[-1])

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self._pool_and_project(x)


class PlainFollowTinyNet(QuantNativeFollowBase):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        follow_head_type: str | None = None,
        stem_channels: int = 12,
        stage_channels: Tuple[int, int, int] = (12, 20, 32),
        stem_mode: str = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    ) -> None:
        super().__init__(
            model_type="plain_follow_tiny",
            input_channels=input_channels,
            image_size=image_size,
            follow_head_type=follow_head_type,
            pool_kernel=8,
        )
        self.stem_channels = int(stem_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stem_mode = normalize_quant_native_follow_stem_mode(stem_mode)
        if self.stem_channels != 12:
            raise ValueError(f"plain_follow_tiny expects stem_channels=12, got {self.stem_channels}.")
        if self.stage_channels != (12, 20, 32):
            raise ValueError(
                "plain_follow_tiny expects stage_channels=(12, 20, 32), got {}.".format(
                    self.stage_channels
                )
            )
        if self.stem_mode != DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE:
            raise ValueError(
                "plain_follow_tiny only supports stem_mode='{}'.".format(
                    DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE
                )
            )

        self.stem = ConvBNReLU(self.input_channels, self.stem_channels, stride=2)
        self.stage1 = SingleConvStage(self.stem_channels, self.stage_channels[0])
        self.stage2 = SingleConvStage(self.stage_channels[0], self.stage_channels[1])
        self.stage3 = SingleConvStage(self.stage_channels[1], self.stage_channels[2])
        self.global_pool = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, count_include_pad=False)
        self._init_output_head(self.stage_channels[-1])

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self._pool_and_project(x)


class DroNetLiteFollowNet(QuantNativeFollowBase):
    def __init__(
        self,
        *,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        follow_head_type: str | None = None,
        stem_channels: int = 16,
        stage_channels: Tuple[int, int, int] = (24, 32, 48),
        stem_mode: str = DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE,
    ) -> None:
        super().__init__(
            model_type="dronet_lite_follow",
            input_channels=input_channels,
            image_size=image_size,
            follow_head_type=follow_head_type,
            pool_kernel=8,
        )
        self.stem_channels = int(stem_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stem_mode = normalize_quant_native_follow_stem_mode(stem_mode)

        if self.stem_mode == DEFAULT_QUANT_NATIVE_FOLLOW_STEM_MODE:
            self.stem = ConvBNReLU(self.input_channels, self.stem_channels, stride=2)
        elif self.stem_mode == "delayed_relu":
            self.stem = DelayedActivationStem(self.input_channels, self.stem_channels, stride=2)
        else:
            raise AssertionError(f"Unhandled stem_mode: {self.stem_mode}")
        self.stage1 = ResidualDownsampleStage(self.stem_channels, self.stage_channels[0])
        self.stage1_refine = ConvBNReLU(self.stage_channels[0], self.stage_channels[0], stride=1)
        self.stage2 = ResidualDownsampleStage(self.stage_channels[0], self.stage_channels[1])
        self.stage2_refine = ConvBNReLU(self.stage_channels[1], self.stage_channels[1], stride=1)
        self.stage3 = StraightStage(self.stage_channels[1], self.stage_channels[2])
        self.global_pool = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, count_include_pad=False)
        self._init_output_head(self.stage_channels[-1])

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.stage1(x)
        x = self.stage1_refine(x)
        features["stage1"] = x
        x = self.stage2(x)
        x = self.stage2_refine(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage1_refine(x)
        x = self.stage2(x)
        x = self.stage2_refine(x)
        x = self.stage3(x)
        return self._pool_and_project(x)

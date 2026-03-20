from typing import Tuple

import torch
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD

from .ssd_mobilenet_v2 import MobileNetV2Backbone


class SSDMobileNetV2Raw(SSD):
    """
    SSD+MobileNetV2 model with dual forward paths:

      - `forward(list[Tensor], targets)` behaves like standard torchvision SSD
      - `forward(Tensor)` returns raw SSD heads for export:
            (locs, cls_logits)

    where:
        locs:       [B, N_boxes, 4]
        cls_logits: [B, N_boxes, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 0.1,
        image_size: Tuple[int, int] = (160, 160),
        input_channels: int = 3,
    ):
        backbone = MobileNetV2Backbone(
            width_mult=width_mult,
            image_size=image_size,
            input_channels=input_channels,
        )

        num_feature_maps = len(backbone.out_channels)
        aspect_ratios = [[1.0, 2.0, 0.5] for _ in range(num_feature_maps)]
        scales = torch.linspace(0.1, 0.7, steps=num_feature_maps + 1).tolist()
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=aspect_ratios,
            scales=scales,
        )

        super().__init__(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=image_size,
            num_classes=num_classes,
            image_mean=([0.5] * input_channels),
            image_std=([0.5] * input_channels),
        )
        self.input_channels = input_channels

    def forward_raw(self, x: torch.Tensor):
        """
        Raw head forward used by NEMO export.

        x: [B, C, H, W]
        returns:
            locs:       [B, N_boxes, 4]
            cls_logits: [B, N_boxes, num_classes]
        """
        features_list = self.backbone.forward_features(x)
        head_outputs = self.head(features_list)
        locs = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        return locs, cls_logits

    def forward(self, x, targets=None):
        if torch.is_tensor(x):
            if targets is not None:
                raise ValueError("Targets must be None when forward() receives a Tensor input.")
            return self.forward_raw(x)
        return super().forward(x, targets)

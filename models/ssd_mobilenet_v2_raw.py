from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn

from .ssd_mobilenet_v2 import create_ssd_mobilenet_v2


class SSDMobileNetV2Raw(nn.Module):
    """
    Wrapper around torchvision SSD+MobileNetV2 that exposes raw SSD heads:

        forward(x) -> (locs, cls_logits)

    where:
        locs:       [B, N_boxes, 4]
        cls_logits: [B, N_boxes, num_classes]

    No decode, no NMS, no postprocessing.
    """

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 0.5,
        image_size: Tuple[int, int] = (320, 320),
    ):
        super().__init__()
        self.ssd = create_ssd_mobilenet_v2(
            num_classes=num_classes,
            width_mult=width_mult,
            image_size=image_size,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]

        Returns:
            locs:       [B, N_boxes, 4]
            cls_logits: [B, N_boxes, num_classes]
        """

        # 1) Backbone → feature maps (same as SSD.forward)
        features = self.ssd.backbone(x)  # can be Tensor or OrderedDict

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features_list = list(features.values())  # list[Tensor]

        # 2) SSD head → raw outputs
        head_outputs = self.ssd.head(features_list)
        # In your torchvision, these are already [B, N_boxes, 4] and [B, N_boxes, C]
        locs = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        return locs, cls_logits

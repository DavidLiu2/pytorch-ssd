from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone that exposes several intermediate feature maps
    for SSD and automatically discovers channel counts.

    We:
      - split MobileNetV2 features into explicit named stages
      - run stage calls in a straight line (NEMO-friendly tracing)
      - run a dummy input once in __init__ to infer out_channels
    """

    def __init__(self, width_mult: float = 0.1, image_size: Tuple[int, int] = (160, 160)):
        super().__init__()

        # Compatible with both new and old torchvision APIs:
        # - New (>=0.13): mobilenet_v2(weights=None, width_mult=...)
        # - Old (<=0.11): mobilenet_v2(pretrained=False, width_mult=...)
        try:
            base = mobilenet_v2(weights=None, width_mult=width_mult)
        except TypeError:
            # Fallback for older torchvision (e.g. in nemoenv)
            base = mobilenet_v2(pretrained=False, width_mult=width_mult)
        
        # Shrink last conv (torchvision does NOT scale this when width_mult < 1.0)
        last_c = int(1280 * width_mult)

        conv = base.features[-1][0]
        bn   = base.features[-1][1]

        in_c = conv.in_channels

        base.features[-1][0] = nn.Conv2d(in_c, last_c, kernel_size=1, stride=1, padding=0, bias=False)
        base.features[-1][1] = nn.BatchNorm2d(last_c)


        features = self._ensure_module_visible_stem(base.features)

        # Explicit stage split points:
        #   stage0: features[0..3]
        #   stage1: features[4..6]
        #   stage2: features[7..13]
        #   stage3: features[14..end]
        self.stage0 = nn.Sequential(*features[:4])
        self.stage1 = nn.Sequential(*features[4:7])
        self.stage2 = nn.Sequential(*features[7:14])
        self.stage3 = nn.Sequential(*features[14:])

        # ---- Discover out_channels with a dummy forward ----
        self.out_channels = self._infer_out_channels(image_size)

    @staticmethod
    def _ensure_module_visible_stem(features: nn.Sequential) -> nn.Sequential:
        stem = features[0]
        stem_children = list(stem.children())
        if len(stem_children) < 2:
            return features

        conv = stem_children[0]
        bn = stem_children[1]
        if len(stem_children) >= 3 and isinstance(stem_children[2], nn.Module):
            act = stem_children[2]
        else:
            # MobileNetV2 stem activation should be ReLU6.
            act = nn.ReLU6(inplace=False)

        if isinstance(act, nn.ReLU6):
            act = nn.ReLU6(inplace=False)

        features[0] = nn.Sequential(conv, bn, act)
        return features

    def _forward_stages(self, x: torch.Tensor):
        x = self.stage0(x)
        fm0 = x
        x = self.stage1(x)
        fm1 = x
        x = self.stage2(x)
        fm2 = x
        x = self.stage3(x)
        fm3 = x
        return fm0, fm1, fm2, fm3

    def forward_features(self, x: torch.Tensor):
        fm0, fm1, fm2, fm3 = self._forward_stages(x)
        return [fm0, fm1, fm2, fm3]

    def _infer_out_channels(self, image_size: Tuple[int, int]) -> List[int]:
        """
        Run a dummy forward once to figure out how many channels
        each selected feature map has, so SSD's heads line up.
        """
        self.eval()
        with torch.no_grad():
            # dummy on CPU is fine
            x = torch.zeros(1, 3, image_size[0], image_size[1])
            out_fms = self.forward_features(x)

        return [fm.shape[1] for fm in out_fms]

    def forward(self, x: torch.Tensor) -> OrderedDict:
        fm0, fm1, fm2, fm3 = self._forward_stages(x)
        return OrderedDict([
            ("0", fm0),
            ("1", fm1),
            ("2", fm2),
            ("3", fm3),
        ])


def create_ssd_mobilenet_v2(num_classes: int = 2,
                            width_mult: float = 0.1,
                            image_size: Tuple[int, int] = (160, 160)) -> SSD:
    """
    num_classes: includes background (for COCO-style detection, person-only → 2)
    """

    backbone = MobileNetV2Backbone(width_mult=width_mult,
                                   image_size=image_size)

    # torchvision's SSD expects the backbone to have 'out_channels' attribute:
    backbone.out_channels = backbone.out_channels  # already a list

    num_feature_maps = len(backbone.out_channels)

    # ---- Anchor generator ----
    # For SSD, we need:
    #   - aspect_ratios: list (len = num_feature_maps) of lists
    #   - scales: list (len = num_feature_maps + 1)
    aspect_ratios = [[1.0, 2.0, 0.5] for _ in range(num_feature_maps)]

    # Rough scales; you can tune these later
    # Just ensure length = num_feature_maps + 1
    scales = torch.linspace(0.1, 0.7, steps=num_feature_maps + 1).tolist()

    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=aspect_ratios,
        scales=scales,
        # steps can be left None; SSD will infer from feature map size
    )

    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=image_size,
        num_classes=num_classes,
    )

    return model

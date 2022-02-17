import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn_dw import YOLOPAFPNDepthwise


class YOLOXDepthwise(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, depth=1.0, width=1.0, act="hard_swish", num_classes: int = 80):
        super().__init__()

        self.backbone = YOLOPAFPNDepthwise(
            depth=depth, width=width,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=(256, 512, 768),
            act=act
        )
        self.head = YOLOXHead(
            num_classes=num_classes, width=width,
            strides=(8, 16, 32),
            in_channels=(256, 512, 768),
            act=act
        )
        self.eval()

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)
        return outputs

import torch
import torch.nn as nn

from .darknet_dw import CSPDarknetDepthwise
from .blocks import BaseConv, CSPLayerCustom


class YOLOPAFPNDepthwise(nn.Module):
    """Default. YOLOv3 model. CSPDarknet is the default backbone of this model."""

    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_features=("dark3", "dark4", "dark5"),
                 in_channels=(256, 512, 768),
                 act="hard_swish",
                 ):
        super().__init__()
        self.backbone = CSPDarknetDepthwise(depth, width, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        assert len(in_channels) == 3

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayerCustom(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            kernel_size=5,
            dilation=1,
            depthwise=True,
            act=act,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayerCustom(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            shortcut=False,
            kernel_size=5,
            dilation=1,
            depthwise=True,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 4, 2, act=act
        )
        self.C3_n3 = CSPLayerCustom(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            kernel_size=5,
            dilation=1,
            depthwise=True,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 4, 2, act=act
        )
        self.C3_n4 = CSPLayerCustom(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            shortcut=False,
            kernel_size=5,
            dilation=1,
            depthwise=True,
            act=act,
        )

    def forward(self, x: torch.Tensor):
        # backbone
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        # x2 = 1/8, x1 = 1/16, x0 = 1/32

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], dim=1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], dim=1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

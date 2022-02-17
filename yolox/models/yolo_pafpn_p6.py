#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet_p6 import CSPDarknetCustomP6
from .network_blocks import BaseConv, CSPLayerCustom


class YOLOPAFPNCustomP6(nn.Module):
    """Default. YOLOv3 model. CSPDarknet is the default backbone of this model."""

    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_features=("dark3", "dark4", "dark5", "dark6"),
                 in_channels=(256, 512, 768, 1024),
                 act="hard_swish",
                 ):
        super().__init__()
        self.backbone = CSPDarknetCustomP6(depth, width, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        assert len(in_channels) == 4

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.C3_p5 = CSPLayerCustom(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

        self.lateral_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayerCustom(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayerCustom(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
            # int(in_channels[0] * width), int(in_channels[0] * width), 4, 2, act=act
        )
        self.C3_n3 = CSPLayerCustom(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
            # int(in_channels[1] * width), int(in_channels[1] * width), 4, 2, act=act
        )
        self.C3_n4 = CSPLayerCustom(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

        # bottom-up conv
        self.bu_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act
            # int(in_channels[2] * width), int(in_channels[2] * width), 4, 2, act=act
        )
        self.C3_n5 = CSPLayerCustom(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            shortcut=False,
            # kernel_size=5,
            dilation=1,
            # depthwise=True,
            kernel_size=3,
            # dilation=2,
            depthwise=False,
            act=act,
            # expansion=0.375,
        )

    def forward(self, x: torch.Tensor):
        #  backbone
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features
        # x3 = 1/8, x2 = 1/16, x1 = 1/32, x0 = 1/64

        fpn_out0 = self.lateral_conv0(x0)  # 1024->1024/64
        f_out0 = self.upsample(fpn_out0)  # 1024/32
        f_out0 = torch.cat([f_out0, x1], dim=1)  # 1024->1024/32
        f_out0 = self.C3_p5(f_out0)  # 1024->1024/32

        fpn_out1 = self.lateral_conv1(f_out0)  # 1024->512/32
        f_out1 = self.upsample(fpn_out1)  # 512/16
        f_out1 = torch.cat([f_out1, x2], dim=1)  # 512->1024/16
        f_out1 = self.C3_p4(f_out1)  # 1024->512/16

        fpn_out2 = self.reduce_conv1(f_out1)  # 512->256/16
        f_out2 = self.upsample(fpn_out2)  # 256/8
        f_out2 = torch.cat([f_out2, x3], dim=1)  # 256->512/8
        pan_out3 = self.C3_p3(f_out2)  # 512->256/8

        p_out2 = self.bu_conv2(pan_out3)  # 256->256/16
        p_out2 = torch.cat([p_out2, fpn_out2], dim=1)  # 256->512/16
        pan_out2 = self.C3_n3(p_out2)  # 512->512/16

        p_out1 = self.bu_conv1(pan_out2)  # 512->512/32
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # 512->1024/32
        pan_out1 = self.C3_n4(p_out1)  # 1024->1024/32

        p_out0 = self.bu_conv0(pan_out1)  # 1024->1024/64
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # 1024->1024/64
        pan_out0 = self.C3_n5(p_out0)  # 1024->1024/64

        outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        return outputs

    def forward_backbone(self, x: torch.Tensor, detach: bool = False):
        #  backbone
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features
        # x3 = 1/8, x2 = 1/16, x1 = 1/32, x0 = 1/64
        if not detach:
            outputs = (x3, x2, x1, x0)
        else:
            outputs = (x3.detach(), x2.detach(), x1.detach(), x0.detach())
        return outputs

    def forward_fpn(self, backbone_features, detach: bool = False):
        x3, x2, x1, x0 = backbone_features  # output of 'forward_backbone'

        fpn_out0 = self.lateral_conv0(x0)  # 1024->1024/64
        f_out0 = self.upsample(fpn_out0)  # 1024/32
        f_out0 = torch.cat([f_out0, x1], dim=1)  # 1024->1024/32
        f_out0 = self.C3_p5(f_out0)  # 1024->1024/32

        fpn_out1 = self.lateral_conv1(f_out0)  # 1024->512/32
        f_out1 = self.upsample(fpn_out1)  # 512/16
        f_out1 = torch.cat([f_out1, x2], dim=1)  # 512->1024/16
        f_out1 = self.C3_p4(f_out1)  # 1024->512/16

        fpn_out2 = self.reduce_conv1(f_out1)  # 512->256/16
        f_out2 = self.upsample(fpn_out2)  # 256/8
        f_out2 = torch.cat([f_out2, x3], dim=1)  # 256->512/8
        pan_out3 = self.C3_p3(f_out2)  # 512->256/8

        p_out2 = self.bu_conv2(pan_out3)  # 256->256/16
        p_out2 = torch.cat([p_out2, fpn_out2], dim=1)  # 256->512/16
        pan_out2 = self.C3_n3(p_out2)  # 512->512/16

        p_out1 = self.bu_conv1(pan_out2)  # 512->512/32
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # 512->1024/32
        pan_out1 = self.C3_n4(p_out1)  # 1024->1024/32

        p_out0 = self.bu_conv0(pan_out1)  # 1024->1024/64
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # 1024->1024/64
        pan_out0 = self.C3_n5(p_out0)  # 1024->1024/64

        # unlike forward, this sub-function returns f_out0 & f_out1 too.
        if not detach:
            outputs = (f_out0, f_out1, pan_out3, pan_out2, pan_out1, pan_out0)
        else:
            outputs = (f_out0.detach(), f_out1.detach(),
                       pan_out3.detach(), pan_out2.detach(), pan_out1.detach(), pan_out0.detach())
        return outputs

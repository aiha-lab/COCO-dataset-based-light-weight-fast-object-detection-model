#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_features=("dark3", "dark4", "dark5")):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, kernel_size=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        assert (depth == 21) or (depth == 53)
        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    @staticmethod
    def make_group_layer(in_channels: int, num_blocks: int, stride: int = 1):
        """starts with conv layer then has `num_blocks` `ResLayer`"""
        return [
            BaseConv(in_channels, in_channels * 2, kernel_size=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    @staticmethod
    def make_spp_block(filters_list, in_filters):
        m = nn.Sequential(*[
            BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            SPPBottleneck(filters_list[1], filters_list[0], activation="lrelu"),
            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
        ])
        return m

    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(self,
                 depth_multiplier: float,
                 width_multiplier: float,
                 out_features=("dark3", "dark4", "dark5"),
                 act="silu",
                 depthwise: bool = False):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(width_multiplier * 64)  # 64 for L
        base_depth = max(round(depth_multiplier * 3), 1)  # 3 for L

        # depth_multiplier = 1.25 for X: base_channels = 80
        # width_multiplier = 1.33 for X: base_depth = 4

        # stem
        self.stem = Focus(3, base_channels, kernel_size=3, act=act)  # 640 -> 320

        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),  # 320 -> 160
            CSPLayer(
                base_channels * 2,  # 128
                base_channels * 2,
                n=base_depth,  # 3
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),  # 160 -> 80
            CSPLayer(
                base_channels * 4,  # 256
                base_channels * 4,
                n=base_depth * 3,  # 9
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),  # 80 -> 40
            CSPLayer(
                base_channels * 8,  # 512
                base_channels * 8,
                n=base_depth * 3,  # 9
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),  # 40 -> 20
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),  # 20 -> 20
            CSPLayer(
                base_channels * 16,  # 1024
                base_channels * 16,
                n=base_depth,  # 3
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

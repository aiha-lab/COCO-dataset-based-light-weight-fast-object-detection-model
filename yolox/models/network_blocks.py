#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str = "silu", inplace=True):
    name = name.lower()
    if (name == "silu") or (name == "swish"):
        module = nn.SiLU(inplace=inplace)
    elif (name == "hsilu") or (name == "hswish") or (name == "hard_silu") or (name == "hard_swish"):
        module = nn.Hardswish(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif (name == "lrelu") or (name == "leaky_relu"):
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ChannelMaskLayer(nn.Module):

    def __init__(self, init_channels: int):
        super().__init__()
        self.init_channels = init_channels
        self.scale = nn.Parameter(torch.ones(init_channels, dtype=torch.float32))  # 0.0 or 1.0
        self.offset = nn.Parameter(torch.zeros(init_channels, dtype=torch.float32))

    def forward(self, x):
        scale = self.scale.view(1, -1, 1, 1)
        x = x * scale

        offset = self.offset.view(1, -1, 1, 1) * (1.0 - scale)  # only activated when scale is 0
        x = x + offset
        return x


class BaseConv(nn.Module):
    """A Conv2d -> (Sync)BatchNorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, dilation=1,
                 act="silu", prune: bool = False):
        super().__init__()
        if stride > 1:
            assert dilation == 1
        # same padding

        pad = ((kernel_size - 1) * dilation) // 2  # 3 + 1 -> 1, 3 + 2 -> 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=pad,
            groups=groups,
            bias=bias,
            dilation=(dilation, dilation),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.SyncBatchNorm(out_channels)
        # self.mask = ChannelMaskLayer(out_channels)
        self.apply_prune = prune  # flag

        self.act_type = act
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.mask(x)
        x = self.act(x)
        return x

    def fused_forward(self, x):
        x = self.conv(x)
        # x = self.mask(x)
        x = self.act(x)
        return x

    @torch.no_grad()
    def score(self):
        if not self.apply_prune:
            return None

        bn_w = self.bn.weight * self.bn.weight.grad
        bn_b = self.bn.bias * self.bn.bias.grad
        importance = torch.square(bn_w + bn_b)
        importance.masked_fill_(torch.less_equal(self.mask.scale, torch.zeros_like(self.mask.scale)), math.inf)
        return importance

    @torch.no_grad()
    def prune(self, mask: torch.Tensor):
        if self.apply_prune:
            scale = mask.float().to(self.mask.scale.device)  # 1 is kept, 0 is removed
            offset = self.bn.bias.clone() * (1.0 - scale) * self.mask.scale  # newly removed

            self.mask.scale.data.mul_(scale.detach().data)  # AND
            self.mask.offset.data.add_(offset.detach().data)


class DWConv(nn.Module):
    """Depth-wise Conv + Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels,
                              kernel_size, stride=stride, dilation=dilation, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels,
                              1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x


class DWConvNoP(nn.Module):
    """Depth-wise Conv Only"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, act="silu"):
        super().__init__()
        assert out_channels == in_channels
        self.dconv = BaseConv(in_channels, in_channels,
                              kernel_size, stride=stride, dilation=dilation, groups=in_channels, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False,
                 kernel_size=3, dilation=1,
                 act="silu", is_last: bool = False):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act_func = act
        self.depthwise = depthwise
        self.use_add = shortcut and in_channels == out_channels

        self.conv1 = BaseConv(in_channels, hidden_channels,
                              1, stride=1, act=act, prune=True)
        if depthwise:
            self.conv2 = DWConv(hidden_channels, out_channels,
                                kernel_size, stride=1, dilation=dilation, act=act)
        else:
            self.conv2 = BaseConv(hidden_channels, out_channels,
                                  kernel_size, stride=1, dilation=dilation,
                                  act=act, prune=(not self.use_add) and (not is_last))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_add:
            x = x + identity
        return x


class BottleneckCustom(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False,
                 kernel_size=3, dilation=1,
                 act="silu", is_last: bool = False):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act_func = act
        self.depthwise = depthwise
        self.use_add = shortcut and in_channels == out_channels

        self.conv1 = BaseConv(in_channels, hidden_channels,
                              1, stride=1, act=act, prune=True)
        if depthwise and (not is_last) and (not self.use_add):
            self.conv2 = DWConvNoP(hidden_channels, out_channels,
                                   kernel_size, stride=1, dilation=dilation, act=act)
        elif depthwise:  # is_last
            self.conv2 = DWConv(hidden_channels, out_channels,
                                kernel_size, stride=1, dilation=dilation, act=act)
        else:
            self.conv2 = BaseConv(hidden_channels, out_channels,
                                  kernel_size, stride=1, dilation=dilation,
                                  act=act, prune=(not self.use_add) and (not is_last))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_add:
            x = x + identity
        return x


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, kernel_size=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, kernel_size=3, stride=1, act="lrelu")

    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + identity
        return x


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        xs = [x]
        for m in self.m:
            xs.append(m(x))
        x = torch.cat(xs, dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolo-v5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",
                 kernel_size=3, dilation=1):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv2 = BaseConv(in_channels, in_channels - hidden_channels, 1, stride=1, act=act)

        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act,
                       kernel_size=kernel_size, dilation=dilation,
                       is_last=(i == n - 1))
            for i in range(n)
        ]
        self.m = nn.Sequential(*module_list)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # self.conv3 = BaseConv(in_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x_0 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_0)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class CSPLayerCustom(nn.Module):
    """C3 in yolo-v5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",
                 kernel_size=3, dilation=1):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, in_channels - hidden_channels, 1, stride=1, act=act)

        module_list = [
            BottleneckCustom(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act,
                             kernel_size=kernel_size, dilation=dilation,
                             is_last=(i == n - 1))
            for i in range(n)
        ]
        self.m = nn.Sequential(*module_list)
        # self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(in_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x_0 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_0)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # unfortunately, this order is different to torch.nn.functional.pixel_unshuffle.
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        x = self.conv(x)
        return x


class FocusCustom(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x = F.pixel_unshuffle(x, downscale_factor=2)
        x = self.conv(x)
        return x


# class SiLU(nn.Module):
#     """export-friendly version of nn.SiLU()"""
#
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)


class BaseConvWithMask(BaseConv):
    """A Conv2d -> (Sync)BatchNorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, dilation=1,
                 act="silu", prune: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, groups, bias, dilation, act, prune)
        self.register_parameter("conv_mask", nn.Parameter(torch.ones_like(self.conv.weight, dtype=torch.float32)))
        # register as parameter and do not assign to the optimizer.

    def forward(self, x):
        masked_weight = self.conv.weight * self.conv_mask
        x = self.conv._conv_forward(x, masked_weight, self.conv.bias)  # noqa
        # x = self.conv(x)
        x = self.bn(x)
        # x = self.mask(x)
        x = self.act(x)
        return x

    def fused_forward(self, x):
        masked_weight = self.conv.weight * self.conv_mask
        x = self.conv._conv_forward(x, masked_weight, self.conv.bias)  # noqa
        # x = self.conv(x)
        # x = self.mask(x)
        x = self.act(x)
        return x

    @staticmethod
    def from_base(m: BaseConv) -> "BaseConvWithMask":
        # currently do not copy parameters; only for initialization purpose.
        return BaseConvWithMask(
            m.conv.in_channels, m.conv.out_channels, m.conv.kernel_size[0], m.conv.stride[0], m.conv.groups,
            bias=(m.conv.bias is not None), dilation=m.conv.dilation[0],
            act=m.act_type, prune=m.apply_prune
        )

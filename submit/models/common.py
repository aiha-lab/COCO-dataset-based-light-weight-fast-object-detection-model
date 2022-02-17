# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import torch
import torch.nn as nn

__all__ = ["Conv", "Bottleneck", "C3", "SPPF", "Concat"]


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, (k, k), stride=(s, s), padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = self.cv2(x)
        if self.add:
            x = x + identity
        return x


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        layers = [Bottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)]
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        x1 = self.cv1(x)
        x1 = self.m(x1)
        x2 = self.cv2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.cv3(x)
        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        x = torch.cat((x, y1, y2, y3), dim=1)
        x = self.cv2(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

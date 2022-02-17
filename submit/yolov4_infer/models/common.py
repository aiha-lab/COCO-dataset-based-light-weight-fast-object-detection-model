import torch
import torch.nn as nn


class Affine2d(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_parameter("weight", nn.Parameter(torch.ones(c, dtype=torch.float32)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(c, dtype=torch.float32)))

    def forward(self, x):
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        x = x * w + b
        return x


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, *, g=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=(k, k), stride=(s, s), padding=p, groups=g, bias=True)
        # self.bn = nn.BatchNorm2d(c2)
        if act:
            self.act = nn.Mish(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.use_add = (shortcut and (c1 == c2))

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = self.cv2(x)
        if self.use_add:
            x = x + identity
        return x


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, (1, 1), stride=(1, 1), bias=True)
        self.cv3 = nn.Conv2d(c_, c_, (1, 1), stride=(1, 1), bias=True)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.bn = Affine2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.Mish(inplace=True)

        self.m = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ])

    def forward(self, x):
        x_0 = self.cv1(x)
        x_2 = self.cv2(x)

        x_1 = self.m(x_0)
        x_1 = self.cv3(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.cv4(x)
        return x


class BottleneckCSP2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, (1, 1), stride=(1, 1), bias=True)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        # self.bn = nn.BatchNorm2d(2 * c_)
        self.bn = Affine2d(2 * c_)
        self.act = nn.Mish(inplace=True)

        self.m = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ])

    def forward(self, x):
        x_0 = self.cv1(x)
        x_1 = self.m(x_0)
        x_2 = self.cv2(x_0)

        x = torch.cat((x_1, x_2), dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.cv3(x)
        return x


class SPPCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, (1, 1), stride=(1, 1), bias=True)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # self.bn = nn.BatchNorm2d(2 * c_)
        self.bn = Affine2d(2 * c_)
        self.act = nn.Mish(inplace=True)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.cv3(x1)
        x1 = self.cv4(x1)

        y1 = torch.cat([x1] + [m(x1) for m in self.m], dim=1)
        y1 = self.cv5(y1)
        y1 = self.cv6(y1)

        y2 = self.cv2(x)

        x = torch.cat((y1, y2), dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.cv7(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.d)

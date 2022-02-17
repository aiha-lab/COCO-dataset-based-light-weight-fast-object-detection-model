import torch
import torch.nn as nn


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes  # 80
        self.no = nc + 5  # number of outputs per anchor  # 85
        self.nl = len(anchors)  # number of detection layers  # 4
        self.na = len(anchors[0]) // 2  # number of anchors  # 3

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # (nl, na, 2)

        # init grid, anchor grid
        self.grid = [torch.zeros((1, 1, 1, 1, 2)) for _ in range(self.nl)]
        self.anchor_grid = [torch.zeros((1, 1, 1, 1, 2)) for _ in range(self.nl)]

        layers = [nn.Conv2d(x, self.no * self.na, (1, 1)) for x in ch]
        self.m = nn.ModuleList(layers)

    def forward(self, x):
        # x = p3, p4, p5, p6
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # (bs, 255, h, w) -> (bs, 3, h, w, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # check hw
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                # print(f"... making new grid for {x[i].shape[2:4]}")
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid_()
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        out = torch.cat(z, dim=1)
        return out

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i].to(d)) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

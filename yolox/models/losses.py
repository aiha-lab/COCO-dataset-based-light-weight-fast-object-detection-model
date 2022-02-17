#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOULoss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)  # (cx, cy, w, h)
        target = target.view(-1, 4)  # (cx, cy, w, h)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))  # top left
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))  # bottom right

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-12)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-12)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise ValueError

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, return_all: bool = False):
        if not return_all:
            fpn_outs = self.backbone(x)

            if self.training:
                assert targets is not None
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs)
            return outputs
        else:
            assert self.training
            dark3, dark4, dark5, C3_p4, C3_p3, C3_n3, C3_n4 = self.forward_backbone(x)  # noqa
            outputs = self.forward_head(x, (C3_p3, C3_n3, C3_n4), targets=targets)
            return outputs, (dark3, dark4, dark5, C3_p4, C3_p3, C3_n3, C3_n4)

    def forward_backbone(self, x, detach: bool = False):
        dark3, dark4, dark5 = self.backbone.forward_backbone(x, detach=detach)
        C3_p4, C3_p3, C3_n3, C3_n4 = self.backbone.forward_fpn((dark3, dark4, dark5), detach=detach)  # noqa
        outputs = (dark3, dark4, dark5, C3_p4, C3_p3, C3_n3, C3_n4)
        return outputs

    def forward_head(self, x, fpn_outs, targets=None):
        # fpn_outs = (C3_p3, C3_n3, C3_n4)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

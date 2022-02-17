import torch
import torch.nn as nn

from .blocks import BaseConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes,
                 width=1.0,
                 strides=(8, 16, 32),
                 in_channels=(256, 512, 1024),
                 act="silu"):
        super().__init__()

        self.n_anchors = 1  # always
        self.num_classes = num_classes
        # self.decode_in_inference = False  # KEY!

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()  # stem -> cls_conv -> cls_pred
        self.reg_preds = nn.ModuleList()  # stem -> reg_conv -> reg_pred
        self.obj_preds = nn.ModuleList()  # stem -> reg_conv -> obj_pred
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    kernel_size=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(*[
                    BaseConv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        kernel_size=3,
                        stride=1,
                        act=act,
                    ),
                    BaseConv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        kernel_size=3,
                        stride=1,
                        act=act,
                    ),
                ])
            )
            self.reg_convs.append(
                nn.Sequential(*[
                    BaseConv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        kernel_size=3,
                        stride=1,
                        act=act,
                    ),
                    BaseConv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        kernel_size=3,
                        stride=1,
                        act=act,
                    ),
                ])
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

        self.strides = strides

    def forward(self, xin):
        assert not self.training

        batch_size = xin[0].shape[0]

        reg_outputs = []
        obj_outputs = []
        cls_outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            reg_outputs.append(reg_output.view(batch_size, 4, -1))
            obj_outputs.append(obj_output.view(batch_size, 1, -1))
            cls_outputs.append(cls_output.view(batch_size, self.num_classes, -1))

        # [batch, positions, 85]
        reg_outputs = torch.cat(reg_outputs, dim=2).permute(0, 2, 1)
        obj_outputs = torch.cat(obj_outputs, dim=2).permute(0, 2, 1)
        cls_outputs = torch.cat(cls_outputs, dim=2).permute(0, 2, 1)
        return reg_outputs, obj_outputs, cls_outputs

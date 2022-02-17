#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import CustomP6v2Exp


class Exp(CustomP6v2Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "silu"

        # --------------- model config ----------------- #
        self.bn_momentum = 0.03  # default

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        self.input_size = (768, 768)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = (-4, 4)
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        # --------------  training config --------------------- #
        self.num_accumulate = 1

        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0.0
        self.basic_lr_per_img = 0.01 / (64.0 / self.num_accumulate)  # batch size 128 -> 0.02
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05  # batch size 128 -> 0.02 * 0.05 = 0.001
        self.ema = True
        self.ema_momentum = 0.9999  # 0.9998 for 128, 0.9999 for 64

        self.iou_type = "giou"  # "iou" or "giou"

        self.weight_decay = 5e-4

        self.print_interval = 25
        self.eval_interval = 5

        # -----------------  testing config ------------------ #
        self.test_size = (768, 768)
        self.test_conf = 0.001
        self.nms_threshold = 0.65

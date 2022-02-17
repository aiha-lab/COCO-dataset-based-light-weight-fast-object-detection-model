#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # start from YOLOX default channels.
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name += "_tune7"

        self.init_ckpt = "yolox_x_prune7.pth"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0  # no mosaic
        self.mixup_prob = 0.0  # no mix-up
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = False

        # --------------- model config ----------------- #
        self.bn_momentum = 0.015  # 0.03 -> 0.03 (accum 2 -> 0.03)

        # ---------------- dataloader config ---------------- #
        self.multiscale_range = (-5, 4)

        # --------------  training config --------------------- #
        self.num_accumulate = 2

        self.warmup_epochs = 0
        self.max_epoch = 50
        self.warmup_lr = 0.001  # final ended lr of pruning
        self.basic_lr_per_img = 0.0005 / (64.0 / self.num_accumulate)  # batch size 128 -> 0.001
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 50
        self.min_lr_ratio = 1.0  # batch size 128 -> 0.001
        self.ema = False  # TEST

        self.iou_type = "iou"

        self.weight_decay = 0.0
        self.print_interval = 25
        self.eval_interval = 1  # evaluate each epoch

        # -------------- fine-tuning config --------------------- #
        self.tune_start_epoch = 0  # epoch
        # self.distill_coefficient = 0.01  # global distill coefficient

        self.distill_coefficient = 0  # global distill coefficient

        self.distill_lr = 0.01
        self.distill_lr_decay = 0.9999
        self.distill_momentum = 0.9
        self.distill_weight_decay = 1e-4

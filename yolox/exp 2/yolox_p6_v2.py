#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from yolox.models.network_blocks import ChannelMaskLayer

from .base_exp import BaseExp


class CustomP6v2Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.act = 'hard_swish'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 2
        self.input_size = (768, 768)  # (height, width)
        # Actual multiscale ranges: [640-3*64, 640+3*64].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = (-3, 1)
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True

        self.random_size = None

        # --------------- model config ----------------- #
        self.bn_momentum = 0.03  # default

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
        self.ema_momentum = 0.9998

        self.iou_type = "giou"  # "iou" or "giou"

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (768, 768)
        self.test_conf = 0.001
        self.nms_threshold = 0.65

        # placeholders
        self.model = None
        self.dataset = None
        self.optimizer = None
        # self.scheduler = None

    def get_model(self, bn_momentum: float = 0.03):
        from yolox.models import YOLOXCustomP6v2, YOLOPAFPNCustomP6v2, YOLOXHeadCustom

        def init_yolo(mod):
            for m in mod.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eps = 1e-3
                    m.momentum = bn_momentum
                elif isinstance(m, nn.Conv2d) and m.in_channels == m.groups:
                    nn.init.uniform_(m.weight, -0.01, 0.01)

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 768, 1024]
            backbone = YOLOPAFPNCustomP6v2(self.depth, self.width, act=self.act, in_channels=in_channels)
            head = YOLOXHeadCustom(self.num_classes, self.width, act=self.act,
                                   strides=(8, 16, 32, 64), in_channels=in_channels)
            self.model = YOLOXCustomP6v2(backbone, head)
            self.model.head.iou_loss.loss_type = self.iou_type

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(  # noqa
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = dict(num_workers=self.data_num_workers, pin_memory=True)
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            if isinstance(self.multiscale_range, (int, float)):
                m_range = (-self.multiscale_range, self.multiscale_range)
            elif isinstance(self.multiscale_range, (tuple, list)):
                assert len(self.multiscale_range) == 2 and \
                       (self.multiscale_range[0] < 0) and (self.multiscale_range[1] > 0)
                m_range = self.multiscale_range
            else:
                raise ValueError

            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if self.random_size is None:
                min_size = int(self.input_size[0] / 64) + m_range[0]
                max_size = int(self.input_size[0] / 64) + m_range[1]
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, test_size):
        scale_y = test_size[0] / self.input_size[0]
        scale_x = test_size[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs, size=test_size, mode="bilinear", align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if self.optimizer is None:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif isinstance(v, ChannelMaskLayer):
                    pg0.append(v.offset)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)

            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch, **kwargs):
        from yolox.utils import LRScheduler

        scheduler_kwargs = dict(
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio
        )

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            **scheduler_kwargs
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, test_dev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        val_dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not test_dev else "image_info_test-dev2017.json",
            name="val2017" if not test_dev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(val_dataset)

        dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, test_dev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, test_dev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nms_threshold,
            num_classes=self.num_classes,
            testdev=test_dev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)

    def get_val_loader(self, batch_size, is_distributed):
        from yolox.data import COCODataset, TrainTransform, MosaicDetection

        val_dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val2017",
            img_size=self.test_size,
            preproc=TrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0),  # no augmentation
        )

        dataset = MosaicDetection(  # noqa
            val_dataset,
            mosaic=False,
            img_size=self.test_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.0,
                hsv_prob=0.0),
            enable_mixup=False,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        return loader

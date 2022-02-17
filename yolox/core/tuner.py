#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import datetime
import os
import time

from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from yolox.exp.yolox_base import Exp as MyExp
# from yolox.models.network_blocks import BaseConv
# from yolox.models.distill import YOLODistiller
from yolox.models.distill2 import YOLODistiller2


class Tuner:
    def __init__(self, exp: MyExp, args):
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0.0  # we track AP50
        self.no_aug = False

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(self.file_name, distributed_rank=self.rank, filename="train_log.txt", mode="a")

        # placeholders
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema_model = None  # always None

        self.teacher = None
        self.distiller = None
        self.distiller_optimizer = None

        self.train_loader = None
        self.prefetcher = None
        self.evaluator = None
        self.tblogger = None

        self.epoch = 0
        self.start_epoch = 0
        self.iter = 0
        self.max_iter = -1  # train epoch iterations
        self.num_accumulate = 1

        self._eval_first = False

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            raise e
        finally:
            self.after_train()

    def train_in_epoch(self):
        for i in range(self.start_epoch, self.max_epoch):
            self.epoch = i
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for i in range(self.max_iter):
            self.iter = i
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        images, targets = self.prefetcher.next()
        images = images.to(self.data_type, non_blocking=True)
        targets = targets.to(self.data_type, non_blocking=True)
        targets.requires_grad = False
        images, targets = self.exp.preprocess(images, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.model.train()
            outputs, intermediates = self.model(images, targets, return_all=True)

            if self.teacher is not None:
                self.teacher.eval()
                self.distiller.train()
                with torch.no_grad():
                    teacher_intermediates = self.teacher.forward_backbone(images, detach=True)

                distill_outputs = self.distiller(intermediates, teacher_intermediates)
                outputs.update(distill_outputs)

        if self.teacher is not None:
            loss = outputs["total_loss"] + outputs["dis_loss"]
        else:
            loss = outputs["total_loss"]

        if self.num_accumulate > 1:
            loss = loss / self.num_accumulate

        if self.iter % self.num_accumulate == 0:
            self.optimizer.zero_grad()
            if self.distiller_optimizer is not None:
                self.distiller_optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if self.iter % self.num_accumulate == (self.num_accumulate - 1):
            if self.epoch >= self.exp.tune_start_epoch:
                self.scaler.step(self.optimizer)
            if self.distiller_optimizer is not None:
                self.scaler.step(self.distiller_optimizer)
            self.scaler.update()

            if self.use_model_ema and (self.epoch >= self.exp.tune_start_epoch):
                self.ema_model.update(self.model)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if self.distiller_optimizer is not None:
                for param_group in self.distiller_optimizer.param_groups:
                    param_group["lr"] *= self.exp.distill_lr_decay

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=self.optimizer.param_groups[0]["lr"],
            **outputs,
        )

    def before_train(self):
        logger.info(f"Config\n... args: {self.args}\n... exp:\n...{self.exp}")

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model(bn_momentum=self.exp.bn_momentum)
        logger.info("Initial model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        if hasattr(self.exp, "init_ckpt") and (self.exp.init_ckpt is not None):
            init_ckpt = torch.load(self.exp.init_ckpt, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(init_ckpt["model"], strict=False)  # do not have channel mask, so strict=False
            logger.info(f"Model initialized with {self.exp.init_ckpt}")
            self._eval_first = True
        else:
            init_ckpt = None

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # setup teacher
        if self.exp.distill_coefficient > 0:
            teacher = copy.deepcopy(model)
            teacher.to(self.device)
            if init_ckpt is not None:
                teacher.load_state_dict(init_ckpt["model"], strict=False)
                logger.info(f"Teacher initialized with {self.exp.init_ckpt}")

            # distiller = YOLODistiller(width=self.exp.width)
            distiller = YOLODistiller2(width=self.exp.width)
            distiller.to(self.device)
            for ck, cv in distiller.coefficients.items():
                distiller.coefficients[ck] *= self.exp.distill_coefficient
            logger.info("Distiller initialized")

        # data related init
        self.no_aug = (self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs)
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        self.num_accumulate = self.exp.num_accumulate
        logger.info(f"... accumulate number of batches: {self.num_accumulate}")

        logger.info("... init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
            # teacher = DDP(teacher, device_ids=[self.local_rank], broadcast_buffers=False)
            # distiller = DDP(distiller, device_ids=[self.local_rank], broadcast_buffers=False)
            # merge this all-together should work better

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)  # may not work

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, self.exp.ema_momentum)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        if self.exp.distill_coefficient > 0:
            self.teacher = teacher
            self.teacher.eval()
            self.distiller = distiller
            self.distiller.train()

        # self.distiller_optimizer = torch.optim.SGD(
        #     self.distiller.parameters(),
        #     lr=self.exp.distill_lr, momentum=self.exp.distill_momentum,
        #     weight_decay=self.exp.distill_weight_decay, nesterov=True
        # )

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info(f"Training start...\n{model}")

    def after_train(self):
        logger.info("Training done, the best AP50 is {:.2f}".format(self.best_ap * 100))

    def before_epoch(self):
        logger.info("Start train epoch {}".format(self.epoch + 1))

        self.exp.eval_interval = 1
        if (self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs) or self.no_aug:
            logger.info("... !! no mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("... !! add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            # if not self.no_aug:
            #     self.save_ckpt(ckpt_name="last_mosaic_epoch")

        if self._eval_first:
            self.evaluate_and_save_model()
            self._eval_first = False

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        all_reduce_norm(self.model)
        self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter)
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()])

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

            logger.info(
                "{}, mem: {:.0f} Mb\n... {}\n....{}, lr: {:.6f}, size: {:d}, {}".format(
                    progress_str,
                    gpu_mem_usage(),
                    loss_str,
                    time_str,
                    self.meter["lr"].latest,
                    int(self.input_size[0]),
                    eta_str
                ))
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:  # noqa
            logger.info("Resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            self._eval_first = True

            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info("... loaded checkpoint '{}' (epoch {})".format(self.args.resume, self.start_epoch))  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("... loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                self._eval_first = True

                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model
            if is_parallel(eval_model):
                eval_model = eval_model.module

        eval_model.eval()
        ap50_95, ap50, summary = self.exp.eval(eval_model, self.evaluator, self.is_distributed)
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)  # noqa
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)  # noqa
            logger.info("\n" + summary)
        synchronize()

        # self.save_ckpt("latest", ap50 >= self.best_ap)  # only check ap50
        self.save_ckpt(f"epoch_{self.epoch + 1}_iter_{self.iter + 1}_{100 * ap50:.1f}",
                       ap50 >= self.best_ap)  # only check ap50
        self.best_ap = max(self.best_ap, ap50)  # only check ap50

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

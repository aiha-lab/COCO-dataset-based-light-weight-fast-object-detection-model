#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil

from loguru import logger
import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for k, v in model_state_dict.items():
        if k not in ckpt:
            logger.warning(f"{k} is not in the ckpt. Please double check and see if this is desired.")
            continue
        v_ckpt = ckpt[k]
        if v.shape != v_ckpt.shape:
            logger.warning(f"Shape of {k} in checkpoint is {v_ckpt.shape}, while shape of {k} in model is {v.shape}.")
            continue
        load_dict[k] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best: bool, save_dir: str, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)

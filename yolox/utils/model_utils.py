#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile
from loguru import logger
from ..models.network_blocks import BaseConv, BaseConvWithMask

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
]


def get_model_info(model, test_size):
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device="cuda")
    flops, params = profile(deepcopy(model).to("cuda"), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= test_size[0] * test_size[1] / stride / stride * 2  # GFlops
    info = "Params: {:.2f}M, GFlops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn, requires_grad: bool = False):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
        bias=True,
    ).requires_grad_(requires_grad)  # .to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    fw = torch.mm(w_bn, w_conv).view(fused_conv.weight.shape)
    fused_conv.weight.data.copy_(fw.detach().data)

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))

    fb = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    fused_conv.bias.data.copy_(fb.detach().data)

    return fused_conv.to(conv.weight.device)


def fuse_model(model, requires_grad: bool = False):
    from yolox.models.network_blocks import BaseConv

    with torch.no_grad():
        for m in model.modules():
            if type(m) is BaseConv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn, requires_grad)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fused_forward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_m_type, new_m_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recursively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def replace_conv(module, prefix=""):
    model = module
    for name, child in module.named_children():
        if isinstance(child, BaseConv) and not isinstance(child, BaseConvWithMask):
            new_child = BaseConvWithMask.from_base(child)
            setattr(module, name, new_child)
            logger.info(f"... {prefix + '.' + name} converted.")
        elif isinstance(child, nn.Sequential):
            for seq_child in child:
                replace_conv(seq_child, prefix=prefix + '.' + name)
        else:
            replace_conv(child, prefix=prefix + '.' + name)

    # double check
    for m_name, m in model.named_modules():
        if isinstance(m, BaseConv) and not isinstance(m, BaseConvWithMask):
            logger.info(f"... ! {m_name} is not converted!")

    return model

import numpy as np
import torch
import torch.nn as nn
from yolox.models import YOLOXCustomP6, YOLOPAFPNCustomP6, YOLOXHeadCustom
from yolox.utils.model_utils import fuse_model
"""
Input: pretrained weights with mask
Output: weights without mask with sparse matrix
TODO: modify load state dict (to_dense)
"""

def merge(depth: float, width: float, state_dict: dict):
    backbone = YOLOPAFPNCustomP6(depth=depth, width=width)
    head = YOLOXHeadCustom(num_classes=80, width=width,
                           strides=(8, 16, 32, 64),
                           in_channels=(256, 512, 768, 1024))
    model = YOLOXCustomP6(backbone, head)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eps = 1e-3  # CRITICAL

    model.eval().cuda()
    model.load_state_dict(state_dict, strict=False)
    model.head.decode_in_inference = False

    fused_model = fuse_model(model)
    print(f'Merge Batch norm done...\n')
    return fused_model.state_dict()

def merge_mask(params):
    #============================ Hyperparameter & load 
    postfix = 'conv.weight'
    
    before = 0
    for k, v in params.items():
        if 'mask' in k: # exclude mask parameters
            continue
        else:
            before += v.numel()
    print(f'[INFO] Number of parameters before compression: {before}\n')
    
    output = dict()
    
    #============================ Merge mask and weight
    for k, v in params.items():
        conv_k = k[:-9]+postfix
        if 'mask' in k and conv_k in params.keys():
            # get weight
            conv_w = params[conv_k].clone().detach()
            out_weight = conv_w * v
            output[conv_k] = out_weight
            print(f'[Merge mask] {conv_k:47s}')
        elif postfix in k:
            continue
        else:
            output[k] = v
    
    after_merge = 0
    for k, v in output.items():
        after_merge += v.numel()
    print(f'\n[INFO] Number of parameters after merge: {after_merge}')
    
    return output

def to_sparse(params, out_ckpt: str):
    out_state_dict = dict() # for compatability
    before = 0
    for k, v in params.items():
        before += v.numel()
    print(f'[INFO] Number of parameters before compression: {before}\n')
    
    output = dict()
    
    #============================ Merge mask and weight
    for k, v in params.items():
        output[k] = v.to_sparse().coalesce()
    
    after_comp = 0
    for k, v in output.items():
        after_comp += len(v.values())
    print(f'\n[INFO] Number of parameters after compression: {after_comp}')
    
    out_state_dict['model'] = output
    
    torch.save(out_state_dict, out_ckpt)
    print(f'\n[INFO] Saving compressed weight file to {out_ckpt}')

if __name__ == '__main__':
    pr = 49
    in_ckpt = f'direct_mask_49.pth'
    out_ckpt = f'merged_49.pth'

    state_dict = torch.load(in_ckpt)['model']
    # step 1. merge conv-bn
    fused_dict = merge(0.67, 0.75, state_dict)
    # step 2. merge conv-mask
    merged_dict = merge_mask(fused_dict)
    # step 3. to sparse
    sparse_dict = to_sparse(merged_dict, out_ckpt)

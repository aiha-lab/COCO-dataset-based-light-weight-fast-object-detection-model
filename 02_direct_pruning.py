import numpy as np
import torch

"""
Input: pretrained weights without mask
Output: weights with mask (direct_mask_49.pth)
TODO: modify load state dict (to_dense)
"""

#============================ Hyperparameter & load 
ref_param = 'best_ckpt.pth'
mask = 'mask_49.pth'
postfix = 'conv_mask'
params = torch.load(ref_param)['model']
mask_dict = torch.load(mask)
out_state_dict = dict() # for compatability

before = 0
for k, v in params.items():
    before += v.numel()
print(f'[INFO] Number of parameters before compression: {before}\n')

output = params
#============================ Merge mask and weight
elements = []
for k, v in mask_dict.items():
    output[k] = v

out_state_dict['model'] = output

torch.save(out_state_dict, f"direct_mask_49.pth")
print(f'\n[INFO] Saving compressed weight file')

import numpy as np
import torch
import matplotlib.pyplot as plt

"""
Input: pretrained weights
Output: mask
"""

#============================ Hyperparameter & load 
PLOT = False
ref_param = 'best_ckpt.pth'
postfix = 'conv_mask'
prune_amount = [49]
params = torch.load(ref_param)['model']

#============================ Generate mask 
elements = []
for k, v in params.items():
    if 'head' in k:
        continue
    else:
        if v.ndim == 4:
            elements.append(v.flatten())

elements = torch.cat(elements, dim=0).abs_().clamp_max_(1.0)
sorted_elements = elements.sort()[0]

for prune_ratio in prune_amount:
    print(f'\n\n =============== Pruning {prune_ratio}%\n')
    threshold = int(len(elements)*prune_ratio/100)
    masks = dict()
    for k, v in params.items():
        if 'head' in k: # Backbone only
            continue
        else:
            if v.ndim == 4:
                key = k[:-11]+postfix # replace conv.weight
                mask = torch.greater(torch.abs(v), sorted_elements[threshold])
                total = mask.view(-1).shape[0]
                nnz = mask.sum()
                is_useful = (nnz+nnz*4 < total)
                masks[key] = mask.detach()
                print(f'{key:47s}{total:10d}  -  {nnz:10d}  =  {total-nnz:10d}    {str(is_useful.item()):5s}')

    torch.save(masks, f'mask_{prune_ratio}.pth')

#============================ Plot 
if PLOT:
    magnitude_histogram = torch.histc(elements, bins=1000, min=0, max=elements.max())
    np_histogram = magnitude_histogram.cpu().numpy() / len(elements)
    plt.figure()
    plt.plot(np_histogram)
    plt.savefig('magnitude.png')
    plt.close()
    cumulative_histogram = np.cumsum(np_histogram)
    plt.figure()
    plt.plot(cumulative_histogram)
    plt.savefig('cumulative.png')
    plt.close()
    for i, c in enumerate(cumulative_histogram):
        print((i + 1) / 1000, c)

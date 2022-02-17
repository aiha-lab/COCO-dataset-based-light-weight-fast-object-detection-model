import numpy as np
import torch
import matplotlib.pyplot as plt

params = torch.load("submit/yolox_l_ms_merged.pth")

elements = []
for k, v in params.items():
    # print(k, v.shape)
    if v.ndim == 4:
        elements.append(v.flatten())

elements = torch.cat(elements, dim=0).abs_().clamp_max_(1.0)
print(elements.shape)
print(elements.min(), elements.max(), elements.mean())
magnitude_histogram = torch.histc(elements, bins=1000, min=0, max=elements.max())
# print(magnitude_histogram.long())
#
np_histogram = magnitude_histogram.cpu().numpy() / len(elements)
# plt.figure()
# plt.plot(np_histogram)
# plt.show()
cumulative_histogram = np.cumsum(np_histogram)
# plt.figure()
# plt.plot(cumulative_histogram)
# plt.show()
for i, c in enumerate(cumulative_histogram):
    print((i + 1) / 1000, c)
#
new_params = dict()
for k, v in params.items():
    if v.ndim != 4:
        new_params[k] = v
    else:
        mask = torch.greater(torch.abs(v), 0.001)
        v_masked = v * mask
        new_params[k] = v_masked.detach()

torch.save(new_params, "submit/yolox_l_ms_merged_cut_0.001.pth")

import math
import os

import numpy as np
from PIL import Image
import torch


def yolox_load_one_image_pil(img_size: int, data_dir: str, img_file: str):
    """Using Pillow-SIMD as image reshape operation.
    - resize w/ preserving the ratio
    - pad into gray-filled squared input (filled with 114)
    """
    img_path = os.path.join(data_dir, img_file)
    img = Image.open(img_path).convert("RGB")  # RGB
    w, h = img.size
    if w > h:
        new_w = img_size
        new_h = int(h * new_w / w)
    else:
        new_h = img_size
        new_w = int(w * new_h / h)
    resized_img = img.resize((new_w, new_h), resample=Image.BILINEAR)  # cv2.INTER_LINEAR
    return resized_img, (h, w, img_file, new_h, new_w)


def yolox_collate_batch(img_size: int, batch):
    # batch: img, (h, w, img_file, new_h, new_w)
    img_list = [b[0] for b in batch]
    img_info = [(b[1][0], b[1][1], b[1][2], 0, 0) for b in batch]

    max_h = max([b[1][3] for b in batch])
    max_w = max([b[1][4] for b in batch])

    if img_size % 64 == 0:
        # set to multiple of 64
        max_h = int(math.ceil(max_h / 64) * 64)
        max_w = int(math.ceil(max_w / 64) * 64)
    else:  # img_size % 32 == 0:
        # set to multiple of 32
        max_h = int(math.ceil(max_h / 32) * 32)
        max_w = int(math.ceil(max_w / 32) * 32)

    batch_size = len(img_list)
    # batch = np.full((batch_size, img_size, img_size, 3), fill_value=114, dtype=np.uint8)
    batch = np.full((batch_size, max_h, max_w, 3), fill_value=114, dtype=np.uint8)
    for i, img in enumerate(img_list):
        # PIL (RGB -> BGR)
        w, h = img.size
        batch[i, :h, :w, :] = np.asarray(img)[..., ::-1]  # fit to top-left, RGB -> BGR
    batch = batch.transpose((0, 3, 1, 2))  # (n, h, w, c) -> (n, c, h, w)
    batch = np.ascontiguousarray(batch, dtype=np.float32)  # no normalize!

    batch = torch.from_numpy(batch)
    return batch, img_info

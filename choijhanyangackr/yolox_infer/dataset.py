import os
from functools import partial

import imagesize
from torch.utils.data import Dataset, DataLoader

from .preprocess_utils import yolox_load_one_image_pil, yolox_collate_batch

IMG_EXT = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes

__all__ = ["YOLOXImageFolderDataset", "YOLOXDataLoader"]


class YOLOXImageFolderDataset(Dataset):

    def __init__(self, data_dir: str, img_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        # files = []
        # file_sizes = []
        files_and_sizes = []
        for f in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, f)) and (f.split('.')[-1].lower() in IMG_EXT):
                # files.append(f)
                w, h = imagesize.get(os.path.join(self.data_dir, f))
                # file_sizes.append((h, w))
                files_and_sizes.append((f, h, w))

        sorted_files = sorted(files_and_sizes, key=lambda x: x[1] / x[2])

        # self.img_files = files
        # self.img_file_sizes = file_sizes
        self.img_files = [f[0] for f in sorted_files]
        self.img_files_sizes = [(f[1], f[2]) for f in sorted_files]

        print(f"Images in {data_dir}: {len(self)}")
        print(f"Image size: {img_size}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img, (h, w, img_file, new_h, new_w) = yolox_load_one_image_pil(self.img_size, self.data_dir, self.img_files[idx])
        return img, (h, w, img_file, new_h, new_w)


class YOLOXDataLoader(DataLoader):

    def __init__(self, *args, img_size: int, **kwargs):
        collate_fn = partial(yolox_collate_batch, img_size)
        super().__init__(*args, collate_fn=collate_fn, **kwargs)

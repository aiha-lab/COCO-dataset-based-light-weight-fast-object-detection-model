import os
import imagesize
import math
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
import nvidia.dali.types as DTypes

IMG_EXT = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes


@pipeline_def
def yolox_load_one_image_dali(img_size: int, data_dir: str, img_files: str, queue_depth: int = 2):
    jpgs, _ = fn.readers.file(file_root=data_dir, files=img_files, prefetch_queue_depth=queue_depth)
    jpgs = fn.decoders.image(jpgs, device="mixed", output_type=DTypes.DALIImageType.BGR)
    jpgs = fn.resize(jpgs, interp_type=DTypes.DALIInterpType.INTERP_LINEAR, resize_longer=img_size)
    jpgs = fn.pad(jpgs, fill_value=114, axes=(0, 1), shape=(img_size, img_size))
    jpgs = fn.cast(jpgs, dtype=DTypes.DALIDataType.FLOAT)

    return jpgs


class YOLOXDaliImageFolderPipeline(object):

    def __init__(self, data_dir: str, img_size: int,
                 batch_size: int, queue_depth: int = 2, num_threads: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        files = []
        file_sizes = []
        for f in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, f)) and (f.split('.')[-1].lower() in IMG_EXT):
                files.append(f)
                w, h = imagesize.get(os.path.join(self.data_dir, f))
                file_sizes.append((h, w))
        self.img_files = files
        self.img_file_sizes = file_sizes

        print(f"Images in {data_dir}: {len(self)}")
        print(f"Image size: {img_size}")

        self.pipeline = yolox_load_one_image_dali(
            batch_size=batch_size, num_threads=num_threads, device_id=0,
            img_size=img_size, data_dir=data_dir, img_files=self.img_files, queue_depth=queue_depth)
        self.pipeline.build()

    def __len__(self):
        return len(self.img_files)


class YOLOXDaliIterator(object):

    def __init__(self, pipeline: YOLOXDaliImageFolderPipeline):
        self.batch_size = pipeline.pipeline.max_batch_size
        self.img_files = pipeline.img_files
        self.img_file_sizes = pipeline.img_file_sizes
        self.count = 0

        self.dali_iter = DALIGenericIterator(
            pipelines=pipeline.pipeline,
            size=len(self.img_files),
            output_map=["data"],
            last_batch_policy="PARTIAL",
            prepare_first_batch=True,
        )

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(len(self.img_files) / self.batch_size))

    def __next__(self):
        try:
            data = next(self.dali_iter)
        except StopIteration:
            self.dali_iter.reset()
            self.count = 0
            raise StopIteration

        data = data[0]["data"]
        _valid_count = min(self.count + self.batch_size, len(self.img_files)) - self.count
        img_files = self.img_files[self.count:self.count + _valid_count]
        img_file_sizes = self.img_file_sizes[self.count:self.count + _valid_count]
        self.count += _valid_count

        if data.shape[0] > _valid_count:
            data = data[:_valid_count]

        img_info = [(h, w, p) for (h, w), p in zip(img_file_sizes, img_files)]

        data = data.permute(0, 3, 1, 2)
        return data, img_info

# if __name__ == '__main__':
#     pp = YOLOXImageFolderDaliPipeline("/home/khshim/data/coco2017/val2017", img_size=640)
#     print(len(pp))
#     iterator = YOLOXDaliIterator(pp)
#     print(iterator)
#
#     for i, (dd, ff) in enumerate(iterator):
#         print(i, type(dd), dd.shape, dd.device, dd.dtype, ff)

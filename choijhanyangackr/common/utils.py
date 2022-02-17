import torch

__all__ = ["convert_to_coco_format"]

COCO_CLASS_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    boxes[:, 2].sub_(boxes[:, 0])
    boxes[:, 3].sub_(boxes[:, 1])
    return boxes


def clip_boxes(boxes, img_h: int, img_w: int):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[..., 0].clamp_(0, img_w)
    boxes[..., 1].clamp_(0, img_h)
    boxes[..., 2].clamp_(0, img_w)
    boxes[..., 3].clamp_(0, img_h)
    return boxes


def convert_to_coco_format(outputs, img_info, img_size, class_ids=None):
    """
    outputs:        [(positions, 7), ... ] of #num_images
                    (x1, y1, x2, y2, obj, cls, label)
    img_info:       [(h, w, file_name), ... ] of #num_images
    """
    data_list = []

    if class_ids is None:
        class_ids = COCO_CLASS_ID

    for output, (img_h, img_w, img_path) in zip(outputs, img_info):
        image_id = int(img_path.split("_")[-1].split(".")[0])  # remove prefix, suffix -> should be "INT"

        if output is None:
            predict_data = {
                "image_id": image_id,
                "category_id": 0,
                "bbox": [0, 0, 0, 0],
                "score": 0.0,
            }
            data_list.append(predict_data)
            continue
        output = output.cpu()
        boxes = output[:, :4]
        scale = min(img_size / float(img_h), img_size / float(img_w))
        boxes /= scale
        # boxes = clip_boxes(boxes, img_h, img_w)
        boxes = xyxy2xywh(boxes)

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        boxes = boxes.numpy().tolist()
        cls = cls.numpy().tolist()
        scores = scores.numpy().tolist()

        for i in range(len(boxes)):
            predict_data = {
                "image_id": image_id,
                "category_id": class_ids[int(cls[i])],
                "bbox": boxes[i],
                "score": scores[i],
            }
            data_list.append(predict_data)

    return data_list

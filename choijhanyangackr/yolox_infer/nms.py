try:
    from pt_soft_nms import soft_nms
    from pt_soft_nms import batched_soft_nms

    SOFT_NMS_SET = True
except ImportError:
    soft_nms = batched_soft_nms = None
    SOFT_NMS_SET = False
    # install github.com/MrParosk/soft_nms

from torchvision.ops import nms, batched_nms


def nms_wrapper(boxes, scores,
                iou_threshold,  # sigma for soft_nms
                conf_threshold: float = 0.001,  # only for soft_nms
                soft: bool = False):
    if not soft:
        return nms(boxes, scores, iou_threshold)
    elif not SOFT_NMS_SET:
        raise ValueError("Soft-NMS is not installed, but using soft_nms.")
    else:
        sigma = iou_threshold
        boxes = boxes.cpu()
        scores = scores.cpu()
        return soft_nms(boxes, scores, sigma, conf_threshold)


def batched_nms_wrapper(boxes, scores, idxs,
                        iou_threshold,  # sigma for soft_nms
                        conf_threshold: float = 0.001,  # only for soft_nms
                        soft: bool = False):
    if not soft:
        return batched_nms(boxes, scores, idxs, iou_threshold)
    elif not SOFT_NMS_SET:
        raise ValueError("Soft-NMS is not installed, but using soft_nms.")
    else:
        sigma = iou_threshold
        boxes = boxes.cpu()
        scores = scores.cpu()
        idxs = idxs.cpu()
        return batched_soft_nms(boxes, scores, idxs, sigma, conf_threshold)

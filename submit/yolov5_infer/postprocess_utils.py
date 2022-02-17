import torch
from torchvision.ops import nms, batched_nms


def yolov5_postprocess_output_torch_batch(output):
    """
    output:         (batch_size, positions, 85)
    """
    output = output.float()
    reg_output, obj_output, cls_output = torch.split(output, [4, 1, 80], dim=-1)

    reg_boxes = torch.empty_like(reg_output)
    w_half = reg_output[..., 2] / 2
    h_half = reg_output[..., 3] / 2
    reg_boxes[..., 0] = reg_output[..., 0] - w_half
    reg_boxes[..., 1] = reg_output[..., 1] - h_half
    reg_boxes[..., 2] = reg_output[..., 0] + w_half
    reg_boxes[..., 3] = reg_output[..., 1] + h_half

    obj_conf = obj_output  # already passed sigmoid
    cls_conf = cls_output * obj_conf  # already passed sigmoid

    return reg_boxes, obj_conf, cls_conf


def yolov5_nms_torch_batch(reg_boxes, obj_conf, cls_conf,
                           nms_threshold: float = 0.65,
                           conf_threshold: float = 0.001,
                           max_num_nms: int = 5000,
                           max_num_det: int = 300,
                           multi_class: bool = False,
                           rmmop=None,
                           class_agnostic: bool = False):
    """COCO output is class-agnostic=False.
    reg_boxes:      (batch_size, positions, 4)  float
    obj_conf:       (batch_size, positions, 1)  float
    cls_conf:       (batch_size, positions, 1)  float, already multiplied by obj_conf
    cls_pred:       (batch_size, positions, 1)  long
    """
    batch_size = cls_conf.shape[0]
    output = [None for _ in range(batch_size)]  # placeholder

    for i in range(batch_size):
        if rmmop is not None:
            rmmop_r1, rmmop_r2 = rmmop
            cls_conf_sorted, cls_conf_indices = torch.sort(cls_conf[i], dim=-1, descending=True)  # (p, c)
            rmmop_r1_mask = torch.greater_equal(cls_conf_sorted[:, 0],
                                                cls_conf_sorted[:, 1] * rmmop_r1)  # (p,)
            rmmop_r2_mask = torch.greater_equal(torch.square(obj_conf[i].squeeze()),
                                                cls_conf_sorted[:, 0] * rmmop_r2)  # (p,), comparing square!
            rmmop_mask = torch.logical_and(rmmop_r1_mask, rmmop_r2_mask)
            detections = torch.cat((reg_boxes[i], obj_conf[i], cls_conf_sorted[:, 0].unsqueeze(-1),
                                    cls_conf_indices[:, 0].unsqueeze(-1).float()), dim=1)
            detections = detections[rmmop_mask]  # (positions, 7)
        elif not multi_class:
            cls_conf_i, cls_pred_i = torch.max(cls_conf[i], dim=-1, keepdim=True)  # (p, 1)
            conf_mask = torch.greater_equal(cls_conf_i.squeeze(), conf_threshold)
            detections = torch.cat((reg_boxes[i], obj_conf[i], cls_conf_i, cls_pred_i.float()), dim=1)
            detections = detections[conf_mask]  # (positions, 7)
        else:
            conf_mask_p, conf_mask_c = torch.greater_equal(cls_conf[i], conf_threshold).nonzero().T
            detections = torch.cat((reg_boxes[i][conf_mask_p],
                                    obj_conf[i][conf_mask_p],
                                    cls_conf[i][conf_mask_p, conf_mask_c].unsqueeze(-1),
                                    conf_mask_c.float().unsqueeze(-1)), dim=1)

        if not detections.size(0):
            continue

        if (max_num_nms > 0) and (detections.size(0) > max_num_nms):
            conf_sort = torch.argsort(detections[:, 5], descending=True)
            conf_sort = conf_sort[:max_num_nms]
            detections = detections[conf_sort]

        if class_agnostic:
            nms_out_indices = nms(
                detections[:, :4],
                detections[:, 5],
                nms_threshold,
            )
        else:  # default for COCO
            nms_out_indices = batched_nms(
                detections[:, :4],
                detections[:, 5],
                detections[:, 6],
                nms_threshold,
            )

        if nms_out_indices.size(0) > max_num_det:
            nms_out_indices = nms_out_indices[:max_num_det]

        detections = detections[nms_out_indices]
        output[i] = detections

    return output

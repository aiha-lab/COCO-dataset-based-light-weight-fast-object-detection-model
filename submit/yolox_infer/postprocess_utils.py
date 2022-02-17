import torch

from .nms import nms_wrapper, batched_nms_wrapper


def yolox_generate_grid(img_size, strides=(8, 16, 32), dtype=torch.float32):
    grids = []
    scales = []

    if isinstance(img_size, int):
        img_size = (img_size, img_size)  # (h, w)

    for s in strides:
        h = img_size[0] // s
        w = img_size[1] // s
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        scales.append(torch.full((1, grid.shape[1], 1), s))

    grids = torch.cat(grids, dim=1).to(dtype)  # (1, positions, 2) xy
    scales = torch.cat(scales, dim=1).to(dtype)  # (1, positions, 1)

    return grids, scales


def yolox_postprocess_output_torch_batch(reg_output, obj_output, cls_output, grids, scales):
    """
    reg_output:     (batch_size, positions, 4)
    obj_output:     (batch_size, positions, 1)  # before sigmoid
    cls_output:     (batch_size, positions, 80)  # before sigmoid
    """
    reg_output = reg_output.float()
    obj_output = obj_output.float()
    cls_output = cls_output.float()

    reg_output[..., :2].add_(grids).mul_(scales)
    reg_output[..., 2:].exp_().mul_(scales / 2)  # half w, h

    reg_boxes = torch.empty_like(reg_output)
    reg_boxes[..., 0] = reg_output[..., 0] - reg_output[..., 2]
    reg_boxes[..., 1] = reg_output[..., 1] - reg_output[..., 3]
    reg_boxes[..., 2] = reg_output[..., 0] + reg_output[..., 2]
    reg_boxes[..., 3] = reg_output[..., 1] + reg_output[..., 3]

    obj_conf = obj_output.sigmoid_()
    cls_conf = cls_output.sigmoid_() * obj_conf
    #
    # cls_conf, cls_pred = torch.max(cls_output, dim=-1, keepdim=True)  # (batch_size, positions, 1)
    # cls_conf = cls_conf.sigmoid_()

    return reg_boxes, obj_conf, cls_conf


def yolox_nms_torch_batch(reg_boxes, obj_conf, cls_conf,
                          nms_threshold: float = 0.65,
                          conf_threshold: float = 0.001,
                          soft: bool = False,
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
            nms_out_indices = nms_wrapper(
                detections[:, :4],
                detections[:, 5],
                nms_threshold,
                conf_threshold,
                soft=soft
            )
        else:  # default for COCO
            nms_out_indices = batched_nms_wrapper(
                detections[:, :4],
                detections[:, 5],
                detections[:, 6],
                nms_threshold,
                conf_threshold,
                soft=soft
            )

        if nms_out_indices.size(0) > max_num_det:
            nms_out_indices = nms_out_indices[:max_num_det]

        detections = detections[nms_out_indices]
        output[i] = detections

    return output

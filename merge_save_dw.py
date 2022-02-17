import torch
import torch.nn as nn

from yolox.models import YOLOXCustom, YOLOPAFPNCustom, YOLOXHeadCustom
from yolox.utils.model_utils import fuse_model


# from submit.yolox_infer.models import YOLOX as YOLOXInfer


def merge(depth: float, width: float, ckpt: str, out_ckpt: str):
    backbone = YOLOPAFPNCustom(depth=depth, width=width)
    head = YOLOXHeadCustom(num_classes=80, width=width)
    model = YOLOXCustom(backbone, head)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eps = 1e-3  # CRITICAL

    model.eval().cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cuda")["model"], strict=True)
    model.head.decode_in_inference = False

    fused_model = fuse_model(model)
    torch.save(fused_model.state_dict(), out_ckpt)

    # infer_model = YOLOXInfer(depth=depth, width=width)
    # infer_model.eval().cuda()
    # infer_model.load_state_dict(fused_model.state_dict(), strict=True)
    #
    # with torch.no_grad():
    #     dummy_input = torch.empty((1, 3, 640, 640), dtype=torch.float32).uniform_(0.0, 255.0)
    #     dummy_input = dummy_input.cuda()
    #
    #     out_model = model(dummy_input)
    #     print(out_model.shape)
    #
    #     out_infer_reg, out_infer_obj, out_infer_cls = infer_model(dummy_input)
    #     out_infer = torch.cat([out_infer_reg, out_infer_obj.sigmoid(), out_infer_cls.sigmoid()], dim=2)
    #     print(out_infer.shape)
    #
    #     diff = torch.abs(out_model - out_infer)
    #     print(diff.min(), diff.max(), diff.mean())

    return model


if __name__ == '__main__':
    # YOLOX-L-DW
    merge(1.0, 1.0, "yolox_l_dw_kh.pth", "yolox_l_dw_kh_merged.pth")

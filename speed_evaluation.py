import time

import torch

from yolox.models import YOLOPAFPN, YOLOXHead, YOLOX
from yolox.models import YOLOPAFPNCustom, YOLOXHeadCustom, YOLOXCustom
from yolox.utils.model_utils import fuse_model


def build_yolox():
    # L config
    backbone = YOLOPAFPN(depth=1.0, width=1.0, act="silu", in_channels=(256, 512, 1024))
    head = YOLOXHead(num_classes=80, width=1.0, act="silu", in_channels=(256, 512, 1024))
    net = YOLOX(backbone, head)
    return net


def build_yolox_custom():
    # L config
    backbone = YOLOPAFPNCustom(depth=1.0, width=1.0, act="hard_swish", in_channels=(256, 512, 768))
    head = YOLOXHeadCustom(num_classes=80, width=1.0, act="hard_swish", in_channels=(256, 512, 768))
    net = YOLOXCustom(backbone, head)
    return net


def count_module(m) -> int:
    count = 0
    for p_name, p in m.named_parameters():
        count += p.numel()
    return count


def run_module(m, d_in):
    with torch.no_grad():
        for _ in range(50):
            _ = m(d_in)

        torch.cuda.synchronize()
        start_time = time.process_time_ns()
        for _ in range(250):
            d_out = m(d_in)
            torch.cuda.synchronize()
        duration = time.process_time_ns() - start_time
    return d_out, float(duration / 250 / 1e3), count_module(m)


def total_latency(net, use_half: bool = False, input_size: int = 640):
    net: YOLOX
    if use_half:
        net = net.eval().half().to("cuda")
        d_in = torch.empty((1, 3, input_size, input_size)).normal_(0, 1).half().to("cuda")
    else:
        net = net.eval().float().to("cuda")
        d_in = torch.empty((1, 3, input_size, input_size)).normal_(0, 1).float().to("cuda")
    net.head.decode_in_inference = False

    stem_out, stem_time, stem_count = run_module(net.backbone.backbone.stem, d_in)
    print(f"Stem time: {stem_time:.3f} us "
          f"(out shape: {tuple(stem_out.shape)}) "
          f"(count: {stem_count})")

    dark2_out0, dark2_out0_time, dark2_out0_count = run_module(net.backbone.backbone.dark2[0], stem_out)
    print(f"Dark2-Conv time: {dark2_out0_time:.3f} us "
          f"(out shape: {tuple(dark2_out0.shape)}) "
          f"(count: {dark2_out0_count})")
    dark2_out1, dark2_out1_time, dark2_out1_count = run_module(net.backbone.backbone.dark2[1], dark2_out0)
    print(f"Dark2-CSP time: {dark2_out1_time:.3f} us "
          f"(out shape: {tuple(dark2_out1.shape)}) "
          f"(count: {dark2_out1_count})")

    dark3_out0, dark3_out0_time, dark3_out0_count = run_module(net.backbone.backbone.dark3[0], dark2_out1)
    print(f"Dark3-Conv time: {dark3_out0_time:.3f} us "
          f"(out shape: {tuple(dark3_out0.shape)}) "
          f"(count: {dark3_out0_count})")
    dark3_out1, dark3_out1_time, dark3_out1_count = run_module(net.backbone.backbone.dark3[1], dark3_out0)
    print(f"Dark3-CSP time: {dark3_out1_time:.3f} us "
          f"(out shape: {tuple(dark3_out1.shape)}) "
          f"(count: {dark3_out1_count})")
    x2 = dark3_out1

    dark4_out0, dark4_out0_time, dark4_out0_count = run_module(net.backbone.backbone.dark4[0], dark3_out1)
    print(f"Dark4-Conv time: {dark4_out0_time:.3f} us "
          f"(out shape: {tuple(dark4_out0.shape)}) "
          f"(count: {dark4_out0_count})")
    dark4_out1, dark4_out1_time, dark4_out1_count = run_module(net.backbone.backbone.dark4[1], dark4_out0)
    print(f"Dark4-CSP time: {dark4_out1_time:.3f} us "
          f"(out shape: {tuple(dark4_out1.shape)}) "
          f"(count: {dark4_out1_count})")
    x1 = dark4_out1

    dark5_out0, dark5_out0_time, dark5_out0_count = run_module(net.backbone.backbone.dark5[0], dark4_out1)
    print(f"Dark5-Conv time: {dark4_out0_time:.3f} us "
          f"(out shape: {tuple(dark5_out0.shape)}) "
          f"(count: {dark5_out0_count})")
    dark5_out1, dark5_out1_time, dark5_out1_count = run_module(net.backbone.backbone.dark5[1], dark5_out0)
    print(f"Dark5-SPP time: {dark5_out1_time:.3f} us "
          f"(out shape: {tuple(dark5_out1.shape)}) "
          f"(count: {dark5_out1_count})")
    dark5_out2, dark5_out2_time, dark5_out2_count = run_module(net.backbone.backbone.dark5[2], dark5_out1)
    print(f"Dark5-CSP time: {dark5_out2_time:.3f} us "
          f"(out shape: {tuple(dark5_out2.shape)}) "
          f"(count: {dark5_out2_count})")
    x0 = dark5_out2

    fpn_out0, f_out0_0_time, f_out0_0_count = run_module(net.backbone.lateral_conv0, x0)
    f_out0, f_out0_1_time, f_out0_1_count = run_module(net.backbone.upsample, fpn_out0)
    print(f"FPN-lateral-upsample time: {f_out0_0_time + f_out0_1_time:.3f} us "
          f"(out shape: {tuple(f_out0.shape)}) "
          f"(count: {f_out0_0_count + f_out0_1_count})")
    f_out0 = torch.cat([f_out0, x1], dim=1)
    f_out0, f_out0_2_time, f_out0_2_count = run_module(net.backbone.C3_p4, f_out0)
    print(f"FPN-C3-p4 time: {f_out0_2_time:.3f} us "
          f"(out shape: {tuple(f_out0.shape)}) "
          f"(count: {f_out0_2_count})")

    fpn_out1, f_out1_0_time, f_out1_0_count = run_module(net.backbone.reduce_conv1, f_out0)
    f_out1, f_out1_1_time, f_out1_1_count = run_module(net.backbone.upsample, fpn_out1)
    print(f"FPN-reduce-upsample time: {f_out1_0_time + f_out1_1_time:.3f} us "
          f"(out shape: {tuple(f_out1.shape)}) "
          f"(count: {f_out1_0_count + f_out1_1_count})")
    f_out1 = torch.cat([f_out1, x2], dim=1)
    pan_out2, f_out1_2_time, f_out1_2_count = run_module(net.backbone.C3_p3, f_out1)
    print(f"FPN-C3-p3 time: {f_out1_2_time:.3f} us "
          f"(out shape: {tuple(pan_out2.shape)}) "
          f"(count: {f_out1_2_count})")

    p_out1, p_out1_0_time, p_out1_0_count = run_module(net.backbone.bu_conv2, pan_out2)
    print(f"FPN-bu-conv2 time: {p_out1_0_time:.3f} us "
          f"(out shape: {tuple(p_out1.shape)}) "
          f"(count: {p_out1_0_count})")
    p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
    pan_out1, p_out1_1_time, p_out1_1_count = run_module(net.backbone.C3_n3, p_out1)
    print(f"FPN-C3-n3 time: {p_out1_1_time:.3f} us "
          f"(out shape: {tuple(pan_out1.shape)}) "
          f"(count: {p_out1_1_count})")

    p_out0, p_out0_0_time, p_out0_0_count = run_module(net.backbone.bu_conv1, pan_out1)
    print(f"FPN-bu-conv1 time: {p_out0_0_time:.3f} us "
          f"(out shape: {tuple(p_out0.shape)}) "
          f"(count: {p_out0_0_count})")
    p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
    pan_out0, p_out0_1_time, p_out0_1_count = run_module(net.backbone.C3_n4, p_out0)
    print(f"FPN-C3-n4 time: {p_out0_1_time:.3f} us "
          f"(out shape: {tuple(pan_out0.shape)}) "
          f"(count: {p_out0_1_count})")

    head_out, head_time, head_count = run_module(net.head, (pan_out2, pan_out1, pan_out0))
    print(f"Head time: {head_time:.3f} us "
          f"(out shape: {tuple(head_out.shape)}) "
          f"(count: {head_count})")

    total_time = (
            stem_time +
            dark2_out0_time + dark2_out1_time +
            dark3_out0_time + dark3_out1_time +
            dark4_out0_time + dark4_out1_time +
            dark5_out0_time + dark5_out1_time + dark5_out2_time +
            f_out0_0_time + f_out0_1_time + f_out0_2_time +
            f_out1_0_time + f_out1_1_time + f_out1_2_time +
            p_out1_0_time + p_out1_1_time +
            p_out0_0_time + p_out0_1_time +
            head_time
    )
    total_count = (
            stem_count +
            dark2_out0_count + dark2_out1_count +
            dark3_out0_count + dark3_out1_count +
            dark4_out0_count + dark4_out1_count +
            dark5_out0_count + dark5_out1_count + dark5_out2_count +
            f_out0_0_count + f_out0_1_count + f_out0_2_count +
            f_out1_0_count + f_out1_1_count + f_out1_2_count +
            p_out1_0_count + p_out1_1_count +
            p_out0_0_count + p_out0_1_count +
            head_count
    )
    print(f"--------------------------------\n"
          f"Total time: {total_time:.3f} us "
          f"(count: {total_count})")


if __name__ == '__main__':
    yolo = build_yolox()
    yolo = fuse_model(yolo, requires_grad=False)
    print(f"Model:\n{yolo}")
    total_latency(yolo, use_half=True, input_size=640)

    yolo = build_yolox_custom()
    yolo = fuse_model(yolo, requires_grad=False)
    print(f"Model:\n{yolo}")
    total_latency(yolo, use_half=True, input_size=640)

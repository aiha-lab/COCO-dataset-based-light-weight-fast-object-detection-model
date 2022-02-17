import argparse
import json
import pprint

import torch
import torch.nn as nn

from common.profile import TimeTracker, time_synchronized
from common.utils import convert_to_coco_format
from common.evaluator import COCOEvaluator
from yolox_infer.models import YOLOX, YOLOXDepthwise, YOLOXP6, YOLOXP6v2
from yolox_infer.dataset import YOLOXImageFolderDataset, YOLOXDataLoader
from yolox_infer.postprocess_utils import (yolox_generate_grid, yolox_nms_torch_batch,
                                           yolox_postprocess_output_torch_batch)

def count_sparse_params(m) -> int:
    count = 0
    for key, p in m.items():
        count += len(p.coalesce().values())
    print(f"Saprse Parameters: {count}")
    return count

def count_params(m) -> int:
    count = 0
    for p in m.parameters():
        count += p.numel()
    print(f"Parameters: {count}")
    return count


def build_yolox(cfg):
    d = cfg["model"]["depth"]
    w = cfg["model"]["width"]
    model_type = cfg["model"]["type"].lower()

    if "dw" in model_type:
        model = YOLOXDepthwise(d, w)
        print(f"YOLOXDepthwise (type: {model_type}) (depth: {d}, width: {w}) initialized.")
    elif "p6-v2" in model_type:
        model = YOLOXP6v2(d, w, act="silu")  # SILU!
        print(f"YOLOXP6v2 (type: {model_type}) (depth: {d}, width: {w}) initialized.")
    elif "p6" in model_type:
        model = YOLOXP6(d, w)
        print(f"YOLOXP6 (type: {model_type}) (depth: {d}, width: {w}) initialized.")
    else:
        model = YOLOX(d, w)
        print(f"YOLOX (type: {model_type}) (depth: {d}, width: {w}) initialized.")

    model.eval().cuda()
    if cfg["ckpt"] is not None and not cfg["sparse"]:
        model.load_state_dict(torch.load(cfg["ckpt"], map_location="cuda"), strict=True)
    elif cfg["sparse"]:
        ckpt = torch.load(cfg["ckpt"], map_location="cuda")['model']
        for key, param in ckpt.items():
            model.state_dict()[key].copy_(param.to_dense().data)

    if cfg["half"]:
        model = model.half()
    return model


def replace_swish(module, prefix=""):
    model = module
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            new_child = nn.Hardswish(inplace=True)
            setattr(module, name, new_child)
            # print(f"... {prefix + '.' + name} converted.")
        elif isinstance(child, nn.Sequential):
            for seq_child in child:
                replace_swish(seq_child, prefix=prefix + '.' + name)
        else:
            replace_swish(child, prefix=prefix + '.' + name)

    # double check
    # for m_name, m in model.named_modules():
    #     if isinstance(m, nn.SiLU):
    #         print(f"... ! {m_name} is not converted!")

    return model


def build_data(cfg):
    dataset = YOLOXImageFolderDataset(cfg["data_dir"], cfg["img_size"])
    dataloader = YOLOXDataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=cfg["dataloader"]["num_workers"],
        prefetch_factor=cfg["dataloader"]["prefetch_factor"],
        img_size=cfg["img_size"]
    )
    print(f"Pytorch Dataloader initialized.")
    return dataset, dataloader

def run(cfg, output_path: str, profile: bool, challenge: bool):
    tracker = TimeTracker(profile=profile)
    start_time = time_synchronized()

    pprint.pprint(cfg)

    # ================================================================================  #
    # Setup
    print("============================================================\nSetup...")
    # ================================================================================  #
    model = build_yolox(cfg)

    if cfg["hard_swish"]:
        model = replace_swish(model)
        
    if cfg["sparse"]:
        params = count_sparse_params(torch.load(cfg["ckpt"], map_location="cuda")['model'])
    else:
        params = count_params(model)

    _, dataloader = build_data(cfg)

    torch.set_grad_enabled(False)
    is_half = cfg["half"]
    batch_size = cfg["dataloader"]["batch_size"]
    img_size = cfg["img_size"]
    is_dummy = (cfg["ckpt"] is None)

    data_load_duration = 0.0
    data_to_gpu_duration = 0.0
    forward_duration = 0.0
    postprocess_duration = 0.0
    nms_duration = 0.0
    json_convert_duration = 0.0

    current_img_size = (0, 0)
    grids = scales = None

    # Dummy Input Experiment
    img = torch.empty((batch_size, 3, img_size, img_size),
            dtype=torch.float16 if is_half else torch.float32, device="cuda")

    _, _, _ = model(img)

    setup_duration = tracker.update()

    # ================================================================================  #
    # Run
    print("============================================================\nRun...")
    # ================================================================================  #
    results = []
    if challenge:
        results.append({"framework": "pytorch"})
        results.append({"parameters": params})

    for batch_i, (img, img_info) in enumerate(dataloader):
        data_load_duration += tracker.update()
        if (batch_i + 1) % 20 == 0:
            print(f"... {batch_i + 1} / {len(dataloader)}")

        _, _, batch_h, batch_w, = img.shape
        # --------------------------------------------------------------------------------  #
        img = img.half() if is_half else img.float()
        img = img.to("cuda", non_blocking=True)

        # Image Aug..?
        img.mul_(0.9).add_(11.4)  # equivalent
        data_to_gpu_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        reg_outputs, obj_outputs, cls_outputs = model(img)

        forward_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        if (batch_h, batch_w) != current_img_size:
            # grids, scales = yolox_generate_grid(img_size, strides=model.head.strides,
            #                                     dtype=torch.float16 if is_half else torch.float32)
            grids, scales = yolox_generate_grid((batch_h, batch_w), strides=model.head.strides,
                                                dtype=torch.float16 if is_half else torch.float32)
            grids = grids.cuda()
            scales = scales.cuda()
            current_img_size = (batch_h, batch_w)

        reg_boxes, obj_conf, cls_conf = yolox_postprocess_output_torch_batch(
            reg_outputs, obj_outputs, cls_outputs, grids, scales)

        postprocess_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        if is_dummy:  # only run when valid
            continue

        batch_outputs = yolox_nms_torch_batch(
            reg_boxes, obj_conf, cls_conf,
            nms_threshold=cfg["postprocess"]["nms_threshold"],
            conf_threshold=cfg["postprocess"]["conf_threshold"],
            soft=cfg["postprocess"].get("soft", False),
            multi_class=cfg["postprocess"].get("multi_class", False),
            rmmop=cfg["postprocess"].get("rmmop", None)
        )

        nms_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        batch_results = convert_to_coco_format(batch_outputs, img_info, img_size)
        results.extend(batch_results)

        json_convert_duration += tracker.update()
        # --------------------------------------------------------------------------------  #

    # ================================================================================  #
    # SAVE JSON
    print(f"============================================================\nSave to {output_path}...")
    # ================================================================================  #
    if not is_dummy:  # only save when valid
        with open(output_path, "w") as f:
            json.dump(results, f)

    json_save_duration = tracker.update()

    end_time = time_synchronized()

    if profile:
        print(f"[TIME] Setup: {setup_duration:.3f}")
        print(f"[TIME] Total Data Loading: {data_load_duration:.3f}")
        print(f"[TIME] Total Data to GPU: {data_to_gpu_duration:.3f}")
        print(f"[TIME] Total Forward: {forward_duration:.3f}")
        print(f"[TIME] Total Postprocessing: {postprocess_duration:.3f}")
        print(f"[TIME] Total NMS: {nms_duration:.3f}")
        print(f"[TIME] Total JSON convert: {json_convert_duration:.3f}")
        print(f"[TIME] JSON save: {json_save_duration:.3f}")
    print(f"[TIME] Final Predict Time: {(end_time - start_time) / 1e6:.3f}")
    print(f"[TIME] Final Predict Time Per Image : {(end_time - start_time) / 5e9:.3f}")
    print(f"[PARAMS] Total Parameter Count: : {params}")

    if (not challenge) and (not is_dummy):
        print("============================================================\nStart evaluation...")
        evaluator = COCOEvaluator(cfg["annotation"])
        ap50_95, ap50, summary = evaluator.evaluate(output_path)
        print(f"AP50:95 = {ap50_95:.6f} | AP50 = {ap50:.6f}")
        print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/yolox_m_p6_sparse.json", type=str, help="Inference config JSON path")
    parser.add_argument("--ckpt", default=None, type=str, help="Checkpoint pth path")
    parser.add_argument("--out", default="answersheet_4_04_choijhanyangackr.json", type=str, help="Output result saving JSON path")
    parser.add_argument("--profile", action="store_true", help="Profiling ON (default: OFF)")
    parser.add_argument("--challenge", action="store_true", help="Challenge mode ON (default: OFF)")
    parser.add_argument("--dummy", action="store_true", help="Load dummy ckpt")

    parser.add_argument("--half", action="store_true", help="Use FP16 (Not recommended)")
    parser.add_argument("--dali", action="store_true", help="Use DALI (Not recommended)")
    parser.add_argument("--hard_swish", action="store_true", help="Use HardSwish instead of Swish (Not recommended)")
    parser.add_argument("--img_size", default=None, type=int, help="Override img_size")
    parser.add_argument("--batch_size", default=None, type=int, help="Override batch_size")
    parser.add_argument("--conf_threshold", default=None, type=float, help="Override conf_threshold")
    parser.add_argument("--nms_threshold", default=None, type=float, help="Override nms_threshold")
    parser.add_argument("--rmmop_r1", default=None, type=float, help="RMMOP r1")
    parser.add_argument("--rmmop_r2", default=None, type=float, help="RMMOP r2")

    args = parser.parse_args()

    with open(args.config, "r") as cf:
        config = json.load(cf)
    if args.ckpt is not None:
        config["ckpt"] = args.ckpt

    # CHALLENGE MODE for Submit
    args.challenge = True
    
    config["half"] = args.half
    config["dali"] = args.dali
    config["hard_swish"] = args.hard_swish
    if args.dummy:
        config["ckpt"] = None
    if args.img_size is not None:
        config["img_size"] = args.img_size
    if args.batch_size is not None:
        config["dataloader"]["batch_size"] = args.batch_size
    if args.conf_threshold is not None:
        config["postprocess"]["conf_threshold"] = args.conf_threshold
    if args.nms_threshold is not None:
        config["postprocess"]["nms_threshold"] = args.nms_threshold

    if (args.rmmop_r1 is not None) and (args.rmmop_r2 is not None):
        config["postprocess"]["rmmop"] = (args.rmmop_r1, args.rmmop_r2)
    else:
        config["postprocess"]["rmmop"] = None

    run(config, args.out, args.profile, args.challenge)

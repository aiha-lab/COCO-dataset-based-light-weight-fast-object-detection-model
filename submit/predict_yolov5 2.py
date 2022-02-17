import os
import argparse
import json
import pprint

import torch

from models import Model as YOLOv5
from yolov5_infer.dataset import YOLOv5ImageFolderDataset, YOLOv5DataLoader
from yolov5_infer.postprocess_utils import yolov5_nms_torch_batch, yolov5_postprocess_output_torch_batch
from common.profile import TimeTracker, time_synchronized
from common.utils import convert_to_coco_format
from common.evaluator import COCOEvaluator


def count_params(m) -> int:
    count = 0
    for p in m.parameters():
        count += p.numel()
    print(f"Parameters: {count}")
    return count


def build_yolov5(cfg):
    model_cfg = os.path.join("models", cfg["model"]["yaml"])
    model = YOLOv5(model_cfg)
    if cfg["ckpt"] is not None:
        ckpt = torch.load(cfg["ckpt"], map_location="cuda")
        csd = ckpt["model"].float().state_dict()
        model.load_state_dict(csd, strict=False)
        print(f"YOLOv5 loaded checkpoint {cfg['ckpt']}")

    model.fuse()
    model.eval().cuda()
    if cfg["half"]:
        model = model.half()
    else:
        model = model.float()
    return model


def build_data(cfg):
    dataset = YOLOv5ImageFolderDataset(cfg["data_dir"], cfg["img_size"])
    dataloader = YOLOv5DataLoader(
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
    model = build_yolov5(cfg)

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

    # Run largest dummy input first
    img = torch.empty((batch_size, 3, img_size, img_size),
                      dtype=torch.float16 if is_half else torch.float32, device="cuda")
    _ = model(img)

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
        img.div_(255.0)

        data_to_gpu_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        outputs = model(img)

        forward_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        # YOLOv5 handles grid inside the Detect layer.

        reg_boxes, obj_conf, cls_conf = yolov5_postprocess_output_torch_batch(outputs)

        postprocess_duration += tracker.update()
        # --------------------------------------------------------------------------------  #
        if is_dummy:  # only run when valid
            continue

        batch_outputs = yolov5_nms_torch_batch(
            reg_boxes, obj_conf, cls_conf,
            nms_threshold=cfg["postprocess"]["nms_threshold"],
            conf_threshold=cfg["postprocess"]["conf_threshold"],
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
    print(f"[PARAMS] Total Parameter Count: : {params}")

    if (not challenge) and (not is_dummy):
        print("============================================================\nStart evaluation...")
        evaluator = COCOEvaluator(cfg["annotation"])
        ap50_95, ap50, summary = evaluator.evaluate(output_path)
        print(f"AP50:95 = {ap50_95:.6f} | AP50 = {ap50:.6f}")
        print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Inference config JSON path")
    parser.add_argument("--ckpt", default=None, type=str, help="Checkpoint pth path")
    parser.add_argument("--out", default="predict.json", type=str, help="Output result saving JSON path")
    parser.add_argument("--profile", action="store_true", help="Profiling ON (default: OFF)")
    parser.add_argument("--challenge", action="store_true", help="Challenge mode ON (default: OFF)")
    parser.add_argument("--dummy", action="store_true", help="Load dummy ckpt")

    parser.add_argument("--half", action="store_true", help="Use FP16 (Not recommended)")
    parser.add_argument("--dali", action="store_true", help="Use DALI (Not recommended)")
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

    config["half"] = args.half
    config["dali"] = args.dali
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

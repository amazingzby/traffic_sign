# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import json
import os
from multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np

import megengine as mge
import megengine.distributed as dist
from megengine.data import DataLoader

from tools.data_mapper import data_mapper
from tools.utils import DetEvaluator, InferenceSampler, import_from_file
from tools.nms import py_cpu_nms

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for testing",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    return parser


def main():
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()

    current_network = import_from_file(args.file)
    cfg = current_network.Cfg()

    if args.weight_file:
        args.start_epoch = args.end_epoch = -1
    else:
        if args.start_epoch == -1:
            args.start_epoch = cfg.max_epoch - 1
        if args.end_epoch == -1:
            args.end_epoch = args.start_epoch
        assert 0 <= args.start_epoch <= args.end_epoch < cfg.max_epoch

    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        if args.weight_file:
            weight_file = args.weight_file
        else:
            weight_file = "logs/{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0] + f'_gpus{args.devices}', epoch_num
            )

        result_list = []
        if args.devices > 1:
            result_queue = Queue(2000)

            master_ip = "localhost"
            server = dist.Server()
            port = server.py_server_port
            procs = []
            for i in range(args.devices):
                proc = Process(
                    target=worker,
                    args=(
                        current_network,
                        weight_file,
                        args.dataset_dir,
                        result_queue,
                        master_ip,
                        port,
                        args.devices,
                        i,
                    ),
                )
                proc.start()
                procs.append(proc)

            # num_imgs = dict(coco=5000, objects365=30000, traffic5=584)  # test set
            num_imgs = dict(coco=5000, objects365=30000, traffic5=299)  # val set

            for _ in tqdm(range(num_imgs[cfg.test_dataset["name"]])):
                result_list.append(result_queue.get())

            for p in procs:
                p.join()
        else:
            worker(current_network, weight_file, args.dataset_dir, result_list)

        all_results = DetEvaluator.format(result_list, cfg)
        json_path = "logs/{}/epoch_{}.json".format(
            os.path.basename(args.file).split(".")[0] + f'_gpus{args.devices}', epoch_num
        )
        all_results = json.dumps(all_results)

        with open(json_path, "w") as fo:
            fo.write(all_results)
        logger.info("Save to %s finished, start evaluation!", json_path)

        eval_gt = COCO(
            os.path.join(
                args.dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]
            )
        )
        eval_dt = eval_gt.loadRes(json_path)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType="bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        metrics = [
            "AP",
            "AP@0.5",
            "AP@0.75",
            "APs",
            "APm",
            "APl",
            "AR@1",
            "AR@10",
            "AR@100",
            "ARs",
            "ARm",
            "ARl",
        ]
        logger.info("mmAP".center(32, "-"))
        for i, m in enumerate(metrics):
            logger.info("|\t%s\t|\t%.03f\t|", m, cocoEval.stats[i])
        logger.info("-" * 32)


def worker(
    current_network, weight_file, dataset_dir, result_list,
    master_ip=None, port=None, world_size=1, rank=0
):
    if world_size > 1:
        dist.init_process_group(
            master_ip=master_ip,
            port=port,
            world_size=world_size,
            rank=rank,
            device=rank,
        )

    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)

    test_loader = build_dataloader(dataset_dir, model.cfg)
    if dist.get_world_size() == 1:
        test_loader = tqdm(test_loader)

    #test_short_sizes = [640,1280,1920]
    test_short_sizes = [640, 1280, 2048]
    #test_short_sizes = [1280]
    for data in test_loader:
        input_w,input_h = data[1][0][0],data[1][1][0]
        box_scale = max(1280.0 / input_w, 1280.0 / input_h)
        pred_res = []
        for test_image_short_size in test_short_sizes:
            pred = []
            image, im_info = DetEvaluator.process_inputs(
                data[0][0],test_image_short_size,int(test_image_short_size*2.0))
            pred_ = evaluator.predict(
                image=mge.tensor(image),
                im_info=mge.tensor(im_info)
            )
            for elem in pred_:
                area = (elem[2] - elem[0]) * (elem[3] - elem[1])
                if   test_image_short_size == test_short_sizes[0]:
                    if area*box_scale > 64*64 and area*box_scale < 400*400:
                        pred.append(elem)
                elif test_image_short_size == test_short_sizes[1]:
                    if area*box_scale > 12*12 and area*box_scale < 300*300:
                        pred.append(elem)
                elif test_image_short_size == test_short_sizes[2]:
                    if area*box_scale < 64*64 and area*box_scale > 6*6:
                        pred.append(elem)
            if len(pred) > 0:
                pred = np.array(pred)
                pred_res.append(pred)
        if len(pred_res) > 0:
            pred_res = np.concatenate(pred_res,axis=0)
            temp_preds = []
            for cls_idx in range(model.cfg.num_classes):
                cls_ids = np.logical_and(pred_res[:,5] < cls_idx + 0.2,pred_res[:,5] > cls_idx - 0.2)
                cls_preds = pred_res[cls_ids]
                if len(cls_preds) > 0:
                    keep = py_cpu_nms(cls_preds,model.cfg.test_nms)
                    cls_preds = cls_preds[keep]
                    temp_preds.append(cls_preds)
            pred_res = np.concatenate(temp_preds,axis=0)
            pred_res = np.array(sorted(pred_res, reverse=True, key=lambda i: i[4]))
        result = {
            "det_res": pred_res,
            "image_id": int(data[1][3][0]),
        }
        if dist.get_world_size() > 1:
            result_list.put_nowait(result)
        else:
            result_list.append(result)


def build_dataloader(dataset_dir, cfg):
    val_dataset = data_mapper[cfg.test_dataset["name"]](
        os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]),
        os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]),
        order=["image", "info"],
    )
    val_sampler = InferenceSampler(val_dataset, 1)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


if __name__ == "__main__":
    main()

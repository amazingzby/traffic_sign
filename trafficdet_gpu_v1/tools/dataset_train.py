# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import json
from collections import defaultdict

import cv2
import numpy as np
import random

from megengine.data.dataset.vision.meta_vision import VisionDataset
"""
[array([[ 974.96875,  328.08618,  991.13696,  361.89258],
       [1248.6011 ,  682.3428 , 1541.6519 ,  942.3281 ],
       [1018.9419 ,  553.6376 , 1056.3047 ,  572.8163 ],
       [ 858.5503 ,  558.1526 ,  902.1953 ,  578.8076 ],
       [ 978.2014 ,  531.24646,  994.8916 ,  540.2538 ],
       [ 888.3926 ,  536.015  ,  917.5342 ,  543.43286],
       [ 491.84204,  732.5764 ,  868.35596, 1005.3594 ],
       [3090.4568 ,  206.23456, 3126.8765 ,  271.66666],
       [1207.4662 , 1676.01   , 1263.5886 , 1727.0303 ],
       [2652.5796 , 1461.359  , 2712.7107 , 1520.0326 ]], dtype=float32), array([1, 2, 2, 3, 2, 3, 3, 1, 5, 5], dtype=int32), [2400, 3200, '3279125,2067e6000cce1694a.jpg', 2574]]

"""
"""
[array([[1032.9688 ,  523.0862 , 1049.137  ,  556.8926 ],
       [1306.6011 ,  877.3428 , 1599.6519 , 1137.3281 ],
       [1076.9419 ,  748.6376 , 1114.3047 ,  767.8163 ],
       [ 916.5503 ,  753.1526 ,  960.1953 ,  773.8076 ],
       [1036.2014 ,  726.24646, 1052.8916 ,  735.2538 ],
       [ 946.3926 ,  731.015  ,  975.5342 ,  738.43286],
       [ 549.84204,  927.5764 ,  926.35596, 1200.3594 ]], dtype=float32), array([1, 2, 2, 3, 2, 3, 3], dtype=int32), [1200, 1600, '3279125,2067e6000cce1694a.jpg', 2574]]
"""
def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    return True


class Traffic5(VisionDataset):
    r"""
    Traffic Detection Challenge Dataset.
    """

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        "info",
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, *, order=None,
        enable_mosaic=True,enable_mixup=True
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "polygons" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):
        #数据传入train接口也是数组，key为data,gt_boxes,'im_info'
        min_box,max_box = 8,400
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]
        #mosaic_prob = np.random.uniform(0,1)
        if self.enable_mosaic:
            info = self.imgs[img_id]
            input_h,input_w = info["height"], info["width"]
            mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)
            #yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            #xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
            mosaic_scale = random.uniform(0.5,2.0)
            yc = int(input_h*mosaic_scale)
            xc = int(input_w*mosaic_scale)
            #yc = int(random.uniform(0.5 * input_h, 2.0 * input_h))
            #xc = int(random.uniform(0.5 * input_w, 2.0 * input_w))
            indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]

            mosaic_bboxes = []
            mosaic_categories = []
            for i_mosaic, idx in enumerate(indices):
                mosaic_id = self.ids[idx]
                mosaic_anno = self.img_to_anns[mosaic_id]

                mosaic_category = [obj["category_id"] for obj in mosaic_anno]
                mosaic_category = [
                    self.json_category_id_to_contiguous_id[c] for c in mosaic_category
                ]
                mosaic_category = np.array(mosaic_category, dtype=np.int32)

                file_name = self.imgs[mosaic_id]['file_name']
                path = os.path.join(self.root, file_name)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                mosaic_bbox = [obj["bbox"] for obj in mosaic_anno]
                mosaic_bbox = np.array(mosaic_bbox, dtype=np.float32).reshape(-1, 4)
                mosaic_bbox[:,2:] += mosaic_bbox[:,:2]

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                (h, w, c) = img.shape[:3]
                #if i_mosaic == 0:
                #    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1
                if len(mosaic_bbox) > 0:
                    mosaic_bbox[:,0] = scale * mosaic_bbox[:,0] + padw
                    mosaic_bbox[:,1] = scale * mosaic_bbox[:,1] + padh
                    mosaic_bbox[:,2] = scale * mosaic_bbox[:,2] + padw
                    mosaic_bbox[:,3] = scale * mosaic_bbox[:,3] + padh
                if i_mosaic == 0:
                    mosaic_bboxes = mosaic_bbox
                    mosaic_categories = mosaic_category
                else:
                    mosaic_bboxes = np.concatenate([mosaic_bboxes,mosaic_bbox])
                    mosaic_categories = np.concatenate([mosaic_categories,mosaic_category])
            if len(mosaic_bboxes):
                np.clip(mosaic_bboxes[:, 0], 0, 2 * input_w, out=mosaic_bboxes[:, 0])
                np.clip(mosaic_bboxes[:, 1], 0, 2 * input_h, out=mosaic_bboxes[:, 1])
                np.clip(mosaic_bboxes[:, 2], 0, 2 * input_w, out=mosaic_bboxes[:, 2])
                np.clip(mosaic_bboxes[:, 3], 0, 2 * input_h, out=mosaic_bboxes[:, 3])

            #random crop
            #mosaic_area = (mosaic_bboxes[:,2] - mosaic_bboxes[:,0])*(mosaic_bboxes[:,3] - mosaic_bboxes[:,1])
            #mosaic_max_area = mosaic_area.max()
            #for temp_box in mosaic_bboxes:
            #    cv2.rectangle(mosaic_img,(int(temp_box[0]),int(temp_box[1])),(int(temp_box[2]),int(temp_box[3])),(0,0,255))
            #cv2.imwrite(str(index)+"_mosaic.jpg",mosaic_img)
            crop_img = mosaic_img.copy()
            crop_bboxes = mosaic_bboxes.copy()
            crop_categories = mosaic_categories.copy()
            dst_input_h = input_h * 2
            dst_input_w = input_w * 2
            for crop_idx in range(20):
                crop_ratio = random.uniform(0.33,1.0)
                crop_x = int(crop_ratio * input_w * 2.0)
                crop_y = int(crop_ratio * input_h * 2.0)
                pad_x = random.randint(0,input_w*2-crop_x)
                pad_y = random.randint(0,input_h*2-crop_y)
                crop_bboxes_ = mosaic_bboxes.copy()
                crop_bboxes_[:,0] -= pad_x
                crop_bboxes_[:,1] -= pad_y
                crop_bboxes_[:,2] -= pad_x
                crop_bboxes_[:,3] -= pad_y
                np.clip(crop_bboxes_[:,0],0,crop_x,out=crop_bboxes_[:,0])
                np.clip(crop_bboxes_[:,1],0,crop_y,out=crop_bboxes_[:,1])
                np.clip(crop_bboxes_[:,2],0,crop_x,out=crop_bboxes_[:,2])
                np.clip(crop_bboxes_[:,3],0,crop_y,out=crop_bboxes_[:,3])
                crop_area = (crop_bboxes_[:,2]-crop_bboxes_[:,0])*(crop_bboxes_[:,3]-crop_bboxes_[:,1])
                crop_max_area = crop_area.max()
                if(crop_max_area>8):
                    crop_img = mosaic_img[pad_y:pad_y + crop_y, pad_x:pad_x + crop_x]
                    crop_bboxes = crop_bboxes_
                    dst_input_h = crop_y
                    dst_input_w = crop_x
                    break
            info = [dst_input_h, dst_input_w, info["file_name"], img_id]
            dst_img = crop_img
            dst_bboxes = []
            dst_categories = []
            box_scale = max(1280.0/dst_input_w,1280.0/dst_input_h)
            for temp_idx in range(len(crop_bboxes)):
                temp_w,temp_h = crop_bboxes[temp_idx][2] - crop_bboxes[temp_idx][0],crop_bboxes[temp_idx][3] - crop_bboxes[temp_idx][1]
                if(temp_w*box_scale > min_box and temp_h*box_scale > min_box and
                   temp_w*box_scale < max_box and temp_h*box_scale < max_box):
                    dst_bboxes.append(crop_bboxes[temp_idx])
                    dst_categories.append(crop_categories[temp_idx])
            if len(dst_bboxes) > 0:
                dst_bboxes = np.array(dst_bboxes)
                dst_categories = np.array(dst_categories)
                target = [dst_img,dst_bboxes,dst_categories,info]
                #print(mosaic_bboxes)
                return tuple(target)

        target = []

        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]

        file_name = self.imgs[img_id]["file_name"]
        path = os.path.join(self.root, file_name)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        info = self.imgs[img_id]
        info = [info["height"], info["width"], info["file_name"], img_id]
        box_scale = max(1280.0 / info[0], 1280.0 / info[1])

        boxes = [obj["bbox"] for obj in anno]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        boxes_category = [obj["category_id"] for obj in anno]
        boxes_category = [
            self.json_category_id_to_contiguous_id[c] for c in boxes_category
        ]
        boxes_category = np.array(boxes_category, dtype=np.int32)

        boxes_ ,boxes_category_ = [],[]
        for id_box in range(len(boxes)):
            temp_w = boxes[id_box,2] - boxes[id_box,0]
            temp_h = boxes[id_box,3] - boxes[id_box,1]
            if(temp_w*box_scale > min_box and temp_h*box_scale > min_box and
               temp_w*box_scale < max_box and temp_h*box_scale < max_box):
                boxes_.append(boxes[id_box])
                boxes_category_.append(boxes_category[id_box])
        if len(boxes_) > 0:
            boxes_ = np.array(boxes_,np.float32)
            boxes_category_ = np.array(boxes_category_,dtype=np.int32)
            target.append(image)
            target.append(boxes_)
            target.append(boxes_category_)
            target.append(info)
        while(len(target) <= 0):
            new_index = random.randint(0, len(self.ids) -1)
            img_id = self.ids[new_index]
            anno = self.img_to_anns[img_id]

            file_name = self.imgs[img_id]["file_name"]
            path = os.path.join(self.root, file_name)
            image = cv2.imread(path, cv2.IMREAD_COLOR)

            info = self.imgs[img_id]
            info = [info["height"], info["width"], info["file_name"], img_id]
            box_scale = max(1280.0 / info[0], 1280.0 / info[1])

            boxes = [obj["bbox"] for obj in anno]
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]

            boxes_category = [obj["category_id"] for obj in anno]
            boxes_category = [
                self.json_category_id_to_contiguous_id[c] for c in boxes_category
            ]
            boxes_category = np.array(boxes_category, dtype=np.int32)

            boxes_, boxes_category_ = [], []
            for id_box in range(len(boxes)):
                temp_w = boxes[id_box, 2] - boxes[id_box, 0]
                temp_h = boxes[id_box, 3] - boxes[id_box, 1]
                if (temp_w * box_scale > min_box and temp_h * box_scale > min_box and
                    temp_w * box_scale < max_box and temp_h * box_scale < max_box):
                    boxes_.append(boxes[id_box])
                    boxes_category_.append(boxes_category[id_box])
            if len(boxes_) > 0:
                boxes_ = np.array(boxes_, np.float32)
                boxes_category_ = np.array(boxes_category_, dtype=np.int32)
                target.append(image)
                target.append(boxes_)
                target.append(boxes_category_)
                target.append(info)
                break

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "red_tl",
        "arr_s",
        "arr_l",
        "no_driving_mark_allsort",
        "no_parking_mark",
    )

    classes_originID = {
        "red_tl": 0,
        "arr_s": 1,
        "arr_l": 2,
        "no_driving_mark_allsort": 3,
        "no_parking_mark": 4,
    }

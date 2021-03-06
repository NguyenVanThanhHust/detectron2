import logging
import numpy as np
import os
import os.path as osp
import json
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
from fvcore.common.file_io import PathManager
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
import pickle


def get_hoi_box(human_bbox, object_bbox):
    h_xmin, h_ymin, h_xmax, h_ymax = human_bbox
    o_xmin, o_ymin, o_xmax, o_ymax = object_bbox
    xmin, ymin = min(h_xmin, o_xmin), min(h_ymin, o_ymin)
    xmax, ymax = max(h_xmax, o_xmax), max(h_ymax, o_ymax)
    hoi_bbox = [xmin, ymin, xmax, ymax]
    return hoi_bbox

def single_iou(bb_gt, pred_bb):
    # compute overlaps
    # intersection
    ixmin = np.maximum(bb_gt[:, 0], pred_bb[0])
    iymin = np.maximum(bb_gt[:, 1], pred_bb[1])
    ixmax = np.minimum(bb_gt[:, 2], pred_bb[2])
    iymax = np.minimum(bb_gt[:, 3], pred_bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (pred_bb[2] - pred_bb[0] + 1.0) * (pred_bb[3] - pred_bb[1] + 1.0)
        + (bb_gt[:, 2] - bb_gt[:, 0] + 1.0) * (bb_gt[:, 3] - bb_gt[:, 1] + 1.0)
        - inters
    )

    overlaps = inters / uni
    # sys.exit()
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return jmax, ovmax


def get_hoi_iou(object_gt, human_gt, hoi_bb):
    pred_human_box = hoi_bb[:4]
    pred_object_box = hoi_bb[4:]
    human_jmax, human_ovmax = single_iou(human_gt, pred_human_box)
    object_jmax, object_ovmax = single_iou(object_gt, pred_object_box)
    if human_ovmax < object_ovmax:
        return human_jmax, human_ovmax, True
    return object_jmax, object_ovmax, False

def get_ground_truth(json_folder):
    with open(osp.join(json_folder, "anno_list.json")) as jfp:
        full_data = json.load(jfp)

    object_names = []
    with open(osp.join(json_folder, "object_list.json")) as jfp:
        list_object = json.load(jfp)
        for each_object_info in list_object:
            object_names.append(each_object_info["name"])

    verb_names = []
    with open(osp.join(json_folder, "verb_list.json")) as jfp:
        list_verb = json.load(jfp)
        for each_verb_info in list_verb:
            verb_names.append(each_verb_info["name"])

    class_names = []
    with open(osp.join(json_folder, "hoi_list.json")) as jfp:
        hoi_list = json.load(jfp)
        for i in range(600):
            class_names.append(str(i+1))

    gts_dict = dict()    
    for each_instance in full_data:
        global_id = each_instance["global_id"]
        if "train" in global_id:
            continue
        instances = []
        hois = each_instance["hois"]
        for hoi in hois:
            human_bboxes = hoi["human_bboxes"]
            hoi_id = hoi["id"]
            invis = hoi["invis"] # invisible
            if invis:
                continue
            object_bboxes = hoi["object_bboxes"]
            object_name = hoi_list[int(hoi_id) - 1]["object"]
            human_bbox = human_bboxes[0]
            object_bbox = object_bboxes[0]
            # hoi_box = get_hoi_box(human_bbox, object_bbox)
            verb = hoi_list[int(hoi_id) - 1]["verb"]
            verb_id = verb_names.index(verb) + 1
            instances.append(
                {"object_id": object_names.index(object_name) + 1,
                "object_bbox":object_bbox,
                "human_bbox":human_bbox,
                "verb": verb,
                "verb_id": verb_id, 
                "hoi_id":hoi_id, 
                    }
                )
        gts_dict[global_id] = instances
    return gts_dict
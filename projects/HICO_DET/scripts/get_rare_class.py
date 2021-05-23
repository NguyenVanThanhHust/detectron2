import numpy as np
import os
import os.path as osp
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
import cv2 
from typing import List, Tuple, Union
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["load_hico_data",]

def load_hico_data(img_folder:str, json_folder:str, split:str):
    """
    Load data
    """
    gts_dict = dict()
    for i in range(600):
        gts_dict[i]=0

    assert split in ["train", "test", "overfit"], "split must be train/test/overfit"
    dicts = []
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

    for each_instance in full_data:
        global_id = each_instance["global_id"]
        if "test" in global_id:
            continue
        r = {
            "file_name": osp.join(img_folder, split+"2015",  global_id + ".jpg"),
            "image_id": int(global_id[-8:]),
        }

        image_size = each_instance["image_size"]
        
        height, width, channels = image_size
        r["width"] = width
        r["height"] = height
        
        instances = []
        hois = each_instance["hois"]
        for hoi in hois:
            human_bboxes = hoi["human_bboxes"]
            hoi_id = hoi["id"]
            invis = hoi["invis"] # invisible

            if invis:
                continue
            gts_dict[int(hoi_id) - 1] += len(hoi["connections"])

            object_bboxes = hoi["object_bboxes"]
            object_name = hoi_list[int(hoi_id) - 1]["object"]
            human_bbox = human_bboxes[0]
            object_bbox = object_bboxes[0]
            # hoi_box = get_hoi_box(human_bbox, object_bbox)
            verb = hoi_list[int(hoi_id) - 1]["verb"]
            verb_id = verb_names.index(verb) + 1
            instances.append(
                {
                    "object_bbox":object_bbox,
                    "human_bbox":human_bbox,
                    "verb": verb,
                    "object_id": object_names.index(object_name) + 1,
                    "verb_id": verb_id, 
                    "hoi_id":hoi_id, 
                    }
                )
        
        r["annotations"] = instances
        dicts.append(r)
    return dicts, gts_dict

class_names = []

img_folder = "../../../data/HICO_DET/images/"
json_folder = "../../../data/HICO_DET/hico_det_json/"
split = "train"

dicts, gts_dict = load_hico_data(img_folder=img_folder, json_folder=json_folder, split=split)

rare_hoi = []
for i in range(600):
    if gts_dict[i] < 10:
        rare_hoi.append(str(i+1).zfill(3))
print(rare_hoi)
print(len(rare_hoi))

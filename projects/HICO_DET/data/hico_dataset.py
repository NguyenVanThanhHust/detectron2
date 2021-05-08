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
try:
    from utils import get_hoi_box
except:
    from .utils import get_hoi_box

__all__ = ["load_hico_data",]

def load_hico_data(img_folder:str, json_folder:str, split:str):
    """
    Load data
    """
    assert split in ["train", "test"], "split must be train or test"
    list_dict = []
    with open(osp.join(json_folder, "anno_list.json")) as jfp:
        full_data = json.load(jfp)
    with open(osp.join(json_folder, "hoi_list.json")) as jfp:
        hoi_list = json.load(jfp)

    for each_instance in full_data:
        global_id = each_instance["global_id"]
        if split not in global_id:
            continue
        if global_id == "HICO_train2015_00011533":
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
            action = hoi["id"]
            invis = hoi["invis"] # invisible
            if invis:
                continue
            object_bboxes = hoi["object_bboxes"]
            object_name = hoi_list[int(action) - 1]["object"]
            human_bbox = human_bboxes[0]
            object_bbox = object_bboxes[0]
            hoi_box = get_hoi_box(human_bbox, object_bbox)
            # insert object
            instances.append(
                {"category_id": class_names.index(object_name), 
                "object_bbox":object_bbox,
                "human_bbox":human_bbox,
                "bbox_mode":BoxMode.XYXY_ABS
                    }
                )
        
        r["annotations"] = instances
        list_dict.append(r)
    return list_dict

class_names = []

img_folder = "../data/HICO_DET/images/"
json_folder = "../data/HICO_DET/hico_det_json/"
with open(osp.join(json_folder, "object_list.json")) as jfp:
    object_list = json.load(jfp)
    for each_object in object_list:
        object_name = each_object["name"]
        class_names.append(object_name)

assert len(class_names)==80, "number class is wrong " + str(len(class_names))

splits = ["train", "test"]
for split in splits:
    DatasetCatalog.register("HICO_DET_" + split, lambda : load_hico_data(img_folder=img_folder, json_folder=json_folder, split=split))
    MetadataCatalog.get("HICO_DET_" + split).set(thing_classes=class_names)

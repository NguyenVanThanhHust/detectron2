# This is to visualize some example

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
from utils import get_hoi_box
import random
from detectron2.utils.visualizer import Visualizer


__all__ = ["load_hico_data", ]

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
        r = {
            "file_name": osp.join(img_folder, split+"2015",  global_id + ".jpg"),
            "image_id": int(global_id[-8:]),
        }
        instances = []
        image_size = each_instance["image_size"]
        r["width"] = image_size[0]
        r["height"] = image_size[1]

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
            if object_name == "person":
                instances.append(
                    {"category_id": class_names.index(object_name), 
                    "bbox":object_bbox,
                    "bbox_mode":BoxMode.XYXY_ABS
                        }
                    )
            # # insert hoi
            # instances.append(
            #     {"category_id": int(action) + 80,
            #     "bbox":hoi_box,
            #     "bbox_mode":BoxMode.XYXY_ABS
            #         }
            #     )
            # insert human
            human_dict = {"category_id": class_names.index("person"), 
                "bbox":human_bbox,
                "bbox_mode":BoxMode.XYXY_ABS
                    }
            if human_dict not in instances:
                instances.append(
                    human_dict
                    )
        r["annotations"] = instances
        list_dict.append(r)
    return list_dict

class_names = []
img_folder = "../../../data/HICO_DET/images/"
json_folder = "../../../data/HICO_DET/hico_det_json/"
split="test"

with open(osp.join(json_folder, "object_list.json")) as jfp:
    object_list = json.load(jfp)
    for each_object in object_list:
        object_name = each_object["name"]
        class_names.append(object_name)

print("Loading ...")    
list_data_dict = load_hico_data(img_folder=img_folder, json_folder=json_folder, split=split)    
print("Loaded")
    

splits = ["train", "test"]
for split in splits:
    DatasetCatalog.register("HICO_DET_" + split, lambda : load_hico_data(img_folder=img_folder, json_folder=json_folder, split=split))
    MetadataCatalog.get("HICO_DET_" + split).set(thing_classes=class_names)
 
        
metadata = MetadataCatalog.get("HICO_DET_train")

debug_list = [ "HICO_test2015_00009762"
                    ," HICO_test2015_00009763"
                    , "HICO_test2015_00009764"
                    , "HICO_test2015_00009765"
                    , "HICO_test2015_00009766"
                    , "HICO_test2015_00009767"]
                    
dataset_dicts = DatasetCatalog.get('HICO_DET_test')

for d in dataset_dicts:
    image_id = d["image_id"]
    global_id = "HICO_test2015_" + str(image_id).zfill(8)
    if global_id not in debug_list:
        continue
    img = cv2.imread(d["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    v = v.draw_dataset_dict(d)
    img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    output_path = "outputs/" + global_id + ".jpg"
    print(output_path)
    cv2.imwrite(output_path, img)
    
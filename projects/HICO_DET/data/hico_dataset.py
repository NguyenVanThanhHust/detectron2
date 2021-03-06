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

list_over_fit = [
    "HICO_train2015_00038129",
    "HICO_train2015_00038131",
    "HICO_train2015_00038133",
    "HICO_train2015_00038135",
]


rare_hoi = ['009', '023', '028', '045', '051', '056', '063', '064', '067', '071', '077', '078', '081', '084', '085', '091', '100', '101', '105', '108', '113', '128', '136', '137', '150', '159', '166', '167', '169', '173', '180', '182', '185', '189', '190', '193', '196', '199', '206', '207', '215', '217', '223', '228', '230', '239', '240', '255', '256', '258', '261', '262', '263', '275', '280', '281', '282', '287', '290', '293', '304', '312', '316', '318', '326', '329', '334', '335', '346', '351', '352', '355', '359', '365', '380', '382', '390', '391', '392', '396', '398', '399', '400', '402', '403', '404', '405', '406', '408', '411', '417', '419', '427', '428', '430', '432', '437', '440', '441', '450', '452', '464', '470', '475', '483', '486', '499', '500', '505', '510', '515', '518', '521', '523', '527', '532', '536', '540', '547', '548', '549', '550', '551', '552', '553', '556', '557', '561', '579', '581', '582', '587', '593', '594', '596', '597', '598', '600']


def load_hico_data(img_folder:str, json_folder:str, split:str):
    """
    Load data
    """
    assert split in ["train", "test", "overfit"], "split must be train/test/overfit"
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

    list_dict = []

    for each_instance in full_data:
        global_id = each_instance["global_id"]
        if split == "overfit":
            if global_id not in list_over_fit:
                continue
        if split != "overfit" and split not in global_id:
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
            is_rare = False
            if hoi_id in rare_hoi:
                is_rare = True
            instances.append(
                {
                    "object_bbox":object_bbox,
                    "human_bbox":human_bbox,
                    "verb": verb,
                    "object_id": object_names.index(object_name) + 1,
                    "verb_id": verb_id, 
                    "hoi_id":hoi_id, 
                    "is_rare":is_rare,
                    }
                )
        
        r["annotations"] = instances
        list_dict.append(r)
    print("number of instance: ", len(list_dict))
    if split != "overfit":
        print("Example of instace: ", list_dict[100])
    else:
        print(list_dict)    
    return list_dict

class_names = []

img_folder = "../../../data/HICO_DET/images/"
json_folder = "../../../data/HICO_DET/hico_det_json/"
with open(osp.join(json_folder, "object_list.json")) as jfp:
    object_list = json.load(jfp)
    for each_object in object_list:
        object_name = each_object["name"]
        class_names.append(object_name)

assert len(class_names)==80, "number class is wrong " + str(len(class_names))

splits = ["train", "test", "overfit", ]
for split in splits:
    DatasetCatalog.register("HICO_DET_" + split, lambda : load_hico_data(img_folder=img_folder, json_folder=json_folder, split=split))
    MetadataCatalog.get("HICO_DET_" + split).set(thing_classes=class_names)

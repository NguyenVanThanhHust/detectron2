# this is to convert all size in json file,
# because original size in json might be wrong
import os.path as osp
import json
import cv2

img_folder = "../../data/HICO_DET/images/"
json_folder = "../../data/HICO_DET/hico_det_json/"

with open(osp.join(json_folder, "anno_list.json")) as jfp:
    full_data = json.load(jfp)

for each_instance in full_data:
    global_id = each_instance["global_id"]
    img_path = osp.join(img_folder, "train2015",  global_id + ".jpg")
    if not osp.isfile(img_path):
        img_path = osp.join(img_folder, "test2015",  global_id + ".jpg")
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    image_size = each_instance["image_size"]
    each_instance["image_size"] = height, width, channels

with open(osp.join(json_folder, "anno_list.json"), "w") as jfp:
    json.dump(full_data, jfp)
"""
This file is to remove excessive person grounth truth of hoir images

for each image, loop through hoi, if second hoi have iou with first hoi > threshold 
replace with first hoi

"""
import os
import os.path as osp
import json
import shutil
import numpy as np

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def remove_human_box(img_folder:str, json_folder:str ):
    """
    Load data
    """

    # backup data
    back_up_json_path = osp.join(json_folder, "backup_anno_list.json")
    json_path = osp.join(json_folder, "anno_list.json") 
    if not osp.isfile(back_up_json_path):
        shutil.copyfile(json_path, back_up_json_path)
        print("Load from : ", json_path)
        with open(json_path) as jfp:
            full_data = json.load(jfp)
    else:
        with open(back_up_json_path) as jfp:
            print("Load from : ", back_up_json_path)
            full_data = json.load(jfp)

    with open(osp.join(json_folder, "hoi_list.json")) as jfp:
        hoi_list = json.load(jfp)

    try:
        for main_idx, each_instance in enumerate(full_data):
            global_id = each_instance["global_id"]
            hois = each_instance["hois"]

            current_human_dict = dict()
            index = 1

            box_index = np.zeros(len(hois))
            for idx, hoi in enumerate(hois):
                human_bboxes = hoi["human_bboxes"]
                invis = hoi["invis"] # invisible
                if invis:
                    continue
                human_bbox = human_bboxes[0]

                if len(current_human_dict.keys()) == 0:
                    current_human_dict[index] = human_bbox
                    index += 1
                tmp_max_iou = 0.0
                tmp_index = -1
                to_be_added_box = None
                for k, box in current_human_dict.items():
                    iou = bb_intersection_over_union(box, human_bbox)
                    if iou > tmp_max_iou:
                        tmp_max_iou = iou
                        tmp_index = k
                if tmp_max_iou > 0.65:
                    box_index[idx] = tmp_index
                else:
                    box_index[idx] = index
                    current_human_dict[index] = human_bbox
                    index += 1
            # finish new hoi
            for idx, hoi in enumerate(hois):
                invis = hoi["invis"] # invisible
                if invis:
                    continue
                human_box_index = box_index[idx]
                hoi["human_bboxes"] = [current_human_dict[human_box_index]]

    except Exception as e:
        print(global_id)
        print(index)
        print(box_index)
        print(current_human_dict)
        sys.exit()

    # sanity check
    current_full_data = full_data
    with open(back_up_json_path) as jfp:
        ori_full_data = json.load(jfp)
    assert len(ori_full_data) == len(current_full_data), "mismatch number of instance"

    for idx, (new_instc, ori_instc) in enumerate(zip(ori_full_data, current_full_data)):
        new_global_id = new_instc["global_id"]
        ori_global_id = ori_instc["global_id"]
        new_hois = new_instc["hois"]
        ori_hois = ori_instc["hois"]
        
        for _, (new_hoi, ori_hoi) in enumerate(zip(new_hois, ori_hois)):
            invis = new_hoi["invis"] # invisible
            if invis:
                continue
            new_human_bbox = new_hoi["human_bboxes"][0]
            ori_human_bbox = ori_hoi["human_bboxes"][0]
            iou = bb_intersection_over_union(new_human_bbox, ori_human_bbox)
            assert iou >= 0.65, "check image : " + new_global_id + " " + ori_global_id

    with open(json_path, 'w') as jfp:
        json.dump(full_data, jfp)

img_folder = "../../data/HICO_DET/images/"
if os.path.isdir(img_folder):
    img_folder = "../../data/HICO_DET/images/"
    json_folder = "../../data/HICO_DET/hico_det_json/"
    remove_human_box(img_folder=img_folder, json_folder=json_folder)

else:
    img_folder = "../data/HICO_DET/images/"
    json_folder = "../data/HICO_DET/hico_det_json/"
    remove_human_box(img_folder=img_folder, json_folder=json_folder)

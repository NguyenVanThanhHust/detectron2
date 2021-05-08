import os
import os.path as osp
import pickle
import cv2
import argparse
import json

# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
def get_list_number(line):
    list_str = line.split()
    res = []
    for com in list_str:
        res.append(float(com))
    return res

def get_hoi_box(human_bbox, object_bbox):
    h_xmin, h_ymin, h_xmax, h_ymax = human_bbox
    o_xmin, o_ymin, o_xmax, o_ymax = object_bbox
    xmin, ymin = min(h_xmin, o_xmin), min(h_ymin, o_ymin)
    xmax, ymax = max(h_xmax, o_xmax), max(h_ymax, o_ymax)
    hoi_bbox = [xmin, ymin, xmax, ymax]
    return hoi_bbox

def main(args):
    det_res_file = args.det_res_file
    img_folder = args.img_folder
    viz_gts = args.viz_gts
    json_folder = args.json_folder
    result_folder = osp.join(args.result_folder, "images")
    with open(det_res_file, "rb") as f:
        predictions = pickle.load(f)

    list_img = next(os.walk(img_folder))[2]
    det_res_dict = dict()
    os.makedirs(result_folder, exist_ok=True)

    image_ids = []
    for k, v in predictions.items():
        for idx, each_det_res in enumerate(v):
            image_id, score, xmin, ymin, xmax, ymax = get_list_number(each_det_res)
            box = [int(xmin), int(ymin), int(xmax), int(ymax)]
            image_id = int(image_id)
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            sample = dict()
            if image_id not in det_res_dict.keys():
                det_res_dict[image_id] = list()
            sample["score"] = score
            sample["box"] = box
            det_res_dict[image_id].append(sample)
            image_ids.append(image_id)

    if viz_gts:
        gts = dict()
        with open(osp.join(json_folder, "anno_list.json")) as jfp:
            full_data = json.load(jfp)
        with open(osp.join(json_folder, "hoi_list.json")) as jfp:
            hoi_list = json.load(jfp)
        for each_instance in full_data:
            global_id = each_instance["global_id"]
            if "train" in global_id:
                continue
            image_id = int(global_id[-8:])
            if image_id not in image_ids:
                continue

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
                if object_name == "person":
                    instances.append(
                        {"label": object_name,
                        "bbox":object_bbox,
                            }
                        )
                # insert human
                human_dict = {"label": "person",
                    "bbox":human_bbox,
                        }
                if human_dict not in instances:
                    instances.append(
                        human_dict
                        )
            gts[image_id] = instances

    for img_id, samples in det_res_dict.items():
        print("img_id: ", img_id)
        image_path = osp.join(img_folder, "HICO_test2015_" + str(img_id).zfill(8) + ".jpg")
        image = cv2.imread(image_path)
        if viz_gts:
            gt = gts[img_id]
            for each_instance in gt:
                box = each_instance["bbox"]
                xmin, ymin, xmax, ymax = box
                start_point = (xmin, ymin)
                end_point = (xmax, ymax)
                image = cv2.rectangle(image, start_point, end_point, (255, 255, 0), thickness) 
        for sample in samples:
            box = sample["box"]
            score = sample["score"]
            xmin, ymin, xmax, ymax = box
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            image = cv2.rectangle(image, start_point, end_point, color, thickness) 
            image = cv2.putText(image, str(score), start_point, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
 
        res_path = osp.join(result_folder, "HICO_test2015_" + str(img_id).zfill(8) + ".jpg")
        cv2.imwrite(res_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--det_res_file', type=str, help='detection result')
    parser.add_argument('--viz_gts', action='store_true', help='visualize ground truth')
    parser.add_argument('-if', '--img_folder', type=str, help='image folder')
    parser.add_argument('-rf', '--result_folder', type=str, help='result image folder')
    parser.add_argument('-jf', '--json_folder', default="../data/HICO_DET/hico_det_json", type=str, help='result image folder')
    args = parser.parse_args()
    print('Args:')
    for x in vars(args):
        print('\t%s:' % x, getattr(args, x))
    main(args)

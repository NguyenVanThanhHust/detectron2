import os
import os.path as osp
import pickle
import cv2
import argparse

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

def main(args):
    det_res_file = args.det_res_file
    img_folder = args.img_folder
    result_folder = osp.join(args.result_folder, "images")
    with open(det_res_file, "rb") as f:
        predictions = pickle.load(f)

    list_img = next(os.walk(img_folder))[2]
    det_res_dict = dict()
    os.makedirs(result_folder, exist_ok=True)

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

    for img_id, samples in det_res_dict.items():
        image_path = osp.join(img_folder, "HICO_test2015_" + str(img_id).zfill(8) + ".jpg")
        # image = cv2.imread(image_path)
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
    parser.add_argument('-if', '--img_folder', type=str, help='image folder')
    parser.add_argument('-rf', '--result_folder', type=str, help='result image folder')
    args = parser.parse_args()
    print('Args:')
    for x in vars(args):
        print('\t%s:' % x, getattr(args, x))
    main(args)

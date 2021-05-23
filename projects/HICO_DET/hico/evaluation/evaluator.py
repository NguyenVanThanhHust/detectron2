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

from detectron2.evaluation.evaluator import DatasetEvaluator

class HicoEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.
    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, img_folder, json_folder):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._input_img_folder = img_folder
        self._json_folder = json_folder
        
        self._class_names = meta.thing_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._detect_result = dict()

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings
        self._detect_result = dict()

    def _process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            human_boxes = instances.human_boxes.tensor.numpy()
            object_boxes = instances.object_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, each_cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                _tmp_dict = {"cls": each_cls, "human_box":[xmin, ymin, xmax, ymax],
                    "object_box":[]
                    "score":score}

                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[each_cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f} "
                )
                if image_id not in self._detect_result.keys():
                    self._detect_result[image_id] = list()
                
                _tmp_dict = {"cls": each_cls, "box":[xmin, ymin, xmax, ymax],
                    "score":score}
                self._detect_result[image_id].append(_tmp_dict)

    def process(self, inputs, outputs, recompute=True):
        if recompute:
            self._process(inputs, outputs)
            with open("det_res.pkl", "wb") as f:
                pickle.dump(self._predictions, f)
        else:
            if os.path.isfile("det_res.pkl"):
                with open("det_res.pkl", "rb") as f:
                    self._predictions = pickle.load(f)
            else:
                self._process(inputs, outputs)
                with open("det_res.pkl", "wb") as f:
                    pickle.dump(self._predictions, f)

    def evaluate(self, debug=False):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} . "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name
            )
        )
        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                # FIX: use inside class variable
                for thresh in [50]:
                    # if debug:
                    rec, prec, ap = debug_voc_eval(
                        self._detect_result, 
                        res_file_template, 
                        self._input_img_folder, 
                        self._json_folder,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )

                    # else:
                    rec, prec, ap = voc_eval(
                        res_file_template, 
                        self._input_img_folder, 
                        self._json_folder,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )
                    aps[thresh].append(ap * 100)
        print(aps)
        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"mAP": mAP[50], }
        return ret


def get_hoi_box(human_bbox, object_bbox):
    h_xmin, h_ymin, h_xmax, h_ymax = human_bbox
    o_xmin, o_ymin, o_xmax, o_ymax = object_bbox
    xmin, ymin = min(h_xmin, o_xmin), min(h_ymin, o_ymin)
    xmax, ymax = max(h_xmax, o_xmax), max(h_ymax, o_ymax)
    hoi_bbox = [xmin, ymin, xmax, ymax]
    return hoi_bbox

##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, img_folder, json_folder, class_name, ovthresh=0.5):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                )
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images

    # read dets
    detfile = detpath.format(class_name)
    with open(detfile, "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [int(x[0]) for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bboxes = [ [x[2], x[3], x[4], x[5]] for x in splitlines]
    sorted_ind = np.argsort(confidence)
    sorted_ind = np.flip(sorted_ind)
    
    global_ids = ["HICO_test2015_" + str(x).zfill(8) + ".jpg" for x in image_ids]
    bboxes = np.array(bboxes)
    bboxes = bboxes[sorted_ind]

    global_ids = sorted(list(set(global_ids)))

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # load annots
    with open(osp.join(json_folder, "anno_list.json")) as jfp:
        full_data = json.load(jfp)

    class_names = []    
    with open(osp.join(json_folder, "object_list.json")) as jfp:
        object_list = json.load(jfp)
        for each_object in object_list:
            object_name = each_object["name"]
            class_names.append(object_name)
    with open(osp.join(json_folder, "hoi_list.json")) as jfp:
        hoi_list = json.load(jfp)
        for i in range(600):
            class_names.append(str(i+1))
    class_names = ["person"]

    # checkLater = ["HICO_test2015_00003183", 
    #               "HICO_test2015_00008435", 
    #                 "HICO_train2015_00011533",
    #                 "HICO_test2015_00007684",
    #                 "HICO_test2015_00008817" ]
                
    recs = {}
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
        recs[image_id] = instances

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for image_id in image_ids:
        instances = recs[image_id]
        bbox = np.array([x["bbox"] for x in instances])
        det = [False] * len(instances)
        difficult = [False] * len(instances)
        npos += len(bbox)
        class_recs[image_id] = {"bbox": bbox, "difficult": difficult, "det": det}

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R["bbox"]).astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            # print(iw, ih)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0
    # print("tp: {}".format(tp))
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap

def debug_voc_eval(detect_result, detpath, img_folder, json_folder, class_name, ovthresh=0.5):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                )
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    
    # read dets
    detection_results = []
    image_ids = []
    for k, v in detect_result.items():
        image_id = k
        dets = v

        for det in dets:
            detection_results.append(det)
        image_ids.append(image_id)

    global_confidence = [det["score"] for det in detection_results]
    global_sorted_ind = np.argsort(global_confidence)
    global_sorted_ind = np.flip(global_sorted_ind)

    # load annots
    with open(osp.join(json_folder, "anno_list.json")) as jfp:
        full_data = json.load(jfp)

    class_names = []    
    with open(osp.join(json_folder, "object_list.json")) as jfp:
        object_list = json.load(jfp)
        for each_object in object_list:
            object_name = each_object["name"]
            class_names.append(object_name)
    with open(osp.join(json_folder, "hoi_list.json")) as jfp:
        hoi_list = json.load(jfp)
        for i in range(600):
            class_names.append(str(i+1))
    class_names = ["person"]

    recs = {}
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
        recs[image_id] = instances

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for image_id in image_ids:
        instances = recs[image_id]
        bbox = np.array([x["bbox"] for x in instances])
        det = [False] * len(instances)
        difficult = [False] * len(instances)
        npos += len(bbox)
        class_recs[image_id] = {"bbox": bbox, "difficult": difficult, "det": det}

    tp = np.zeros(1,)
    fp = np.zeros(1,)

    for k, v in detect_result.items():
        image_id = k
        dets = v

        results = []
        
        results = dets

        current_nd = len(dets)
        current_tp = np.zeros(current_nd)
        current_fp = np.zeros(current_nd)
        confidence = [det["score"] for det in results]
        sorted_ind = np.argsort(confidence)
        sorted_ind = np.flip(sorted_ind)
        R = class_recs[image_id]

        for index in sorted_ind:
            bb = results[index]["box"]
            ovmax = -np.inf
            BBGT = np.array(R["bbox"]).astype(float)
            # print(bb)
            # print(BBGT)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                # print(iw, ih)
                inters = iw * ih

                # union
                uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not R["difficult"][jmax]:
                    if not R["det"][jmax]:
                        current_tp[index] = 1.0
                        R["det"][jmax] = 1
                    else:
                        current_fp[index] = 1.0
            else:
                current_fp[index] = 1.0

        tp = np.concatenate((tp, current_tp), axis=0)
        fp = np.concatenate((fp, current_fp), axis=0)
        # print("image id {} current_tp {}".format(image_id, current_tp))
    # remove first element
    tp = tp[1:]
    fp = fp[1:]

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap
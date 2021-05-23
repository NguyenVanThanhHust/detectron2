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
from utils import get_ground_truth, get_hoi_box

rare_list = []
non_rare_list = []

class HicoEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.
    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, json_folder):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._json_folder = json_folder

        class_names = []
        with open(osp.join(json_folder, "object_list.json")) as jfp:
            object_list = json.load(jfp)
            for each_object in object_list:
                class_names.append(each_object["name"])
        self._class_names = class_names
        
        self._gts = get_ground_truth(json_folder)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._detect_result = dict()
        self._predictions = defaultdict(list)  # class name -> list of prediction strings
        self._detect_result = dict()

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings
        self._detect_result = dict()

    def _process_test(self, det_res_file=None):
        if det_res_file:
            with open(det_res_file, "rb") as f:
                self._detect_result = pickle.load(f)
        else:
            with open(osp.join(self._json_folder, "anno_list.json")) as jfp:
                full_data = json.load(jfp)
            with open(osp.join(self._json_folder, "hoi_list.json")) as jfp:
                hoi_list = json.load(jfp)

            object_names = []
            with open(osp.join(self._json_folder, "object_list.json")) as jfp:
                object_list = json.load(jfp)
                for each_object in object_list:
                    object_names.append(each_object["name"])
                    
            for each_instance in full_data:
                global_id = each_instance["global_id"]
                if "train" in global_id:
                    continue

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
                    verb = hoi_list[int(action) - 1]["verb"]
                    h_xmin, h_ymin, h_xmax, h_ymax = human_bbox
                    o_xmin, o_ymin, o_xmax, o_ymax = object_bbox
                    score = 1.0
                    each_cls = action
                    object_id = object_names.index(object_name) + 1
                    self._predictions[object_id].append(
                        f"{global_id} {score:.3f} {h_xmin:.1f} {h_ymin:.1f} {h_xmax:.1f} {h_ymax:.1f} {o_xmin:.1f} {o_ymin:.1f} {o_xmax:.1f} {o_ymax:.1f} "
                    )

    def _process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, each_cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                _tmp_dict = {"cls": each_cls, "box":[xmin, ymin, xmax, ymax],
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
                lines = predictions.get(cls_id+1, [""])
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                # FIX: use inside class variable
                for thresh in [50]:
                    # if debug:
                    rec, prec, ap = debug_voc_eval(
                        self._gts, 
                        res_file_template, 
                        self._json_folder,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"mAP": mAP[50], }
        print(ret)
        return ret


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

def debug_voc_eval(gts, detpath, json_folder, classname, ovthresh=0.5):
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

    # load annots
    # first load gt
    recs = gts
    full_image_ids = list(recs.keys())
    
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for image_id in full_image_ids:
        instances = recs[image_id]
        instances = [obj for obj in recs[image_id] if obj["object_name"]==classname]
        human_bbox = np.array([x["human_bbox"] for x in instances])
        # print(human_bbox)
        object_bbox = np.array([x["object_bbox"] for x in instances])
        det = [False] * len(instances)
        difficult = [False] * len(instances)
        npos += len(human_bbox)
        class_recs[image_id] = {"human_bbox": human_bbox,"object_bbox": object_bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    # print(splitlines[0])
    Human_Object_BB = np.array([[float(x) for x in x[2:10]] for x in splitlines]).reshape(-1, 8)
    # Object_BB = np.array([[float(z) for z in x[6:9]] for x in splitlines]).reshape(-1, 4)

     # sort by confidence
    sorted_ind = np.argsort(-confidence)
    Human_Object_BB = Human_Object_BB[sorted_ind, :]
    # Object_BB = Object_BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]


    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        hoi_bb = Human_Object_BB[d, :][4:].astype(float)
        ovmax = -np.inf
        HOI_BBGT = R["object_bbox"].astype(float)
        # print(HOI_BBGT.shape)
        # print(hoi_bb)
        # sys.exit()
        if HOI_BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(HOI_BBGT[:, 0], hoi_bb[0])
            iymin = np.maximum(HOI_BBGT[:, 1], hoi_bb[1])
            ixmax = np.minimum(HOI_BBGT[:, 2], hoi_bb[2])
            iymax = np.minimum(HOI_BBGT[:, 3], hoi_bb[3])
            # if classname == "toaster":
            #     print(ixmin, iymin, ixmax, iymax)
            #     print(hoi_bb)
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (hoi_bb[2] - hoi_bb[0] + 1.0) * (hoi_bb[3] - hoi_bb[1] + 1.0)
                + (HOI_BBGT[:, 2] - HOI_BBGT[:, 0] + 1.0) * (HOI_BBGT[:, 3] - HOI_BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            # sys.exit()
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

    # compute precision recall
    # if classname=="toaster":
    if 1.0 in fp or 0.0 in tp:
        print(classname)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap

def test():
    dataset_name = "HICO_DET_test_person"
    # img_folder = "../../../data/HICO_DET/images/test2015"
    json_folder = "../../../data/HICO_DET/hico_det_json"
    # hico_evaluator = HicoEvaluator(dataset_name, img_folder, json_folder)
    hico_evaluator = HicoEvaluator(dataset_name, json_folder)
    hico_evaluator._process_test()
    ret = hico_evaluator.evaluate()

test()
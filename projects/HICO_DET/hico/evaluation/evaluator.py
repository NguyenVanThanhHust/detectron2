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
from utils import get_ground_truth, get_hoi_box, get_hoi_iou

rare_hoi = ['009', '023', '028', '045', '051', '056', '063', '064', '067', '071', '077', '078', '081', '084', '085', '091', '100', '101', '105', '108', '113', '128', '136', '137', '150', '159', '166', '167', '169', '173', '180', '182', '185', '189', '190', '193', '196', '199', '206', '207', '215', '217', '223', '228', '230', '239', '240', '255', '256', '258', '261', '262', '263', '275', '280', '281', '282', '287', '290', '293', '304', '312', '316', '318', '326', '329', '334', '335', '346', '351', '352', '355', '359', '365', '380', '382', '390', '391', '392', '396', '398', '399', '400', '402', '403', '404', '405', '406', '408', '411', '417', '419', '427', '428', '430', '432', '437', '440', '441', '450', '452', '464', '470', '475', '483', '486', '499', '500', '505', '510', '515', '518', '521', '523', '527', '532', '536', '540', '547', '548', '549', '550', '551', '552', '553', '556', '557', '561', '579', '581', '582', '587', '593', '594', '596', '597', '598', '600']

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

    def evaluate(self):
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

            full_aps = defaultdict(list)  # iou -> ap per class
            rare_aps = defaultdict(list)
            non_rare_aps = defaultdict(list)
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_name, [""])
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                # FIX: use inside class variable
                for thresh in [50]:
                    # if debug:
                    rec, prec, ap = voc_eval(
                        self._gts, 
                        res_file_template, 
                        self._json_folder,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )
                    full_aps[thresh].append(ap * 100)
                    if cls_name in rare_hoi:
                        rare_aps[thresh].append(ap * 100)
                    else:
                        non_rare_aps[thresh].append(ap * 100)

        ret = OrderedDict()
        full_mAP = {iou: np.mean(x) for iou, x in full_aps.items()}
        rare_mAP = {iou: np.mean(x) for iou, x in rare_aps.items()}
        non_rare_mAP = {iou: np.mean(x) for iou, x in non_rare_aps.items()}
        
        ret["bbox"] = {"full mAP": full_mAP[50], "rare mAP":rare_mAP[50], "non rare mAP":non_rare_mAP[50]}
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

def voc_eval(gts, detpath, json_folder, classname, ovthresh=0.5):
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
        instances = [obj for obj in recs[image_id] if obj["hoi_id"]==classname]
        human_bbox = np.array([x["human_bbox"] for x in instances])
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
    Human_Object_BB = np.array([[float(x) for x in x[2:10]] for x in splitlines]).reshape(-1, 8)

     # sort by confidence
    sorted_ind = np.argsort(-confidence)
    Human_Object_BB = Human_Object_BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]


    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        hoi_bb = Human_Object_BB[d, :].astype(float)
        ovmax = -np.inf
        OBJ_BBGT = R["object_bbox"].astype(float)
        HUM_BBGT = R["human_bbox"].astype(float)
        assert OBJ_BBGT.shape == HUM_BBGT.shape, "wrong shape, check"
        if OBJ_BBGT.size > 0 :
            jmax, ovmax, is_human_box = get_hoi_iou(OBJ_BBGT, HUM_BBGT, hoi_bb)
        
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
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return rec, prec, ap

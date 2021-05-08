"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import time 
import datetime
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.logger import log_every_n_seconds

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.evaluation import inference_context, print_csv_format
# from data.hico_dataset import *
from data.hico_person_dataset import *
from hico import add_hico_base_config, HicoEvaluator
from hico.data.build import custom_train_loader, custom_test_loader
from detectron2.utils.comm import get_world_size, is_main_process

import torch.nn.functional as F

from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat

logger = logging.getLogger("detectron2")


def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def get_evaluator(cfg, dataset_name, img_folder, json_folder, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = [HicoEvaluator(dataset_name, img_folder, json_folder)]
    
    if len(evaluator_list) == 0:
        raise NotImplementedError("no Evaluator for the dataset {}".format(dataset_name))
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    
    return DatasetEvaluators(evaluator_list)

def inference_custom(cfg, model, data_loader, evaluator):
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    f = open("detect_result.pkl", "wb")
    detect_result = dict()
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()

            # this code works only with batch size 1
            image_id = inputs[0]["image_id"]
            detect_result[image_id] = dict()

            tmp_inputs = model.preprocess_image(inputs)
            features = model.backbone(tmp_inputs.tensor)
            proposals, _ = model.proposal_generator(tmp_inputs, features)
            instances, _ = model.roi_heads(tmp_inputs, features, proposals)
            features = [features[f] for f in model.roi_heads.in_features]
            
            proposal_boxes = [x.proposal_boxes for x in proposals]

            box_features = model.roi_heads.box_pooler(
                features, proposal_boxes
                )
            box_features = model.roi_heads.box_head(box_features)
            pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(box_features)
            
            pred_class_prob = F.softmax(pred_class_logits, -1)
            pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
            boxes = [x.clone().tensor for x in proposal_boxes]
            boxes = torch.cat(boxes)
            keep = batched_nms(boxes, pred_scores, pred_classes, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
            
            detect_result[image_id]["keep"] = keep
            detect_result[image_id]["boxes"] = boxes
            detect_result[image_id]["pred_scores"] = pred_scores
            detect_result[image_id]["box_features"] = box_features
            detect_result[image_id]["pred_classes"] = pred_classes

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
    pickle.dump(detect_result, f)
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    return results

def do_test(cfg, model):
    
    img_folder = "../data/HICO_DET/images/test2015"
    json_folder = "../data/HICO_DET/hico_det_json"
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = custom_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), img_folder, json_folder
        )
        results_i = inference_custom(cfg, model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_hico_base_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    model.eval()
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

import torch
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes
from detectron2.utils.registry import Registry

ROI_HUMAN_HEAD_REGISTRY = Registry("HUMAN_ROI_HEADS")
ROI_HUMAN_HEAD_REGISTRY.__doc__ == """
Registry for contact heads, which make contact predictions from per-region features.
The registered object will be called with obj(cfg,, input_shape).
"""

@ROI_HUMAN_HEAD_REGISTRY.register()
class HumanROIHeads(StandardROIHeads): 

    def __init__(self, cfg, input_shape):
        super(HumanROIHeads, self).__init__(cfg, input_shape)
        self.config = cfg

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            # assert targets
            print(targets)
            proposals = self.label_and_sample_proposals(proposals, targets)    
        
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
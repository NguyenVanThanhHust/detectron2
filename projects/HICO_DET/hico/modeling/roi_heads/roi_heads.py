import torch
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads): 

    def __init__(self, cfg, input_shape):
        super(CustomROIHeads, self).__init__(cfg, input_shape)
        self.config = cfg

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            assert targets
            # print(len(proposals))
            # print(proposals[0])
            print(targets)
            sys.exit()
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
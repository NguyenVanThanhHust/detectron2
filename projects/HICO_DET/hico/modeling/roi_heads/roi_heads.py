import torch
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes
from .object_roi_heads import build_object_roi_head, ObjectROIHeads

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads): 
    def __init__(self, cfg, input_shape):
        super(CustomROIHeads, self).__init__(cfg, input_shape)
        self.config = cfg
        # self.object_roi_heads = ObjectROIHeads(cfg, input_shape)
        self._init_custom_object_roi_head(cfg, input_shape)

    def _init_custom_object_roi_head(self, cfg, input_shape):
        self.object_roi_heads = ObjectROIHeads(cfg, input_shape)

    # def _init_custom_object_roi_head(self, cfg, input_shape):
    #     # modify from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/meshrcnn/modeling/roi_heads/roi_heads.py
    #     in_features       = cfg.MODEL.OBJECT_ROI_HEADS.IN_FEATURES
    #     pooler_resolution = cfg.MODEL.OBJECT_ROI_HEADS.POOLER_RESOLUTION

    #     pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
    #     sampling_ratio    = cfg.MODEL.OBJECT_ROI_HEADS.POOLER_SAMPLING_RATIO
    #     pooler_type       = cfg.MODEL.OBJECT_ROI_HEADS.POOLER_TYPE
    #     self.object_pooler = ROIPooler(
    #         output_size=pooler_resolution,
    #         scales=pooler_scales,
    #         sampling_ratio=sampling_ratio,
    #         pooler_type=pooler_type,
    #     )
    #     in_channels = [input_shape[f].channels for f in self.in_features][0]
    #     shape = ShapeSpec(
    #         channels=in_channels, width=pooler_resolution, height=pooler_resolution
    #     )

    #     self.object_roi_heads = build_object_roi_head(cfg, shape)

    def _forward_object(self, images, features, proposals, targets=None):
        if self.training:
            assert targets
            object_proposals, loss = self.object_roi_heads.forward(images, features, proposals, targets)
            return object_proposals, loss
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)    
            losses = self._forward_box(features, proposals)
            object_proposals, object_losses = self._forward_object(images, features, proposals, targets)

            losses.update(object_losses)
            del images, targets
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
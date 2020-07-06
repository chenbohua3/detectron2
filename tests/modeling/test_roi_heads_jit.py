# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
from unittest import mock
import torch

from detectron2.core.instance import register_fields
attr_list = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor", "pred_classes": "Tensor",
                 "pred_masks": "Tensor", "pred_boxes": "Boxes", "scores": "Tensor", "gt_classes": "Tensor"}
register_fields(attr_list)
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.structures import BitMasks, Boxes, ImageList
from detectron2.utils.events import EventStorage
from detectron2.core.instance import Instances

logger = logging.getLogger(__name__)

"""
Make sure the losses of ROIHeads/RPN do not change, to avoid
breaking the forward logic by mistake.
This relies on assumption that pytorch's RNG is stable.
"""


class ROIHeadsTest(unittest.TestCase):

    def test_roi_heads(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        feature_shape = {"res4": ShapeSpec(channels=num_channels, stride=16)}

        image_shape = (15, 15)
        gt_boxes0 = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        gt_instance0 = Instances(image_shape)
        gt_instance0.gt_boxes = Boxes(gt_boxes0)
        gt_instance0.gt_classes = torch.tensor([2, 1])
        gt_instance0.gt_masks = BitMasks(torch.rand((2,) + image_shape) > 0.5)
        gt_boxes1 = torch.tensor([[1, 5, 2, 8], [7, 3, 10, 5]], dtype=torch.float32)
        gt_instance1 = Instances(image_shape)
        gt_instance1.gt_boxes = Boxes(gt_boxes1)
        gt_instance1.gt_classes = torch.tensor([1, 2])
        gt_instance1.gt_masks = BitMasks(torch.rand((2,) + image_shape) > 0.5)
        gt_instances = [gt_instance0, gt_instance1]

        proposal_generator = build_proposal_generator(cfg, feature_shape).eval()
        roi_heads = StandardROIHeads(cfg, feature_shape).eval()
        roi_heads_jit = torch.jit.script(roi_heads)
        with EventStorage():  # capture events in a new storage to discard them
            proposals, _ = proposal_generator(images, features, gt_instances)
            pred_instance, _ = roi_heads(images, features, proposals, gt_instances)

            proposals_jit = [Instances((10, 10)), Instances((20, 30))]
            proposals_jit[0].proposal_boxes = proposals[0].proposal_boxes
            proposals_jit[0].objectness_logits = proposals[0].objectness_logits
            proposals_jit[1].proposal_boxes = proposals[1].proposal_boxes
            proposals_jit[1].objectness_logits = proposals[1].objectness_logits
            pred_instance_jit, _ = roi_heads_jit(images, features, proposals_jit, gt_instances)

        for i, ins in enumerate(pred_instance_jit):
            for attr in attr_list:
                val = getattr(ins, attr)
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        self.assertTrue(torch.equal(val, pred_instance[i]._fields[attr]))
                    elif isinstance(val, Boxes):
                        self.assertTrue(torch.equal(val.tensor, pred_instance[i]._fields[attr].tensor))


if __name__ == "__main__":
    unittest.main()

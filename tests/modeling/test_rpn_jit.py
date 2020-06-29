# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch
print(torch.__version__)

from detectron2.core import instance
from detectron2.core.instance import register_fields
register_fields({"proposal_boxes": "Boxes", "objectness_logits": "Tensor"})
# sys.modules["detectron2"].structures.Instances = sys.modules["detectron2.core.instance"].Instances

from detectron2.config import get_cfg
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.structures import boxes, Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage


logger = logging.getLogger(__name__)


class RPNTest(unittest.TestCase):

    def test_rpn(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1)

        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape()).eval()

        proposal_generator = torch.jit.script(proposal_generator)
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        image_shape = (15, 15)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        gt_boxes1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        gt_instances1 = instance.Instances(image_shape)
        gt_instances1.gt_boxes = Boxes(gt_boxes1)
        gt_boxes2 = torch.tensor([[2, 2, 6, 6]], dtype=torch.float32)
        gt_instances2 = instance.Instances(image_shape)
        gt_instances2.gt_boxes = Boxes(gt_boxes2)
        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator.forward(
                images, features, [gt_instances1, gt_instances2]
            )

        expected_proposal_boxes = [
            Boxes(torch.tensor([[0, 0, 10, 10], [7.3365392685, 0, 10, 10]])),
            Boxes(
                torch.tensor(
                    [
                        [0, 0, 30, 20],
                        [0, 0, 16.7862777710, 13.1362524033],
                        [0, 0, 30, 13.3173446655],
                        [0, 0, 10.8602609634, 20],
                        [7.7165775299, 0, 27.3875980377, 20],
                    ]
                )
            ),
        ]

        expected_objectness_logits = [
            torch.tensor([0.1225359365, -0.0133192837]),
            torch.tensor([0.1415634006, 0.0989848152, 0.0565387346, -0.0072308783, -0.0428492837]),
        ]

        for proposal, expected_proposal_box, im_size, expected_objectness_logit in zip(
            proposals, expected_proposal_boxes, image_sizes, expected_objectness_logits
        ):
            # self.assertEqual(len(proposal), len(expected_proposal_box))
            self.assertEqual(proposal.image_size, im_size)
            self.assertTrue(
                torch.allclose(proposal.proposal_boxes.tensor, expected_proposal_box.tensor)
            )
            self.assertTrue(torch.allclose(proposal.objectness_logits, expected_objectness_logit))

    def test_rpn_proposals_inf(self):
        N, Hi, Wi, A = 3, 3, 3, 3
        proposals = [torch.rand(N, Hi * Wi * A, 4)]
        pred_logits = [torch.rand(N, Hi * Wi * A)]
        pred_logits[0][1][3:5].fill_(float("inf"))
        find_top_rpn_proposals(proposals, pred_logits, [(10, 10)], 0.5, 1000, 1000, 0, False)


if __name__ == "__main__":
    unittest.main()

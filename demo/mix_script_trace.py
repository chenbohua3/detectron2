import argparse
import copy
import logging
from typing import List
import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2.engine.defaults import build_model
from detectron2.export.torchscript_patch import patch_instances

logger = logging.getLogger(__name__)


def export(model, script_list=None, model_inputs=None):
    def replace_module_with_script(mod, script_list, prefix=""):
        """
        Replace a submodule with scripting torchscript when its name is matched in script_list.
        """
        for name, sub_mod in mod.named_children():
            curr_name = prefix + "." + name if prefix != "" else name
            if curr_name in script_list:
                try:
                    scripted_module = torch.jit.script(sub_mod)
                except Exception as e:
                    logger.warning(
                        str(e) + "\n" + "Fail to convert module {} with scripting module.\n"
                        "We will leave it unchanged and export it through torch.jit.trace"
                        "on top level of the model.".format(curr_name)
                    )
                else:
                    logger.info("Replacing module {} with script module".format(curr_name))
                    setattr(mod, name, scripted_module)
            else:
                replace_module_with_script(sub_mod, script_list, curr_name)

    assert isinstance(model, nn.Module), "Model must be nn.Module"
    # if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
    #     model = model.module

    _model = copy.deepcopy(model)

    if script_list:
        if script_list is True:
            logger.info("Export the whole model through torch.jit.script")
            try:
                scripted_model = torch.jit.script(_model)
            except Exception as e:
                logger.warning(
                    str(e) + "\n" + "Failed! Try to export it through torch.jit.trace:\n"
                )
            else:
                logger.info("Done!")
                return scripted_model

        elif isinstance(script_list, List):
            assert all(isinstance(s, str) for s in script_list)
            logger.info(
                "Submodules listed below will be transformed by torch.jit.script:\n"
                "{}".format(script_list)
            )
            replace_module_with_script(_model, script_list, "")
        else:
            raise TypeError("Only List[str] and bool is supported for script_list")
    else:
        logger.info("Export the whole model through torch.jit.trace")

    try:
        traced_model = torch.jit.trace(_model, model_inputs)
    except Exception as e:
        logger.warning(
            str(e)
            + "\n"
            + "Fail to export torchscript on the top level of the model, We will iterate over "
            "the submodules and replace those that can be successfully exported by the "
            "torch.jit.script"
        )
        replace_module_with_script(_model)
        return _model
    else:
        logger.info("Done!")
        return traced_model


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    dummy = torch.randn
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    model = build_model(cfg).eval()
    fields = {
        "objectness_logits": "Tensor",
        "proposal_boxes": "Boxes",
        "pred_classes": "Tensor",
        "scores": "Tensor",
        "pred_masks": "Tensor",
        "pred_boxes": "Boxes",
        "pred_keypoints": "Tensor",
        "pred_keypoint_heatmaps": "Tensor",
    }
    image = torch.randn(3, 800, 1071).cuda()
    height = torch.tensor(512.0)
    width = torch.tensor(640.0)
    inputs = {"image": image, "height": height, "width": width}
    with patch_instances(fields):
        export(
            model,
            ["proposal_generator", "roi_heads"],
            [
                [
                    inputs,
                ],
            ],
        )

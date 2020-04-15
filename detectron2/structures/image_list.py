# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division
import math
from typing import Any, List, Sequence, Tuple, Union
import torch
from torch.nn import functional as F

@torch.jit.script
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        # we must limit idx to be torch.BoolTensor
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    def to(self,
           other=None,
           device=None,
           dtype=None,
           non_blocking: bool=False,
           copy: bool=False):
        if other is not None:
            cast_tensor = self.tensor.to(other=other, non_blocking=non_blocking, copy=copy)
        else:
            cast_tensor = self.tensor.to(device=device, dtype=dtype , non_blocking=non_blocking, copy=copy)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device


def build_imagelist_from_tensors(
    tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
):
    """
    Args:
        tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad

    Returns:
        an `ImageList`.
    """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
    # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
    max_size = [max(s) for s in zip(*[img.shape for img in tensors])]

    if size_divisibility > 0:

        stride = size_divisibility
        # max_size = list(max_size)
        max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)
        max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)
        max_size = tuple(max_size)

    image_sizes = [tuple(im.shape[-2:]) for im in tensors]

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        # if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
        #     batched_imgs = tensors[0].unsqueeze(0)
        # else:
        #     padded = F.pad(tensors[0], padding_size, value=pad_value)
        #     batched_imgs = padded.unsqueeze_(0)
        batched_imgs = tensors[0].unsqueeze(0)
        for x in padding_size:
            if not x == 0:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
    else:
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return ImageList(batched_imgs.contiguous(), image_sizes)

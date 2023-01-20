#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for SRCNN models.
"""

from __future__ import annotations

__all__ = [
    "SRCNN",
]

from typing import Any

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import Int2T


@constant.LAYER.register()
class SRCNN(base.ConvLayerParsingMixin, nn.Module):
    """SRCNN (Super-Resolution Convolutional Neural Network).
    
    In SRCNN, actually the network is not deep. There are only 3 parts, patch
    extraction and representation, non-linear mapping, and reconstruction.
    
    References:
        https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c
        https://github.com/jspan/dualcnn/blob/master/Denoise/code/srcnn.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size1: Int2T,
        stride1     : Int2T,
        padding1    : Int2T | str,
        kernel_size2: Int2T,
        stride2     : Int2T,
        padding2    : Int2T | str,
        kernel_size3: Int2T,
        stride3     : Int2T,
        padding3    : Int2T | str,
        dilation    : Int2T = 1,
        groups      : int   = 1,
        bias        : bool  = True,
        padding_mode: str   = "zeros",
        device      : Any   = None,
        dtype       : Any   = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = kernel_size1,
            stride       = stride1,
            padding      = padding1,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2 = common.Conv2d(
            in_channels  = 64,
            out_channels = 32,
            kernel_size  = kernel_size2,
            stride       = stride2,
            padding      = padding2,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3 = common.Conv2d(
            in_channels  = 32,
            out_channels = out_channels,
            kernel_size  = kernel_size3,
            stride       = stride3,
            padding      = padding3,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.relu = common.ReLU()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.conv3(y)
        return y

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`mon` package. """

from __future__ import annotations

__all__ = [
    # Extend :mod:`mon.core.typing`
    "CallableType", "ConfigType", "DictType",  "Float1T", "Float2T", "Float3T",
    "Float4T", "Float5T", "Float6T", "FloatAnyT", "Floats", "ImageFormatType",
    "Int1T", "Int2T", "Int3T", "Int4T", "Int5T", "Int6T", "IntAnyT", "Ints",
    "MemoryUnitType", "PathType", "PathsType", "Strs", "VideoFormatType",
    # Extend :mod:`mon.coreimage.typing`
    "AppleRGBType", "BBoxFormatType", "BBoxFormatType", "BBoxType",
    "BasicRGBType", "BorderTypeType", "CFAType", "DistanceMetricType", "Image",
    "Images", "InterpolationModeType", "MaskType", "PaddingModeType",
    "PaddingModeType", "PointsType", "RGBType", "VisionBackendType",
    # Extend :mod:`mon.coreml.typing`
    "CallbackType", "CallbacksType", "ClassLabelsType", "EpochOutput",
    "LRSchedulerType", "LRSchedulersType", "LoggerType", "LoggersType",
    "LossType", "LossesType", "MetricType", "MetricsType", "ModelPhaseType",
    "OptimizerType", "OptimizersType", "ReductionType", "StepOutput",
    "TransformType", "TransformsType", "WeightsType",
    # Extend :mod:`mon.vision.typing`
    "LogitsType",
]

from mon.core.typing import *
from mon.coreimage.typing import *
from mon.coreml.typing import *
from mon.vision.typing import *
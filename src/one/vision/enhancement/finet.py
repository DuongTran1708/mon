#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    # EXPERIMENTAL

    # Ablation 01: Using Conv, change alpha and feature selection
    "finet-a": {
        "name"    : "finet-a",
        "channels": 3,
        "backbone": [
            # [from,       number, module,         args(out_channels, ...)]
            [-1,           1,      Identity,       []],                               # 0  (x)
            # UNet 01 Down
            [-1,           1,      Conv2d,         [64, 3, 1, 1]],                    # 1  (x1)
            [[-1],         1,      FINetConvBlock, [64,   True,  0.2, False, True]],  # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,    [0]],                              # 3  (x1_1_down)
            [[-1],         1,      FINetConvBlock, [128,  True,  0.2, False, True]],  # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,    [0]],                              # 5  (x1_2_down)
            [[-1],         1,      FINetConvBlock, [256,  True,  0.2, False, True]],  # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,    [0]],                              # 7  (x1_3_down)
            [[-1],         1,      FINetConvBlock, [512,  True,  0.2, False, True]],  # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,    [0]],                              # 9  (x1_4_down)
            [[-1],         1,      FINetConvBlock, [1024, False, 0.2, False, True]],  # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,    [1]],                              # 11 (x1_1)
            [-1,           1,      Conv2d,         [64,  3, 1, 1]],                   # 12 (x1_1_skip)
            [4,            1,      ExtractItem,    [1]],                              # 13 (x1_2)
            [-1,           1,      Conv2d,         [128, 3, 1, 1]],                   # 14 (x1_2_skip)
            [6,            1,      ExtractItem,    [1]],                              # 15 (x1_3)
            [-1,           1,      Conv2d,         [256, 3, 1, 1]],                   # 16 (x1_3_skip)
            [8,            1,      ExtractItem,    [1]],                              # 17 (x1_4)
            [-1,           1,      Conv2d,         [512, 3, 1, 1]],                   # 18 (x1_4_skip)
            [10,           1,      ExtractItem,    [1]],                              # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      FINetUpBlock,   [512, 0.2]],                       # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      FINetUpBlock,   [256, 0.2]],                       # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      FINetUpBlock,   [128, 0.2]],                       # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      FINetUpBlock,   [64,  0.2]],                       # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      SAM,            [3]],                              # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,    [0]],                              # 25 (sam_features)
            [-2,           1,      ExtractItem,    [1]],                              # 26 (y1)
            # UNet 02 Down
            [0,            1,      Conv2d,         [64,  3, 1, 1]],                   # 27 (x2)
            [[-1, 25],     1,      Concat,         []],                               # 28 (x2 + sam_features)
            [-1,           1,      Conv2d,         [64,  1, 1, 0]],                   # 29 (x2)
            [[-1, 11, 23], 1,      FINetConvBlock, [64,   True,  0.2, True,  True]],  # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,    [0]],                              # 31 (x2_1_down)
            [[-1, 13, 22], 1,      FINetConvBlock, [128,  True,  0.2, True,  True]],  # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,    [0]],                              # 33 (x2_2_down)
            [[-1, 15, 21], 1,      FINetConvBlock, [256,  True,  0.2, True,  True]],  # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,    [0]],                              # 35 (x2_3_down)
            [[-1, 17, 20], 1,      FINetConvBlock, [512,  True,  0.2, True,  True]],  # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,    [0]],                              # 37 (x2_4_down)
            [[-1],         1,      FINetConvBlock, [1024, False, 0.2, False, True]],  # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,    [1]],                              # 39 (x2_1)
            [-1,           1,      Conv2d,         [64,  3, 1, 1]],                   # 40 (x2_1_skip)
            [32,           1,      ExtractItem,    [1]],                              # 41 (x2_2)
            [-1,           1,      Conv2d,         [128, 3, 1, 1]],                   # 42 (x2_2_skip)
            [34,           1,      ExtractItem,    [1]],                              # 43 (x2_3)
            [-1,           1,      Conv2d,         [256, 3, 1, 1]],                   # 44 (x2_3_skip)
            [36,           1,      ExtractItem,    [1]],                              # 45 (x2_4)
            [-1,           1,      Conv2d,         [512, 3, 1, 1]],                   # 46 (x2_4_skip)
            [38,           1,      ExtractItem,    [1]],                              # 47 (x2_5)
            # UNet 02 Up
            [[-1, 46],     1,      FINetUpBlock,   [512, 0.2]],                       # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      FINetUpBlock,   [256, 0.2]],                       # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      FINetUpBlock,   [128, 0.2]],                       # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      FINetUpBlock,   [64,  0.2]],                       # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head"    : [
            [-1,           1,      Conv2d,         [3, 3, 1, 1]],                     # 52
            [[-1,  0],     1,      Sum,            []],                               # 53
            [[26, -1],     1,      Join,           []],                               # 54
        ],
    },
  
    # Ablation 02: Using GhostConv
    "finet-b": {
        "name"    : "finet-b",
        "channels": 3,
        "backbone": [
            # [from,       number, module,            args(out_channels, ...)]
            [-1,           1,      Identity,          []],                               # 0  (x)
            # UNet 01 Down
            [-1,           1,      GhostConv2d,       [64, 2, 1, 3, 1, 1]],              # 1  (x1)
            [[-1],         1,      FINetGhostConv,    [64,   True,  0.2, False, True]],  # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,       [0]],                              # 3  (x1_1_down)
            [[-1],         1,      FINetGhostConv,    [128,  True,  0.2, False, True]],  # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,       [0]],                              # 5  (x1_2_down)
            [[-1],         1,      FINetGhostConv,    [256,  True,  0.2, False, True]],  # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,       [0]],                              # 7  (x1_3_down)
            [[-1],         1,      FINetGhostConv,    [512,  True,  0.2, False, True]],  # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,       [0]],                              # 9  (x1_4_down)
            [[-1],         1,      FINetGhostConv,    [1024, False, 0.2, False, True]],  # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,       [1]],                              # 11 (x1_1)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 12 (x1_1_skip)
            [4,            1,      ExtractItem,       [1]],                              # 13 (x1_2)
            [-1,           1,      GhostConv2d,       [128, 2, 1, 3, 1, 1]],             # 14 (x1_2_skip)
            [6,            1,      ExtractItem,       [1]],                              # 15 (x1_3)
            [-1,           1,      GhostConv2d,       [256, 2, 1, 3, 1, 1]],             # 16 (x1_3_skip)
            [8,            1,      ExtractItem,       [1]],                              # 17 (x1_4)
            [-1,           1,      GhostConv2d,       [512, 2, 1, 3, 1, 1]],             # 18 (x1_4_skip)
            [10,           1,      ExtractItem,       [1]],                              # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      FINetGhostUpBlock, [512, 0.2]],                       # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      FINetGhostUpBlock, [256, 0.2]],                       # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      FINetGhostUpBlock, [128, 0.2]],                       # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      FINetGhostUpBlock, [64,  0.2]],                       # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      SAM,               [3]],                              # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,       [0]],                              # 25 (sam_features)
            [-2,           1,      ExtractItem,       [1]],                              # 26 (y1)
            # UNet 02 Down
            [0,            1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 27 (x2)
            [[-1, 25],     1,      Concat,            []],                               # 28 (x2 + sam_features)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 1, 1, 0]],             # 29 (x2)
            [[-1, 11, 23], 1,      FINetGhostConv,    [64,   True,  0.2, True,  True]],  # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,       [0]],                              # 31 (x2_1_down)
            [[-1, 13, 22], 1,      FINetGhostConv,    [128,  True,  0.2, True,  True]],  # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,       [0]],                              # 33 (x2_2_down)
            [[-1, 15, 21], 1,      FINetGhostConv,    [256,  True,  0.2, True,  True]],  # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,       [0]],                              # 35 (x2_3_down)
            [[-1, 17, 20], 1,      FINetGhostConv,    [512,  True,  0.2, True,  True]],  # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,       [0]],                              # 37 (x2_4_down)
            [[-1],         1,      FINetGhostConv,    [1024, False, 0.2, False, True]],  # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,       [1]],                              # 39 (x2_1)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 40 (x2_1_skip)
            [32,           1,      ExtractItem,       [1]],                              # 41 (x2_2)
            [-1,           1,      GhostConv2d,       [128, 2, 1, 3, 1, 1]],             # 42 (x2_2_skip)
            [34,           1,      ExtractItem,       [1]],                              # 43 (x2_3)
            [-1,           1,      GhostConv2d,       [256, 2, 1, 3, 1, 1]],             # 44 (x2_3_skip)
            [36,           1,      ExtractItem,       [1]],                              # 45 (x2_4)
            [-1,           1,      GhostConv2d,       [512, 2, 1, 3, 1, 1]],             # 46 (x2_4_skip)
            [38,           1,      ExtractItem,       [1]],                              # 47 (x2_5)
            # UNet 02 Up                                                                 
            [[-1, 46],     1,      FINetGhostUpBlock, [512, 0.2]],                       # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      FINetGhostUpBlock, [256, 0.2]],                       # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      FINetGhostUpBlock, [128, 0.2]],                       # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      FINetGhostUpBlock, [64,  0.2]],                       # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head"    : [
            [-1,           1,      Conv2d,            [3, 3, 1, 1]],                     # 52
            [[-1,  0],     1,      Sum,               []],                               # 53
            [[26, -1],     1,      Join,              []],                               # 54
        ],
    },
    
    # Ablation 03: Using GhostConv and GhostSAM
    "finet-c": {
        "name"    : "finet-c",
        "channels": 3,
        "backbone": [
            # [from,       number, module,            args(out_channels, ...)]
            [-1,           1,      Identity,          []],                               # 0  (x)
            # UNet 01 Down
            [-1,           1,      GhostConv2d,       [64, 2, 1, 3, 1, 1]],              # 1  (x1)
            [[-1],         1,      FINetGhostConv,    [64,   True,  0.2, False, True]],  # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,       [0]],                              # 3  (x1_1_down)
            [[-1],         1,      FINetGhostConv,    [128,  True,  0.2, False, True]],  # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,       [0]],                              # 5  (x1_2_down)
            [[-1],         1,      FINetGhostConv,    [256,  True,  0.2, False, True]],  # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,       [0]],                              # 7  (x1_3_down)
            [[-1],         1,      FINetGhostConv,    [512,  True,  0.2, False, True]],  # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,       [0]],                              # 9  (x1_4_down)
            [[-1],         1,      FINetGhostConv,    [1024, False, 0.2, False, True]],  # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,       [1]],                              # 11 (x1_1)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 12 (x1_1_skip)
            [4,            1,      ExtractItem,       [1]],                              # 13 (x1_2)
            [-1,           1,      GhostConv2d,       [128, 2, 1, 3, 1, 1]],             # 14 (x1_2_skip)
            [6,            1,      ExtractItem,       [1]],                              # 15 (x1_3)
            [-1,           1,      GhostConv2d,       [256, 2, 1, 3, 1, 1]],             # 16 (x1_3_skip)
            [8,            1,      ExtractItem,       [1]],                              # 17 (x1_4)
            [-1,           1,      GhostConv2d,       [512, 2, 1, 3, 1, 1]],             # 18 (x1_4_skip)
            [10,           1,      ExtractItem,       [1]],                              # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      FINetGhostUpBlock, [512, 0.2]],                       # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      FINetGhostUpBlock, [256, 0.2]],                       # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      FINetGhostUpBlock, [128, 0.2]],                       # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      FINetGhostUpBlock, [64,  0.2]],                       # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      GhostSAM,          [3]],                              # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,       [0]],                              # 25 (sam_features)
            [-2,           1,      ExtractItem,       [1]],                              # 26 (y1)
            # UNet 02 Down
            [0,            1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 27 (x2)
            [[-1, 25],     1,      Concat,            []],                               # 28 (x2 + sam_features)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 1, 1, 0]],             # 29 (x2)
            [[-1, 11, 23], 1,      FINetGhostConv,    [64,   True,  0.2, True,  True]],  # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,       [0]],                              # 31 (x2_1_down)
            [[-1, 13, 22], 1,      FINetGhostConv,    [128,  True,  0.2, True,  True]],  # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,       [0]],                              # 33 (x2_2_down)
            [[-1, 15, 21], 1,      FINetGhostConv,    [256,  True,  0.2, True,  True]],  # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,       [0]],                              # 35 (x2_3_down)
            [[-1, 17, 20], 1,      FINetGhostConv,    [512,  True,  0.2, True,  True]],  # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,       [0]],                              # 37 (x2_4_down)
            [[-1],         1,      FINetGhostConv,    [1024, False, 0.2, False, True]],  # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,       [1]],                              # 39 (x2_1)
            [-1,           1,      GhostConv2d,       [64,  2, 1, 3, 1, 1]],             # 40 (x2_1_skip)
            [32,           1,      ExtractItem,       [1]],                              # 41 (x2_2)
            [-1,           1,      GhostConv2d,       [128, 2, 1, 3, 1, 1]],             # 42 (x2_2_skip)
            [34,           1,      ExtractItem,       [1]],                              # 43 (x2_3)
            [-1,           1,      GhostConv2d,       [256, 2, 1, 3, 1, 1]],             # 44 (x2_3_skip)
            [36,           1,      ExtractItem,       [1]],                              # 45 (x2_4)
            [-1,           1,      GhostConv2d,       [512, 2, 1, 3, 1, 1]],             # 46 (x2_4_skip)
            [38,           1,      ExtractItem,       [1]],                              # 47 (x2_5)
            # UNet 02 Up
            [[-1, 46],     1,      FINetGhostUpBlock, [512, 0.2]],                       # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      FINetGhostUpBlock, [256, 0.2]],                       # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      FINetGhostUpBlock, [128, 0.2]],                       # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      FINetGhostUpBlock, [64,  0.2]],                       # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head"    : [
            [-1,           1,      Conv2d,            [3, 3, 1, 1]],                     # 52
            [[-1,  0],     1,      Sum,               []],                               # 53
            [[26, -1],     1,      Join,              []],                               # 54
        ],
    },
}


@MODELS.register(name="finet")
class FINet(ImageEnhancementModel):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        hyperparams (dict | None): Hyperparameters' values. This is usually
            used in hyperparameter tuning (i.e., Grid Search or Random Search).
            Defaults to None.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "finet.yaml",
        hyperparams: dict         | None = None,
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg, variant = parse_cfg_variant(
            cfg     = cfg,
            cfgs    = cfgs,
            cfg_dir = CFG_DIR,
            to_dict = True
        )
        if hyperparams is not None:
            p        = hyperparams["p"]
            scheme   = hyperparams["scheme"]
            pool     = hyperparams["pool"]
            bias     = hyperparams["bias"]
            backbone = cfg["backbone"]
            for i, (f, n, m, args) in enumerate(backbone):
                if m in [FINetConvBlock, FINetGhostConv]:
                    args.extend([p, scheme, pool, bias])
            cfg["backbone"] = backbone
            
        pretrained   = parse_pretrained(pretrained=pretrained, variant=variant)
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            variant     = variant,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINet.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-DCE++
"""

from __future__ import annotations

from one.nn import *
from one.vision.enhancement.zero_dce import CombinedLoss

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zero-dce++": {
        "name"    : "zero-dce++",
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                      # 0  (x)
            [-1,      1,      Downsample,                  [None, 1, "bilinear"]],   # 1  (x_down)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 2
            [-1,      1,      ReLU,                        [True]],                  # 3  (x1)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 4
            [-1,      1,      ReLU,                        [True]],                  # 5  (x2)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 6
            [-1,      1,      ReLU,                        [True]],                  # 7  (x3)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 8
            [-1,      1,      ReLU,                        [True]],                  # 9  (x4)
            [[7, 9],  1,      Concat,                      []],                      # 10
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 11
            [-1,      1,      ReLU,                        [True]],                  # 12 (x5)
            [[5, 12], 1,      Concat,                      []],                      # 13
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 14
            [-1,      1,      ReLU,                        [True]],                  # 15 (x6)
            [[3, 15], 1,      Concat,                      []],                      # 16
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],  # 17 (a)
            [-1,      1,      Tanh,                        []],                      # 18
            [-1,      1,      UpsamplingBilinear2d,        [None, 1]],               # 19
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                     # 20
        ]
    }
}


@MODELS.register(name="zero-dce++")
class ZeroDCEPP(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE_extension
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
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
    
    model_zoo = {
        "zero-dce++-lol": dict(
            name        = "lol",
            path        = "",
            filename    = "zero-dce++-lol.pth",
            num_classes = None,
        ),
        "zero-dce++-sice": dict(
            name        = "sice",
            path        = "",
            filename    = "zero-dce++-sice.pth",
            num_classes = None,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zero-dce++.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "zero-dce++",
        fullname   : str          | None = "zero-dce++",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=1600.0),
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
            pretrained  = ZeroDCEPP.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(tv_weight=1600.0),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) \
            and self.pretrained["name"] in ["sice"]:
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            model_state_dict["2.dw_conv.weight"]  = state_dict["e_conv1.depth_conv.weight"]
            model_state_dict["2.dw_conv.bias"]    = state_dict["e_conv1.depth_conv.bias"]
            model_state_dict["2.pw_conv.weight"]  = state_dict["e_conv1.point_conv.weight"]
            model_state_dict["2.pw_conv.bias"]    = state_dict["e_conv1.point_conv.bias"]
            model_state_dict["4.dw_conv.weight"]  = state_dict["e_conv2.depth_conv.weight"]
            model_state_dict["4.dw_conv.bias"]    = state_dict["e_conv2.depth_conv.bias"]
            model_state_dict["4.pw_conv.weight"]  = state_dict["e_conv2.point_conv.weight"]
            model_state_dict["4.pw_conv.bias"]    = state_dict["e_conv2.point_conv.bias"]
            model_state_dict["6.dw_conv.weight"]  = state_dict["e_conv3.depth_conv.weight"]
            model_state_dict["6.dw_conv.bias"]    = state_dict["e_conv3.depth_conv.bias"]
            model_state_dict["6.pw_conv.weight"]  = state_dict["e_conv3.point_conv.weight"]
            model_state_dict["6.pw_conv.bias"]    = state_dict["e_conv3.point_conv.bias"]
            model_state_dict["8.dw_conv.weight"]  = state_dict["e_conv4.depth_conv.weight"]
            model_state_dict["8.dw_conv.bias"]    = state_dict["e_conv4.depth_conv.bias"]
            model_state_dict["8.pw_conv.weight"]  = state_dict["e_conv4.point_conv.weight"]
            model_state_dict["8.pw_conv.bias"]    = state_dict["e_conv4.point_conv.bias"]
            model_state_dict["11.dw_conv.weight"] = state_dict["e_conv5.depth_conv.weight"]
            model_state_dict["11.dw_conv.bias"]   = state_dict["e_conv5.depth_conv.bias"]
            model_state_dict["11.pw_conv.weight"] = state_dict["e_conv5.point_conv.weight"]
            model_state_dict["11.pw_conv.bias"]   = state_dict["e_conv5.point_conv.bias"]
            model_state_dict["14.dw_conv.weight"] = state_dict["e_conv6.depth_conv.weight"]
            model_state_dict["14.dw_conv.bias"]   = state_dict["e_conv6.depth_conv.bias"]
            model_state_dict["14.pw_conv.weight"] = state_dict["e_conv6.point_conv.weight"]
            model_state_dict["14.pw_conv.bias"]   = state_dict["e_conv6.point_conv.bias"]
            model_state_dict["17.dw_conv.weight"] = state_dict["e_conv7.depth_conv.weight"]
            model_state_dict["17.dw_conv.bias"]   = state_dict["e_conv7.depth_conv.bias"]
            model_state_dict["17.pw_conv.weight"] = state_dict["e_conv7.point_conv.weight"]
            model_state_dict["17.pw_conv.bias"]   = state_dict["e_conv7.point_conv.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss


class DSConv(Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels,
        )
        self.point_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        output = self.depth_conv(input)
        output = self.point_conv(output)
        return output


@MODELS.register(name="zero-dce++-vanilla")
class ZeroDCEPPVanilla(Module):
    """
    Original implementation of ZeroDCE++
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE_extension
    """
    
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        number_f          = 32
        self.relu         = ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample     = UpsamplingBilinear2d(scale_factor=self.scale_factor)
        
        self.conv1 = DSConv(3,            number_f)
        self.conv2 = DSConv(number_f,     number_f)
        self.conv3 = DSConv(number_f,     number_f)
        self.conv4 = DSConv(number_f,     number_f)
        self.conv5 = DSConv(number_f * 2, number_f)
        self.conv6 = DSConv(number_f * 2, number_f)
        self.conv7 = DSConv(number_f * 2, 3       )
    
    def enhance(self, x: Tensor, a: Tensor) -> Tensor:
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        return x
    
    def forward(self, input: Tensor) -> Tensor:
        x = input
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        x1  = self.relu(self.conv1(x_down))
        x2  = self.relu(self.conv2(x1))
        x3  = self.relu(self.conv3(x2))
        x4  = self.relu(self.conv4(x3))
        x5  = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        a = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            a = a
        else:
            a = self.upsample(a)
        x = self.enhance(x, a)
        return x

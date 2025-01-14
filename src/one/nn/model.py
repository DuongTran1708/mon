#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model and training-related components.
"""

from __future__ import annotations

import platform

import torch.cuda
from munch import Munch
from pytorch_lightning.strategies import DDPStrategy
from torch import distributed as dist
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn.modules.loss import _Loss

from one.data import *
from one.nn.layer import *


# H1: - Checkpoint -------------------------------------------------------------

def extract_weights_from_checkpoint(
    ckpt       : Path_,
    weight_file: Path_ | None = None,
):
    """
    Extract and save model's weights from checkpoint file.
    
    Args:
        ckpt (Path_): Checkpoint file.
        weight_file (Path_ | None): Path save the weights file. Defaults to
            None which saves at the same location as the .ckpt file.
    """
    ckpt = Path(ckpt)
    assert_ckpt_file(ckpt)
    
    state_dict = load_state_dict_from_path(str(ckpt))
    if state_dict is None:
        raise ValueError()
    
    if weight_file is None:
        weight_file = ckpt.parent / f"{ckpt.stem}.pth"
    else:
        weight_file = Path(weight_file)
    create_dirs([weight_file.parent])
    torch.save(state_dict, str(weight_file))


def get_epoch(ckpt: Path_ | None) -> int:
    """
    Get the current epoch from the saved weights file.

    Args:
        ckpt (Path_ | None): Checkpoint path.

    Returns:
        Current epoch.
    """
    if ckpt is None:
        return 0
    
    epoch = 0
    ckpt  = Path(ckpt)
    assert_ckpt_file(ckpt)
    if is_torch_saved_file(ckpt):
        ckpt  = torch.load(ckpt)
        epoch = ckpt.get("epoch", 0)
    return epoch


def get_global_step(ckpt: Path_ | None) -> int:
    """
    Get the global step from the saved weights file.

    Args:
        ckpt (Path_ | None): Checkpoint path.

    Returns:
        Global step.
    """
    if ckpt is None:
        return 0

    global_step = 0
    ckpt        = Path(ckpt)
    assert_ckpt_file(ckpt)
    if is_torch_saved_file(ckpt):
        ckpt        = torch.load(ckpt)
        global_step = ckpt.get("global_step", 0)
    return global_step


def get_latest_checkpoint(dirpath: Path_) -> str | None:
    """
    Get the latest weights in the `dir`.

    Args:
        dirpath (Path_): Directory that contains the checkpoints.

    Returns:
        Checkpoint path.
    """
    dirpath  = Path(dirpath)
    ckpt     = get_latest_file(dirpath)
    if ckpt is None:
        error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt


def load_pretrained(
    module	  	: Module,
    path  		: Path_,
    model_dir   : Path_ | None = None,
    map_location: str   | None = torch.device("cpu"),
    progress	: bool 		   = True,
    check_hash	: bool		   = False,
    filename	: str   | None = None,
    strict		: bool		   = False,
    **_
) -> Module:
    """
    Load pretrained weights. This is a very convenient function to load the
    state dict from saved pretrained weights or checkpoints. Filter out mismatch
    keys and then load the layers' weights.
    
    Args:
        module (Module): Module to load pretrained.
        path (Path_): The weights or checkpoints file to load. If it is a URL,
            it will be downloaded.
        model_dir (Path_ | None): Directory to save the weights or checkpoint
            file. Defaults to None.
        map_location (str | None): A function or a dict specifying how to
            remap storage locations (see torch.load). Defaults to `cpu`.
        progress (bool): Whether to display a progress bar to stderr.
            Defaults to True.
        check_hash (bool): If True, the filename part of the URL should follow
            the naming convention `filename-<sha256>.ext` where `<sha256>` is
            the first eight or more digits of the SHA256 hash of the contents
            of the file. Hash is used to ensure unique names and to verify the
            contents of the file. Defaults to False.
        filename (str | None): Name for the downloaded file. Filename from
            `url` will be used if not set.
        strict (bool): Whether to strictly enforce that the keys in `state_dict`
            match the keys returned by this module's
            `~torch.Module.state_dict` function. Defaults to False.
    """
    state_dict = load_state_dict_from_path(
        path         = path,
        model_dir    = model_dir,
        map_location = map_location,
        progress     = progress,
        check_hash   = check_hash,
        filename     = filename
    )
    module = load_state_dict(
        module     = module,
        state_dict = state_dict,
        strict     = strict
    )
    # Debug
    # print(state_dict.keys())
    # print(module.state_dict().keys())
    return module


def load_state_dict(
    module	  : Module,
    state_dict: dict,
    strict    : bool = False,
    **_
) -> Module:
    """
    Load the module state dict. This is an extension of `Module.load_state_dict()`.
    We add an extra snippet to drop missing keys between module's state_dict
    and pretrained state_dict, which will cause an error.

    Args:
        module (Module): Module to load state dict.
        state_dict (dict): A dict containing parameters and persistent buffers.
        strict (bool): Whether to strictly enforce that the keys in `state_dict`
            match the keys returned by this module's `~torch.Module.state_dict`
            function. Defaults to False.

    Returns:
        Module after loading state dict.
    """
    module_dict = module.state_dict()
    module_dict = match_state_dicts(
        model_dict      = module_dict,
        pretrained_dict = state_dict
    )
    module.load_state_dict(module_dict, strict=strict)
    return module


def load_state_dict_from_path(
    path  		: Path_,
    model_dir   : Path_ | None = None,
    map_location: str   | None = torch.device("cpu"),
    progress	: bool 		   = True,
    check_hash	: bool		   = False,
    filename 	: str   | None = None,
    **_
) -> dict | None:
    """
    Load state dict at the given URL. If downloaded file is a zip file, it
    will be automatically decompressed. If the object is already present in
    `model_dir`, it's deserialized and returned.
    
    Args:
        path (Path_): The weights or checkpoints file to load. If it is a URL,
            it will be downloaded.
        model_dir (Path_ | None): Directory in which to save the object.
            Default to None.
        map_location (optional): A function or a dict specifying how to remap
            storage locations (see torch.load). Defaults to `cpu`.
        progress (bool): Whether to display a progress bar to stderr.
            Defaults to True.
        check_hash (bool): If True, the filename part of the URL should follow
            the naming convention `filename-<sha256>.ext` where `<sha256>`
            is the first eight or more digits of the SHA256 hash of the
            contents of the file. Hash is used to ensure unique names and to
            verify the contents of the file. Defaults to False.
        filename (str | None): Name for the downloaded file. Filename from
            `url` will be used if not set.
    """
    if path is None:
        raise ValueError()
    if model_dir:
        model_dir = Path(model_dir)
    
    path = Path(path)
    if not is_torch_saved_file(path) and \
        (model_dir is None or not model_dir.is_dir()):
        raise ValueError(f"`model_dir` must be defined. But got: {model_dir}.")
    
    save_weight = ""
    if filename:
        save_weight = model_dir / filename
    
    state_dict = None
    if is_torch_saved_file(save_weight):
        state_dict = torch.load(str(save_weight), map_location=map_location)
    elif is_torch_saved_file(path):
        state_dict = torch.load(str(path), map_location=map_location)
    elif is_url(path):
        state_dict = load_state_dict_from_url(
            url          = str(path),
            model_dir    = str(model_dir),
            map_location = map_location,
            progress     = progress,
            check_hash   = check_hash,
            file_name    = filename
        )
    
    if state_dict and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if state_dict and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def match_state_dicts(
    model_dict	   : dict,
    pretrained_dict: dict,
    exclude		   : tuple | list = ()
) -> dict:
    """
    Filter out unmatched keys btw the model's `state_dict` and the pretrained's
    `state_dict`. Omitting `exclude` keys.

    Args:
        model_dict (dict): Model's `state_dict`.
        pretrained_dict (dict): Pretrained's `state_dict`.
        exclude (tuple | list): List of excluded keys. Defaults to ().
        
    Returns:
        Filtered model's `state_dict`.
    """
    # 1. Filter out unnecessary keys
    intersect_dict = intersect_weight_dicts(
        pretrained_dict,
        model_dict,
        exclude
    )
    """
       intersect_dict = {
           k: v for k, v in pretrained_dict.items()
           if k in model_dict and
              not any(x in k for x in exclude) and
              v.shape == model_dict[k].shape
       }
       """
    # 2. Overwrite entries in the existing state dict
    model_dict.update(intersect_dict)
    return model_dict


# H1: - Distribution -----------------------------------------------------------

def get_distributed_info() -> tuple[int, int]:
    """
    If distributed is available, return the rank and world size, otherwise
    return 0 and 1
    
    Returns:
        The rank and world size of the current process.
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1
    return rank, world_size


def is_parallel(model: Module) -> bool:
    """
    If the model is a parallel model, then it returns True, otherwise it returns
    False
    
    Args:
        model (Module): The model to check.
    
    Returns:
        A boolean value.
    """
    return type(model) in (
        parallel.DataParallel,
        parallel.DistributedDataParallel
    )


def set_distributed_backend(strategy: str | Callable, cudnn: bool = True):
    """
    If you're running on Windows, set the distributed backend to gloo. If you're
    running on Linux, set the distributed backend to nccl
    
    Args:
        strategy (str | Callable): The distributed strategy to use. One of
            ["ddp", "ddp2"]
        cudnn (bool): Whether to use cuDNN or not. Defaults to True.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        console.log(
            f"cuDNN available: [bright_green]True[/bright_green], "
            f"used:" + "[bright_green]True" if cudnn else "[red]False"
        )
    else:
        console.log(f"cuDNN available: [red]False")
    
    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        if platform.system() == "Windows":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
            console.log(
                "Running on a Windows machine, set torch distributed backend "
                "to gloo."
            )
        elif platform.system() == "Linux":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            console.log(
                "Running on a Unix machine, set torch distributed backend "
                "to nccl."
            )


# H1: - Model ------------------------------------------------------------------

def attempt_load(
    name       : str,
    cfg        : Path_,
    weights    : Path_,
    fullname   : str | None = None,
    num_classes: int | None = None,
    phase      : str        = "inference",
    *args, **kwargs
) -> BaseModel:
    if is_ckpt_file(weights):
        model = MODELS.build(
            name        = name,
            cfg         = cfg,
            num_classes = num_classes,
            phase       = "inference",
        )
        model = model.load_from_checkpoint(
            checkpoint_path = weights,
            name            = name,
            cfg             = cfg,
            num_classes     = num_classes,
            phase           = "inference",
        )
    else:
        model = MODELS.build(
            name        = name,
            cfg         = cfg,
            pretrained  = weights,
            num_classes = num_classes,
            phase       = "inference",
        )
    if fullname is not None:
        model.fullname = fullname
    return model


def parse_cfg_variant(
    cfg    : dict | Path_ | None,
    cfgs   : dict,
    cfg_dir: Path,
    to_dict: bool = True,
) -> tuple[Path_ | dict | Munch, str]:
    variant = None
    if isinstance(cfg, str) and cfg in cfgs:
        variant = str(cfg)
        cfg     = cfgs[cfg]
    elif isinstance(cfg, (str, Path)):
        if not is_yaml_file(cfg):
            cfg = cfg_dir / cfg
        variant = str(cfg.stem)
    elif isinstance(cfg, dict):
        variant = cfg.get("name", None)
    else:
        error_console.log(
            f"`cfg` must be a dict or one of: {cfgs.keys()}. But got: {cfg}."
        )
    if to_dict:
        cfg = load_config(cfg=cfg)
    return cfg, variant


def parse_model(
    d : dict      | None = None,
    ch: list[int] | None = None
) -> tuple[Sequential, list[int], list[dict]]:
    """
    Build the model. We inherit the same idea of model parsing in YOLOv5.
    
    Each layer should have the following attributes:
        - i (int): index of the layer.
        - f (int | list[int]): from, i.e., the current layer receive output
          from the f-th layer. For example: -1 means from previous layer;
          -2 means from 2 previous layers; [99, 101] means from the 99th
          and 101st layers. This attribute is used in forward pass.
        - t: type of the layer using this script:
          t = str(m)[8:-2].replace("__main__.", "")
        - np (int): number of parameters using the following script:
          np = sum([x.numel() for x in m.parameters()])
    
    Args:
        d (dict | None): Model definition dictionary. Default to None means
            building the model manually.
        ch (list[int] | None): The first layer's input channels. If given,
            it will be used to further calculate the next layer's input
            channels. Defaults to None means defines each layer in_ and
            out_channels manually.
    
    Returns:
        A Sequential model.
        A list of layer index to save the features during forward pass.
        A list of layer's info (dict) for debugging.
    """
    anchors = d.get("anchors",        None)
    nc      = d.get("num_classes",    None)
    gd      = d.get("depth_multiple", 1)
    gw      = d.get("width_multiple", 1)
    
    layers = []      # layers
    save   = []      # savelist
    ch     = ch or [3]
    c2     = ch[-1]  # out_channels
    info   = []      # print data as table
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # Convert string class name into class
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            if a == "random":
                continue
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        # Group 01
        if m in [
            ABSConv2dS,
            ABSConv2dS1,
            ABSConv2dS2,
            ABSConv2dS3,
            ABSConv2dS4,
            ABSConv2dS5,
            ABSConv2dS6,
            ABSConv2dS7,
            ABSConv2dS8,
            ABSConv2dS9,
            ABSConv2dS10,
            ABSConv2dS11,
            ABSConv2dS12,
            ABSConv2dS13,
            ABSConv2dU,
            ADCE,
            BSConv2dS,
            BSConv2dU,
            Conv2d,
            Conv2dNormActivation,
            ConvAct2d,
            ConvReLU2d,
            ConvTranspose2d,
            DCE,
            DepthwiseSeparableConv2d,
            DepthwiseSeparableConvReLU2d,
            EnhancementModule,
            FFAPostProcess,
            FFAPreProcess,
            FINetConvBlock,
            FINetUpBlock,
            FINetGhostConv,
            FINetGhostUpBlock,
            GhostConv2d,
            HINetConvBlock,
            HINetUpBlock,
            InceptionBasicConv2d,
            MobileOneConv2d,
            MobileOneStage,
            SRCNN,
            UnconstrainedBlueprintSeparableConv2d,
            UNetBlock,
            VDSR,
        ]:
            if isinstance(f, (list, tuple)):
                c1, c2 = ch[f[0]], args[0]
            else:
                c1, c2 = ch[f],    args[0]
            args = [c1, c2, *args[1:]]
        # Group 02
        elif m in [
            FFA,
            FFABlock,
            FFAGroup,
            PixelAttentionModule,
            GhostSAM,
            GhostSupervisedAttentionModule,
            SAM,
            SupervisedAttentionModule,
        ]:
            if isinstance(f, (list, tuple)):
                c1 = c2 = ch[f[0]]
            else:
                c1 = c2 = ch[f]
            args = [c1, *args[0:]]
        # Group 03
        elif m in [
            InvertedResidual,
        ]:
            if isinstance(f, (list, tuple)):
                c1, c2 = ch[f[0]], args[0]
            else:
                c1, c2 = ch[f],    args[0]
            args = [c1, *args[0:]]
        # Group 04
        elif m in [
            AlexNetClassifier,
            ConvNeXtClassifier,
            GoogleNetClassifier,
            InceptionAux1,
            InceptionAux2,
            InceptionClassifier,
            LeNetClassifier,
            LinearClassifier,
            MobileOneClassifier,
            ShuffleNetV2Classifier,
            SqueezeNetClassifier,
            VGGClassifier,
        ]:
            c1   = args[0]
            c2   = nc
            args = [c1, c2, *args[1:]]
        # Group 05
        elif m in [
            BatchNorm2d,
        ]:
            args = [ch[f]]
        # Group 06
        elif m in [
            DenseBlock,
            DenseTransition,
            Fire,
            Inception,
            InceptionA,
            InceptionB,
            InceptionC,
            InceptionD,
            InceptionE,
        ]:
            c1 = args[0]
            if m in [DenseBlock]:
                out_channels = args[1]
                num_layers   = args[2]
                c2           = c1 + out_channels * num_layers
            elif m in [DenseTransition]:
                c2           = c1 // 2
            if m in [Fire]:
                expand1x1_planes = args[2]
                expand3x3_planes = args[3]
                c2               = expand1x1_planes + expand3x3_planes
            elif m in [Inception]:
                ch1x1     = args[1]
                ch3x3     = args[3]
                ch5x5     = args[5]
                pool_proj = args[6]
                c2        = ch1x1 + ch3x3 + ch5x5 + pool_proj
            elif m in [InceptionA]:
                c2 = m.base_out_channels + args[1]
            elif m in [InceptionB, InceptionD]:
                c2 = m.base_out_channels + c1
            elif m in [InceptionC, InceptionE]:
                c2 = m.base_out_channels
        # Group 07
        elif m in [
            ResNetBlock,
        ]:
            c1 = args[2]
            c2 = args[3]
        # Group 08
        elif m in [
            Join,
            PixelwiseHigherOrderLECurve,
            Shortcut,
            Sum,
        ]:
            c2 = ch[f[-1]]
        # Group 09
        elif m in [Foldcut]:
            c2 = ch[f] // 2
        # Group 10
        elif m in [
            Chuncat,
            Concat,
            InterpolateConcat,
        ]:
            c2 = sum([ch[x] for x in f])
        # Group 11
        elif m in [
            ExtractFeature,
        ]:
            c2 = args[0]
        # Group 12
        elif m in [
            ExtractFeatures,
        ]:
            c2 = args[1] - args[0]
        # Group 13
        else:
            c2 = ch[f]
        
        # Append c2 as c1 for next layers
        if i == 0:
            ch = []
        ch.append(c2)
        
        # Create layers
        m_    = Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        m_.i  = i
        m_.f  = f
        m_.t  = t  = str(m)[8:-2].replace("__main__.", "")      # module type
        m_.np = np = sum([x.numel() for x in m_.parameters()])  # number params
        sa    = [x % i for x in ([f] if isinstance(f, int) else f) if x != -1]
        save.extend(sa)  # append to savelist
        layers.append(m_)
        info.append({
            "index"    : i,
            "from"     : f,
            "n"        : n,
            "params"   : np,
            "module"   : t,
            "arguments": args,
        })
    
    return Sequential(*layers), sorted(save), info


def parse_pretrained(pretrained: Pretrained, variant: str | None = None) -> Pretrained:
    if isinstance(pretrained, str) and not is_torch_saved_file(path=pretrained):
        if (variant is not None) and (variant not in pretrained):
            pretrained = f"{variant}-{pretrained}"
    return pretrained


def sparsity(model: Module) -> float:
    """
    Return global model sparsity.
    """
    a = 0.0
    b = 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def strip_optimizer(weight_file: str, new_file: str = ""):
    """
    Strip optimizer from saved weight file to finalize training.
    Optionally save as `new_file`.
    """
    assert_weights_file(weight_file)
    
    x = torch.load(weight_file, map_location=torch.device("cpu"))
    x["optimizer"]        = None
    x["training_results"] = None
    x["epoch"]            = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
        
    torch.save(x, new_file or weight_file)
    mb = os.path.getsize(new_file or weight_file) / 1E6  # filesize
    console.log(
        "Optimizer stripped from %s,%s %.1fMB"
        % (weight_file, (" saved as %s," % new_file) if new_file else "", mb)
    )


# H2: - Base Model -------------------------------------------------------------

class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    """
    Base model for all models. Base model only provides access to the
    attributes. In the model, each head is responsible for generating the
    appropriate output with accommodating loss and metric (obviously, we can
    only calculate specific loss and metric with specific output type). So we
    define the loss functions and metrics in the head implementation instead of
    the model.
    
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
    
    model_zoo = {}  # A dictionary of all pretrained weights.
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = None,
        hyperparams: dict         | None = None,
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = None,
        variant    : str          | None = None,
        fullname   : str          | None = None,
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg           = cfg
        self.hyperparams   = hyperparams
        self.name          = name
        self.fullname      = fullname
        self.variant       = variant
        self.project       = project
        self.root          = root
        self.num_classes   = num_classes
        self.pretrained    = pretrained
        self.loss          = loss
        self.train_metrics = metrics
        self.val_metrics   = metrics
        self.test_metrics  = metrics
        self.optims        = optimizers
        self.debug         = debug
        self.verbose       = verbose
        self.epoch_step    = 0
        
        # Define model
        self.model = None
        self.save  = None
        self.info  = None
        
        if self.cfg is not None:
            console.log(f"Parsing model from `cfg`.")
            self.cfg = load_config(cfg=self.cfg)
            assert_dict(self.cfg)
            
            self.channels        = self.cfg.get("channels", channels)
            self.cfg["channels"] = self.channels
            
            self.classlabels = ClassLabels.from_value(classlabels)
            if self.classlabels:
                num_classes = num_classes or self.classlabels.num_classes()
            if is_dict(pretrained) and "num_classes" in self.pretrained:
                num_classes = num_classes or self.pretrained["num_classes"]
            self.num_classes = num_classes
            
            if self.num_classes:
                nc = self.cfg.get("num_classes", None)
                if self.num_classes != nc:
                    console.log(
                        f"Overriding model.yaml num_classes={nc} "
                        f"with num_classes={self.num_classes}."
                    )
                    self.cfg["num_classes"] = self.num_classes
            
            assert_dict_contain_keys(input=self.cfg, keys=["backbone", "head"])
            
            # Actual model, save list during forward, layer's info
            self.model, self.save, self.info = self.parse_model(
                d  = self.cfg,
                ch = [self.channels],
            )
            
            # Parse hyperparameters if given
            if self.hyperparams is not None:
                self.cfg = self.parse_hyperparams(hyperparams=self.hyperparams)
            
            # Load pretrained
            if self.pretrained:
                self.load_pretrained()
            else:
                self.apply(self.init_weights)
            self.print_info()

        # Set phase to freeze or unfreeze layers
        self.phase = phase
        
    @property
    def debug(self) -> Munch | None:
        return self._debug
    
    @debug.setter
    def debug(self, debug: dict | Munch | None):
        if debug is None:
            self._debug = None
        else:
            if isinstance(debug, dict):
                debug = Munch.fromDict(debug)
            self._debug = debug
        
            if "every_best_epoch" not in self._debug:
                self._debug.every_best_epoch = True
            if "every_n_epochs" not in self._debug:
                self._debug.every_n_epochs = 1
            if "save_to_subdir" not in self._debug:
                self._debug.save_to_subdir = True
            if "image_quality" not in self._debug:
                self._debug.image_quality = 95
            if "max_n" not in self._debug:
                self._debug.max_n = 8
            if "nrow" not in self._debug:
                self._debug.nrow = 8
            if "wait_time" not in self._debug:
                self._debug.wait_time = 0.01
    
    @property
    def debug_dir(self) -> Path:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def debug_subdir(self) -> Path:
        """
        Return the debug subdir path located at: <debug_dir>/<phase>_<epoch>.
        """
        debug_dir = self.debug_dir / \
                    f"{self.phase.value}_{(self.current_epoch + 1):03d}"
        create_dirs(paths=[debug_dir])
        return debug_dir
    
    @property
    def debug_image_filepath(self) -> Path:
        """
        Return the debug image filepath located at: <debug_dir>/
        """
        save_dir = self.debug_subdir \
            if self.debug.save_to_subdir \
            else self.debug_dir
            
        return save_dir / f"{self.phase.value}_" \
                          f"{(self.current_epoch + 1):03d}_" \
                          f"{(self.epoch_step + 1):06}.jpg"
    
    @property
    def dim(self) -> int | None:
        """
        Return the number of dimensions.
        """
        return None if self.size is None else len(self.size)
    
    @property
    def fullname(self) -> str:
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str | None = None):
        """
        Assign the model full name in the following format:
        {name}-{data_name}-{postfix}. For example: `yolov5-coco-1920`
 
        Args:
            fullname (str | None): Model fullname. In case None is given, it
                will be `self.name`. Defaults to None.
        """
        self._fullname = fullname \
            if (fullname is not None and fullname != "") \
            else self.name
    
    @property
    def loss(self) -> _Loss | None:
        return self._loss
    
    @loss.setter
    def loss(self, loss: Losses_ | None):
        if isinstance(loss, (_Loss, Module)):
            self._loss = loss
        elif isinstance(loss, dict):
            self._loss = LOSSES.build_from_dict(cfg=loss)
        else:
            self._loss = None
        
        if self._loss:
            self._loss.requires_grad = True
            # self._loss.cuda()
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str | None = None):
        """
        Assign the model's name.
        
        For example: `yolov7-e6-coco`, the name is `yolov7`.
        
        Args:
            name (str | None): Model name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
		"""
        self._name = name \
            if (name is not None and name != "") \
            else self.__class__.__name__.lower()
    
    @property
    def ndim(self) -> int | None:
        """
        Alias of `self.dim()`.
        """
        return self.dim

    @property
    def params(self) -> int:
        if self.info is not None:
            params = [i["params"] for i in self.info]
            return sum(params)
        else:
            return 0
    
    @property
    def phase(self) -> ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhase_ = "training"):
        """
        Assign the model's running phase.
        """
        self._phase = ModelPhase.from_value(phase)
        if self._phase is ModelPhase.TRAINING:
            self.unfreeze()
            if self.cfg is not None:
                freeze = self.cfg.get("freeze", None)
                if is_list(freeze):
                    for k, v in self.model.named_parameters():
                        if any(x in k for x in freeze):
                            v.requires_grad = False
        else:
            self.freeze()
    
    @property
    def pretrained_dir(self) -> Path:
        return PRETRAINED_DIR / self.name
    
    @property
    def root(self) -> Path:
        return self._root
    
    @root.setter
    def root(self, root: Path_):
        """
        Assign the root directory of the model.
        
        Args:
            root (Path_): The root directory of the model.
        """
        if root is None:
            root = RUNS_DIR / "train"
        else:
            root = Path(root)
        self._root = root
        
        if self.project is not None and self.project != "":
            self._root = self._root / self.project
        if self._root.name != self.fullname:
            self._root = self._root / self.fullname

        self._debug_dir   = self._root / "debugs"
        self._weights_dir = self._root / "weights"
    
    @property
    def train_metrics(self) -> list[Metric] | None:
        return self._train_metrics
    
    @train_metrics.setter
    def train_metrics(self, metrics: Metrics_ | None):
        """
        Assign train metrics.
        
        Args:
            metrics (Metrics_): One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "train" in metrics:
            metrics = metrics.get("train", metrics)
            
        self._train_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._train_metrics:
            for metric in self._train_metrics:
                name = f"train_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def val_metrics(self) -> list[Metric] | None:
        return self._val_metrics
    
    @val_metrics.setter
    def val_metrics(self, metrics: Metrics_ | None):
        """
        Assign val metrics.
        
        Args:
            metrics (Metrics_): One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "val" in metrics:
            metrics = metrics.get("val", metrics)
            
        self._val_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class
        if self._val_metrics:
            for metric in self._val_metrics:
                name = f"val_{metric.name}"
                setattr(self, name, metric)
    
    @property
    def test_metrics(self) -> list[Metric] | None:
        return self._test_metrics
    
    @test_metrics.setter
    def test_metrics(self, metrics: Metrics_ | None):
        """
        Assign test metrics.
        
        Args:
            metrics (Metrics_): One of the 2 options:
                - Common metrics for train_/val_/test_metrics:
                    "metrics": dict(name="accuracy")
                  or,
                    "metrics": [dict(name="accuracy"), torchmetrics.Accuracy(),]
                
                - Define train_/val_/test_metrics separately:
                    "metrics": {
                        "train": [dict(name="accuracy"), dict(name="f1")]
                        "val":   torchmetrics.Accuracy(),
                        "test":  None,
                    }
        """
        if isinstance(metrics, dict) and "test" in metrics:
            metrics = metrics.get("test", metrics)
            
        self._test_metrics = self.create_metrics(metrics)
        # This is a simple hack since LightningModule require the
        # metric to be defined with self.<metric>. Here we dynamically
        # add the metric attribute to the class.
        if self._test_metrics:
            for metric in self._test_metrics:
                name = f"test_{metric.name}"
                setattr(self, name, metric)
        
    @property
    def weights_dir(self) -> Path:
        if self._weights_dir is None:
            self._weights_dir = self.root / "weights"
        return self._weights_dir
    
    @staticmethod
    def create_metrics(metrics: Metrics_ | None) -> list[Metric] | None:
        if isinstance(metrics, Metric):
            return [metrics]
        elif isinstance(metrics, dict):
            return [METRICS.build_from_dict(cfg=metrics)]
        elif isinstance(metrics, list):
            return [METRICS.build_from_dict(cfg=m)
                    if isinstance(m, dict)
                    else m for m in metrics]
        else:
            return None
    
    @abstractmethod
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        """
        Build the model. You have 2 options to build a model: (1) define each
        layer manually, or (2) build model automatically from a config
        dictionary.
        
        We inherit the same idea of model parsing in YOLOv5.
        
        Either way each layer should have the following attributes:
            - i (int): index of the layer.
            - f (int | list[int]): from, i.e., the current layer receive output
              from the f-th layer. For example: -1 means from previous layer;
              -2 means from 2 previous layers; [99, 101] means from the 99th
              and 101st layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
              t = str(m)[8:-2].replace("__main__.", "")
            - np (int): number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d (dict | None): Model definition dictionary. Default to None means
                building the model manually.
            ch (list[int] | None): The first layer's input channels. If given,
                it will be used to further calculate the next layer's input
                channels. Defaults to None means defines each layer in_ and
                out_channels manually.
        
        Returns:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info for debugging.
        """
        pass
    
    @abstractmethod
    def parse_hyperparams(self, hyperparams: dict | Munch) -> dict | Munch:
        """
        Update layers' parameters with the provided hyperparameters. Specify
        which layer and which parameter will be updated in this function. This
        function is called before `parse_model`. This is usually used in
        hyperparameter tuning procedure (i.e, Grid Search or Random Search).
        
        Args:
            hyperparams (dict | Munch): Hyperparameters' values.
        """
        pass
    
    @classmethod
    def init_pretrained(cls, pretrained: Pretrained = False):
        """
        Assign model's pretrained.
        
        Args:
            pretrained (Pretrained): Initialize weights from pretrained.
                - If True, use the original pretrained described by the
                  author (usually, ImageNet or COCO). By default, it is the
                  first element in the `model_zoo` dictionary.
                - If str and is a file/path, then load weights from saved
                  file.
                - In each inherited model, `pretrained` can be a dictionary's
                  key to get the corresponding local file or url of the weight.
        """
        if is_dict(pretrained):
            return pretrained
        if pretrained is True and len(cls.model_zoo):
            return list(cls.model_zoo.values())[0]
        elif pretrained in cls.model_zoo:
            return cls.model_zoo[pretrained]
        else:
            return pretrained
    
    @abstractmethod
    def init_weights(self, m: Module):
        """
        Initialize model's weights.
        """
        pass
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained):
            create_dirs(paths=[self.pretrained_dir])
            load_pretrained(
                module	  = self,
                model_dir = self.pretrained_dir,
                strict	  = False,
                **self.pretrained
            )
            if self.verbose:
                console.log(f"Load pretrained from: {self.pretrained}!")
        elif is_url_or_file(self.pretrained):
            load_pretrained(
                module    = self,
                path 	  = str(self.pretrained),
                model_dir = self.pretrained_dir,
                strict	  = False,
            )
            # raise NotImplementedError("This function has not been implemented.")
        else:
            error_console.log(f"[yellow]Cannot load from pretrained: "
                              f"{self.pretrained}!")
    
    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you’d need one. But in the case of GANs or
        similar you might have multiple.

        Returns:
            Any of these 6 options:
                - Single optimizer.
                - List or Tuple of optimizers.
                - Two lists - First list has multiple optimizers, and the
                  second has multiple LR schedulers (or multiple
                  lr_scheduler_config).
                - Dictionary, with an "optimizer" key, and (optionally) a
                  "lr_scheduler" key whose value is a single LR scheduler or
                  lr_scheduler_config.
                - Tuple of dictionaries as described above, with an optional
                  "frequency" key.
                - None - Fit will run without any optimizer.
        """
        optims = self.optims

        if optims is None:
            console.log(
                f"[yellow]No optimizers have been defined! Consider subclassing "
                f"this function to manually define the optimizers."
            )
            return None
        if isinstance(optims, dict):
            optims = [optims]
        assert_list_of(optims, dict)
      
        for optim in optims:
            # Define optimizer measurement
            optimizer = optim.get("optimizer", None)
            if optimizer is None:
                raise ValueError(f"`optimizer` must be defined.")
            if isinstance(optimizer, dict):
                optimizer = OPTIMIZERS.build_from_dict(net=self, cfg=optimizer)
            optim["optimizer"] = optimizer

            # Define learning rate scheduler
            lr_scheduler = optim.get("lr_scheduler", None)
            if "lr_scheduler" in optim and lr_scheduler is None:
                optim.pop("lr_scheduler")
            elif lr_scheduler is not None:
                scheduler = lr_scheduler.get("scheduler", None)
                if scheduler is None:
                    raise ValueError(f"`scheduler` must be defined.")
                if isinstance(scheduler, dict):
                    scheduler = SCHEDULERS.build_from_dict(
                        optimizer = optim["optimizer"],
                        cfg       = scheduler
                    )
                lr_scheduler["scheduler"] = scheduler
            
            # Define optimizer frequency
            frequency = optim.get("frequency", None)
            if "frequency" in optim and frequency is None:
                optim.pop("frequency")
        
        # Re-assign optims
        self.optims = optims
        return self.optims
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor | None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor | None): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred     = self.forward(input=input, *args, **kwargs)
        features = None
        if isinstance(pred, (list, tuple)):
            features = pred[0:-1]
            pred     = pred[-1]
        loss = self.loss(pred, target) if self.loss else None
        return pred, loss
    
    @abstractmethod
    def forward(
        self,
        input    : Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            augment (bool): Perform test-time augmentation. Defaults to False.
            profile (bool): Measure processing time. Defaults to False.
            out_index (int): Return specific layer's output from `out_index`.
                Defaults to -1 means the last layer.
            
        Returns:
            Predictions.
        """
        pass
    
    def forward_once(
        self,
        input    : Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass once. Implement the logic for a single forward pass.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            profile (bool): Measure processing time. Defaults to False.
            out_index (int): Return specific layer's output from `out_index`.
                Defaults to -1 means the last layer.
                
        Returns:
            Predictions.
        """
        x     = input
        y, dt = [], []
        for m in self.model:
            
            # console.log(f"{m.i}")
            
            if m.f != -1:  # Get features from previous layer
                if isinstance(m.f, int):
                    x = y[m.f]  # From directly previous layer
                else:
                    x = [x if j == -1 else y[j] for j in m.f]  # From earlier layers
            
            x = m(x)  # pass features through current layer
            y.append(x if m.i in self.save else None)

        if out_index > -1 and out_index in self.save:
            output = y[out_index]
        else:
            output = x
        return output
    
    def on_fit_start(self):
        """
        Called at the very beginning of fit.
        """
        create_dirs(
            paths = [
                self.root,
                self.weights_dir,
                self.debug_dir
            ]
        )
    
    def on_train_epoch_start(self):
        """
        Called in the training loop at the very beginning of the epoch.
        """
        self.epoch_step = 0
    
    def training_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Training step.

        Args:
            batch (Any): Batch of inputs. It can be a tuple of
                (input, target, extra).
            batch_idx (int): Batch index.

        Returns:
            Outputs:
                - A single loss tensor.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def training_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Use this when training with dp or ddp2 because training_step() will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.
        
        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each device
        input  = outputs["input"]   # images from each device
        target = outputs["target"]  # ground-truths from each device
        pred   = outputs["pred"]    # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
        
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/train_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
       
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(
                    f"checkpoint/{metric.name}/train_step", value, True
                )
                # self.tb_log(f"{metric.name}/train_step", value, "step")
        
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def training_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/train_epoch", loss)
        self.tb_log_scalar(f"loss/train_epoch", loss, "epoch")
        
        # Metrics
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/train_epoch", value)
                self.tb_log_scalar(f"{metric.name}/train_epoch", value, "epoch")
                metric.reset()
    
    def on_validation_epoch_start(self):
        """
        Called in the validation loop at the very beginning of the epoch.
        """
        self.epoch_step = 0
    
    def validation_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Validation step.

        Args:
            batch (Any): Batch of inputs. It can be a tuple of
                (input, target, extra).
            batch_idx (int): Batch index.

        Returns:
            Outputs:
                - A single loss image.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def validation_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Use this when validating with dp or ddp2 because `validation_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each device
        input  = outputs["input"]   # images from each device
        target = outputs["target"]  # ground-truths from each device
        pred   = outputs["pred"]    # predictions from each device
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
            
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug \
            and epoch % self.debug.every_n_epochs == 0 \
            and self.epoch_step < self.debug.max_n:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    filepath = self.debug_image_filepath,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
            
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/val_step", loss)
        # self.tb_log(f"{loss_tag}", loss, "step")
        
        # Metrics
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_step", value)
                # self.tb_log(f"{metric.name}/val_step", value, "step")
            
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def validation_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/val_epoch", loss)
        self.tb_log_scalar(f"loss/val_epoch", loss, "epoch")
        
        # Metrics
        if self.val_metrics:
            for i, metric in enumerate(self.val_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/val_epoch", value)
                self.tb_log_scalar(f"{metric.name}/val_epoch", value, "epoch")
                metric.reset()
    
    def on_test_start(self) -> None:
        """
        Called at the very beginning of testing.
        """
        create_dirs(
            paths=[
                self.root,
                self.weights_dir,
                self.debug_dir
            ]
        )
    
    def on_test_epoch_start(self):
        """
        Called in the test loop at the very beginning of the epoch.
        """
        self.epoch_step = 0
    
    def test_step(
        self,
        batch    : Any,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Test step.

        Args:
            batch (Any): Batch of inputs. It can be a tuple of
                (input, target, extra).
            batch_idx (int): Batch index.

        Returns:
            Outputs:
                - A single loss image.
                - A dictionary with the first key must be the loss.
                - None, training will skip to the next batch.
        """
        input, target, extra = batch[0], batch[1], batch[2:]
        pred, loss = self.forward_loss(
            input  = input,
            target = target,
            *args, **kwargs
        )
        return {
            "loss"  : loss,
            "input" : input,
            "target": target,
            "pred"  : pred
        }
    
    def test_step_end(
        self,
        outputs: StepOutput | None,
        *args, **kwargs
    ) -> StepOutput | None:
        """
        Use this when testing with dp or ddp2 because `test_step` will
        operate on only part of the batch. However, this is still optional and
        only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be
            called so that you don't have to change your code.
        """
        if not isinstance(outputs, dict):
            return None
        
        # Gather results
        # For DDP strategy, gather outputs from multiple devices.
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        loss   = outputs["loss"]    # losses from each GPU
        input  = outputs["input"]   # images from each GPU
        target = outputs["target"]  # ground-truths from each GPU
        pred   = outputs["pred"]    # predictions from each GPU
        
        # Tensors
        if self.trainer.num_devices > 1:
            input  = input.flatten(start_dim=0,  end_dim=1)
            target = target.flatten(start_dim=0, end_dim=1)
            pred   = pred.flatten(start_dim=0,   end_dim=1)
        
        # Debugging
        epoch = self.current_epoch + 1
        if self.debug and \
            epoch % self.debug.every_n_epochs == 0 and \
            self.epoch_step < self.debug.max_n:
            if self.trainer.is_global_zero:
                self.show_results(
                    input    = input,
                    target   = target,
                    pred     = pred,
                    filepath = self.debug_image_filepath,
                    **self.debug | {
                        "max_n": input[0],
                        "nrow" : input[0],
                    }
                )
                
        # Loss
        loss = loss.mean() if loss is not None else None
        self.ckpt_log_scalar(f"checkpoint/loss/test_step", loss)
        # self.tb_log(f"loss/test_step", loss, "step")
        
        # Metrics
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric(pred, target)
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_step", value)
                # self.tb_log(f"{metric.name}/test_step", value, "step")
        
        self.epoch_step += 1
        return {
            "loss"  : loss,
            # "input" : input,
            # "target": target,
            # "pred"  : pred,
        }
    
    def test_epoch_end(self, outputs: EpochOutput):
        # Loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.ckpt_log_scalar(f"checkpoint/loss/test_epoch", loss)
        self.tb_log_scalar(f"loss/test_epoch", loss, "epoch")

        # Metrics
        if self.test_metrics:
            for i, metric in enumerate(self.test_metrics):
                value = metric.compute()
                self.ckpt_log_scalar(f"checkpoint/{metric.name}/test_epoch", value)
                self.tb_log_scalar(f"{metric.name}/test_epoch", value, "epoch")
                metric.reset()
    
    def export_to_onnx(
        self,
        input_dims   : Ints  | None = None,
        filepath     : Path_ | None = None,
        export_params: bool         = True
    ):
        """
        Export the model to `onnx` format.

        Args:
            input_dims (Ints | None): Input dimensions. Defaults to None.
            filepath (Path_ | None): Path to save the model. If None or empty,
                then save to root. Defaults to None.
            export_params (bool): Should export parameters also?
                Defaults to True.
        """
        # Check filepath
        if filepath in [None, ""]:
            filepath = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(filepath):
            filepath = Path(str(filepath) + ".onnx")
        
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"`input_dims` must be defined.")
        
        self.to_onnx(
            file_path     = filepath,
            input_sample  = input_sample,
            export_params = export_params
        )
        
    def export_to_torchscript(
        self,
        input_dims: Ints  | None = None,
        filepath  : Path_ | None = None,
        method    : str          = "script"
    ):
        """Export the model to `TorchScript` format.

        Args:
            input_dims (Ints | None): Input dimensions. Defaults to None.
            filepath (Path_ | None): Path to save the model. If None or empty,
                then save to root. Defaults to None.
            method (str):
                Whether to use TorchScript's `script` or `trace` method.
                Default: `script`
        """
        # Check filepath
        if filepath in [None, ""]:
            filepath = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(filepath):
            filepath = Path(str(filepath) + ".pt")
            
        if input_dims is not None:
            input_sample = torch.randn(input_dims)
        else:
            raise ValueError(f"`input_dims` must be defined.")
        
        script = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, filepath)
    
    @abstractmethod
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : Path_  | None = None,
        image_quality: int           = 95,
        max_n        : int | None    = 8,
        nrow         : int | None    = 8,
        wait_time    : float         = 0.01,
        save         : bool          = False,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (Path_ | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n items if `input` has a batch size
                of more than `max_n` items. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            save (bool): Save debug image. Defaults to False.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        pass
    
    def print_info(self):
        if self.verbose and self.model is not None:
            console.log(f"[red]{self.fullname}")
            print_table(self.info)
            console.log(f"Save indexes: {self.save}")
    
    def tb_log_scalar(
        self,
        tag : str,
        data: Any | None,
        step: str | int = "step"
    ):
        """
        Log scalar values using tensorboard.
        """
        if data is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar(tag, data, step)
    
    def tb_log_class_metrics(
        self,
        tag : str,
        data: Any | None,
        step: str | int = "step"
    ):
        """
        Log class metrics using tensorboard.
        """
        if data is None:
            return
        if self.classlabels is None:
            return
        if isinstance(step, str):
            step = self.current_epoch if step == "epoch" else self.global_step
        if self.trainer.is_global_zero:
            for n, a in zip(self.classlabels.names(), data):
                n = f"{tag}/{n}"
                self.logger.experiment.add_scalar(n, a, step)
    
    def ckpt_log_scalar(
        self,
        tag     : str,
        data    : Any | None,
        prog_bar: bool = False
    ):
        """
        Log for model checkpointing.
        """
        if data is None:
            return
        if self.trainer.is_global_zero:
            self.log(
                name           = tag,
                value          = data,
                prog_bar       = prog_bar,
                sync_dist      = True,
                rank_zero_only = True
            )


# H2: - Classification ---------------------------------------------------------

class ImageClassificationModel(BaseModel, metaclass=ABCMeta):
    
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        """
        Build the model. You have 2 options to build a model: (1) define each
        layer manually, or (2) build model automatically from a config
        dictionary.
        
        We inherit the same idea of model parsing in YOLOv5.
        
        Either way each layer should have the following attributes:
            - i (int): index of the layer.
            - f (int | list[int]): from, i.e., the current layer receive output
              from the f-th layer. For example: -1 means from previous layer;
              -2 means from 2 previous layers; [99, 101] means from the 99th
              and 101st layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
              t = str(m)[8:-2].replace("__main__.", "")
            - np (int): number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d (dict | None): Model definition dictionary. Default to None means
                building the model manually.
            ch (list[int] | None): The first layer's input channels. If given,
                it will be used to further calculate the next layer's input
                channels. Defaults to None means defines each layer in_ and
                out_channels manually.
        
        Returns:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info (dict) for debugging.
        """
        return parse_model(d=d, ch=ch)
    
    def forward(
        self,
        input  : Tensor,
        augment: bool = False,
        profile: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            augment (bool): Perform test-time augmentation. Defaults to False.
            profile (bool): Measure processing time. Defaults to False.
            
        Returns:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
        else:
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
    
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : Path_  | None = None,
        image_quality: int           = 95,
        max_n        : int    | None = 8,
        nrow         : int    | None = 8,
        wait_time    : float         = 0.01,
        save         : bool          = False,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (Path_ | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n images if `image` has a batch size
                of more than `max_n` images. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            save (bool): Save debug image. Defaults to False.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        from one.plot import imshow_classification
        
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        imshow_classification(
            winname   = self.fullname,  # self.phase.value,
            image     = input,
            pred      = pred,
            target    = target,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = max_n,
            nrow      = nrow,
            wait_time = wait_time,
        )


# H2: - Enhancement ------------------------------------------------------------

class ImageEnhancementModel(BaseModel, metaclass=ABCMeta):
    
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        """
        Build the model. You have 2 options to build a model: (1) define each
        layer manually, or (2) build model automatically from a config
        dictionary.
        
        We inherit the same idea of model parsing in YOLOv5.
        
        Either way each layer should have the following attributes:
            - i (int): index of the layer.
            - f (int | list[int]): from, i.e., the current layer receive output
              from the f-th layer. For example: -1 means from previous layer;
              -2 means from 2 previous layers; [99, 101] means from the 99th
              and 101st layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
              t = str(m)[8:-2].replace("__main__.", "")
            - np (int): number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d (dict | None): Model definition dictionary. Default to None means
                building the model manually.
            ch (list[int] | None): The first layer's input channels. If given,
                it will be used to further calculate the next layer's input
                channels. Defaults to None means defines each layer in_ and
                out_channels manually.
        
        Returns:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info (dict) for debugging.
        """
        return parse_model(d=d, ch=ch)
    
    def forward(
        self,
        input  : Tensor,
        augment: bool = False,
        profile: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            augment (bool): Perform test-time augmentation. Defaults to False.
            profile (bool): Measure processing time. Defaults to False.
            
        Returns:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
        else:
            return self.forward_once(
                input=input, profile=profile, *args, **kwargs
            )
    
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : Path_  | None = None,
        image_quality: int           = 95,
        max_n        : int    | None = 8,
        nrow         : int    | None = 8,
        wait_time    : float         = 0.01,
        save         : bool          = False,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (Path_ | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n images if `image` has a batch size
                of more than `max_n` images. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            save (bool): Save debug image. Defaults to False.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        from one.plot import imshow_enhancement

        result = {}
        if input is not None:
            result["input"]  = input
        if target is not None:
            result["target"] = target
        if pred is not None:
            if isinstance(pred, (tuple, list)):
                result["pred"] = pred[-1]
            else:
                result["pred"] = pred
        
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        imshow_enhancement(
            winname   = self.fullname,  # self.phase.value,
            image     = result,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = max_n,
            nrow      = nrow,
            wait_time = wait_time,
        )

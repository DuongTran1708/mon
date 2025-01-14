#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from onedetection.models.yolov5_v6_1.train import parse_opt
from onedetection.models.yolov5_v6_1.train import train
from onedetection.models.yolov5_v6_1.utils.callbacks import Callbacks
from onedetection.models.yolov5_v6_1.utils.general import increment_path
from onedetection.models.yolov5_v6_1.utils.torch_utils import select_device


# H1: - Functional

def sweep():
    wandb.init()
    # Get hyp dict from sweep agent
    hyp_dict = vars(wandb.config).get("_items")

    # Workaround: get necessary opt args
    opt            = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir   = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs     = hyp_dict.get("epochs")
    opt.nosave     = True
    opt.data       = hyp_dict.get("data")
    opt.weights    = str(opt.weights)
    opt.cfg        = str(opt.cfg)
    opt.data       = str(opt.data)
    opt.hyp        = str(opt.hyp)
    opt.project    = str(opt.project)
    device         = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(hyp_dict, opt, device, callbacks=Callbacks())


# H1: - Main

if __name__ == "__main__":
    sweep()

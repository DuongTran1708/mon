#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""File for accessing YOLOv5 models via PyTorch Hub https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
"""

from __future__ import annotations

from pathlib import Path

import torch

from one.vision.detection.yolov5_v4.models.yolo import Model
from one.vision.detection.yolov5_v4.utils.general import check_requirements
from one.vision.detection.yolov5_v4.utils.general import set_logging
from one.vision.detection.yolov5_v4.utils.google_utils import attempt_download
from one.vision.detection.yolov5_v4.utils.torch_utils import select_device

dependencies = ["torch", "yaml"]
check_requirements(Path(__file__).parent / "requirements.txt", exclude=("pycocotools", "thop"))
set_logging()


def create(name, pretrained, channels, classes, autoshape):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. "yolov5s"
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    config = Path(__file__).parent / "models" / f"{name}.yaml"  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = f"{name}.pt"  # weights filename
            attempt_download(fname)  # download if not found locally
            ckpt = torch.load(fname, map_location=torch.device("cpu"))  # load
            msd  = model.state_dict()   # model state_dict
            csd  = ckpt["model"].float().state_dict()  # weights state_dict as FP32
            csd  = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
            model.load_state_dict(csd, strict=False)  # load
            if len(ckpt["model"].names) == classes:
                model.names = ckpt["model"].names  # set class names attribute
            if autoshape:
                model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device("0" if torch.cuda.is_available() else "cpu")  # default to GPU if available
        return model.to(device)

    except Exception as e:
        help_url = "https://github.com/ultralytics/yolov5/issues/36"
        s        = "Cache maybe be out of date, try force_reload=True. See %s for help." % help_url
        raise Exception(s) from e


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True):
    """YOLOv5-small model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create("yolov5s", pretrained, channels, classes, autoshape)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True):
    """YOLOv5-medium model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create("yolov5m", pretrained, channels, classes, autoshape)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True):
    """YOLOv5-large model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create("yolov5l", pretrained, channels, classes, autoshape)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True):
    """YOLOv5-xlarge model from https://github.com/ultralytics/yolov5

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create("yolov5x", pretrained, channels, classes, autoshape)


def custom(path_or_model="path/to/model.pt", autoshape=True):
    """YOLOv5-custom model from https://github.com/ultralytics/yolov5

    Arguments (3 options):
        path_or_model (str): "path/to/model.pt"
        path_or_model (dict): torch.load("path/to/model.pt")
        path_or_model (nn.Module): torch.load("path/to/model.pt")["model"]

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load weights
    if isinstance(model, dict):
        model = model["ema" if model.get("ema") else "model"]  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device("0" if torch.cuda.is_available() else "cpu")  # default to GPU if available
    return hub_model.to(device)


if __name__ == "__main__":
    model = create(name="yolov5s", pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example
    # model = custom(path_or_model="path/to/model.pt")  # custom example

    # Verify inference
    import numpy as np
    from PIL import Image

    imgs = [
        Image.open("data/images/bus.jpg"),  # PIL
        "data/images/zidane.jpg",  # filename
        "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",  # URI
        np.zeros((640, 480, 3))
    ]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()

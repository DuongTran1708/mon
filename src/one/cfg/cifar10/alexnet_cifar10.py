#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-DCE trained on LIME dataset.
"""

from __future__ import annotations

from one.cfg import default
from one.constants import RUNS_DIR
from one.constants import VISION_BACKEND
from one.vision.transformation import Resize


# H1: - Basic ------------------------------------------------------------------

model_name = "alexnet"
model_cfg  = "alexnet.yaml"
data_name  = "cifar10"
fullname   = f"{model_name}-{data_name}"
root       = RUNS_DIR / "train"
project    = None
shape      = [3, 64, 64]


# H1: - Data -------------------------------------------------------------------

data = {
    "name": data_name,
        # Dataset's name.
    "shape": shape,
        # Image shape as [C, H, W], [H, W], or [S, S].
	"num_classes": 10,
		# Number of classes in the dataset.
    "transform": [
        Resize(size=shape)
    ],
        # Functions/transforms that takes in an input sample and returns a
        # transformed version.
    "target_transform": None,
        # Functions/transforms that takes in a target and returns a
        # transformed version.
    "transforms": None,
        # Functions/transforms that takes in an input and a target and
        # returns the transformed versions of both.
    "cache_data": False,
        # If True, cache data to disk for faster loading next time.
        # Defaults to False.
    "cache_images": False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Defaults to False.
    "backend": VISION_BACKEND,
        # Vision backend to process image. Defaults to VISION_BACKEND.
    "batch_size": 64,
        # Number of samples in one forward & backward pass. Defaults to 1.
    "devices" : 0,
        # The devices to use. Defaults to 0.
    "shuffle": True,
        # If True, reshuffle the data at every training epoch. Defaults to True.
    "verbose": True,
        # Verbosity. Defaults to True.
}


# H1: - Model ------------------------------------------------------------------

model = {
	"cfg": model_cfg,
        # Model's layers configuration. It can be an external .yaml path or a
        # dictionary. Defaults to None means you should define each layer
        # manually in `self.parse_model()` method.
    "root": root,
        # The root directory of the model. Defaults to RUNS_DIR.
    "project": project,
		# Project name. Defaults to None.
    "name": model_name,
        # Model's name. In case None is given, it will be
        # `self.__class__.__name__`. Defaults to None.
    "fullname": fullname,
        # Model's fullname in the following format: {name}-{data_name}-{postfix}.
        # In case None is given, it will be `self.basename`. Defaults to None.
    "channels": 3,
        # Input channel. Defaults to 3.
    "num_classes": data["num_classes"],
        # Number of classes for classification or detection tasks.
        # Defaults to None.
    "classlabels": None,
        # ClassLabels object that contains all labels in the dataset.
        # Defaults to None.
    "phase": "training",
        # Model's running phase. Defaults to training.
    "pretrained": None,
        # Initialize weights from pretrained.
    "loss": {"name": "cross_entropy_loss"},
        # Loss function for training model. Defaults to None.
    "metrics": [
		{"name": "accuracy", "num_classes": data["num_classes"]}
	],
        # Metric(s) for validating and testing model. Defaults to None.
    "optimizers": [
        {
            "optimizer": {
				"name": "adam",
				"lr": 0.0001,
				"weight_decay": 0.0001,
				"betas": [0.9, 0.99],
			},
			"lr_scheduler": None,
            "frequency": None,
        }
    ],
        # Optimizer(s) for training model. Defaults to None.
    "debug": default.debug,
        # Debug configs. Defaults to None.
	"verbose": True,
		# Verbosity. Defaults to True.
}


# H1: - Training ---------------------------------------------------------------

callbacks = [
    default.model_checkpoint | {
	    "monitor": "checkpoint/accuracy/val_epoch",
		    # Quantity to monitor. Defaults to None which saves a checkpoint
	        # only for the last epoch.
		"mode": "max",
			# One of {min, max}. If `save_top_k != 0`, the decision to
	        # overwrite the current save file is made based on either the
	        # maximization or the minimization of the monitored quantity.
	        # For `val_acc`, this should be `max`, for `val_loss` this should
	        # be `min`, etc.
	},
	default.learning_rate_monitor,
	default.rich_model_summary,
	default.rich_progress_bar,
]

logger = {
	"tensorboard": default.tensorboard,
}

trainer = default.trainer | {
	"default_root_dir": root,
        # Default path for logs and weights when no logger/ckpt_callback passed.
        # Can be remote file paths such as `s3://mybucket/path` or
        # 'hdfs://path/'. Defaults to os.getcwd().
}

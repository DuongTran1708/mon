#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5 object_detectors.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn

from core.io.filedir import is_torch_saved_file
from core.utils.bbox import scale_bbox_xyxy



from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from configuration import models_zoo_dir
from detectors.base import BaseDetector

# NOTE: add model YOLOv5 source to here
sys.path.append('src/detectors/ultralytics')

from detectors.ultralytics.ultralytics import YOLO

__all__ = [
	"yolov8_trafficsafety"
]


# MARK: - YOLOv5

# noinspection PyShadowingBuiltins
@DETECTORS.register(name="yolov8_trafficsafety")
class YOLOv8(BaseDetector):
	"""YOLOv8 object detector."""

	# MARK: Magic Functions

	def __init__(self, name: str = "yolov5_mtmc", *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: Create model
		path = self.weights
		if not is_torch_saved_file(path):
			path, _ = os.path.splitext(path)
			path    = os.path.join(models_zoo_dir, f"{path}.pt")
		assert is_torch_saved_file(path), f"Not a weights file: {path}"

		# NOTE: load model
		self.model  = YOLO(path)

		# NOTE: Eval
		self.model.to(device=self.device)
		self.model.eval()


	# MARK: Detection

	def forward(self, input: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			input (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		pred = self.model(input, augment=False)[0]

		return pred

	def postprocess(
			self,
			indexes: np.ndarray,
			images : np.ndarray,
			input  : Tensor,
			pred   : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].
			input (Tensor):
				Input image of shape [B, C, H, W].
			pred (Tensor):
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Resize
		if self.resize_original:
			for pre_ in pred:
				bboxes_xyxy = pre_[:, :4].cpu().detach()
				pre_[:, :4] = scale_bbox_xyxy(
					xyxy       = bboxes_xyxy,
					image_size = input.shape[2 :],
					new_size   = images.shape[1: 3]
				).round()

		# NOTE: Create Detection objects
		instances = []
		for idx, pre_ in enumerate(pred):
			# SUGAR: add more for the same as original code
			pre_[:, :4] = scale_coords(input.shape[2:], pre_[:, :4], images.shape[1: 3]).round()

			inst = []
			for *xyxy, conf, cls in pre_:
				bbox_xyxy = np.array([xyxy[0].item(), xyxy[1].item(),
									  xyxy[2].item(), xyxy[3].item()], np.int32)

				# SUGAR:
				if bbox_xyxy[0] < 0 or bbox_xyxy[1] < 0 or  \
						bbox_xyxy[2] > images.shape[2] - 1 or bbox_xyxy[3] > im0.shape[0] - 1:
					continue

				confident  = float(conf)
				class_id   = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = indexes[0] + idx,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label
					)
				)
			instances.append(inst)
		return instances


# MARK: - Utils

def adjust_state_dict(state_dict: OrderedDict) -> OrderedDict:
	od = OrderedDict()
	for key, value in state_dict.items():
		new_key     = key.replace("module.", "")
		od[new_key] = value
	return od

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
	# Rescale coords (xyxy) from img1_shape to img0_shape
	if ratio_pad is None:  # calculate from img0_shape
		gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
		pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad = ratio_pad[1]

	coords[:, [0, 2]] -= pad[0]  # x padding
	coords[:, [1, 3]] -= pad[1]  # y padding
	coords[:, :4] /= gain
	# clip_coords(coords, img0_shape)
	return coords

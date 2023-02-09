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

from core.io.filedir import is_torch_saved_file
from core.utils.bbox import scale_bbox_xyxy
from core.utils.geometric_transformation import padded_resize
from core.utils.image import to_tensor
from core.utils.image import is_channel_first
from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from configuration import models_zoo_dir
from detectors.base import BaseDetector

# NOTE: add model YOLOv5 source to here
sys.path.append('src/detectors/ultralytics')

from detectors.ultralytics.ultralytics import YOLO

__all__ = [
	"YOLOv8"
]


# MARK: - YOLOv5

@DETECTORS.register(name="yolov8_trafficsafety")
class YOLOv8(BaseDetector):
	"""YOLOv8 object detector."""

	# MARK: Magic Functions

	def __init__(self, name: str = "yolov8_trafficsafety", *args, **kwargs):
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
		self.model = YOLO(path)
		self.model.to(device=self.device)

		# DEBUG:
		# print("*************")
		# print(dir(self.model))
		# print(self.model.overrides)
		# print("*************")

	# MARK: Detection

	def detect(self, indexes: np.ndarray, images: np.ndarray) -> list:
		"""Detect objects in the images.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			print("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input = self.preprocess(images=images)
		# NOTE: Forward
		pred  = self.forward(input)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input=input, pred=pred
		)
		# NOTE: Suppression
		instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: np.ndarray) -> Tensor:
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		input = images
		# if self.shape:
		# 	input = padded_resize(input, self.shape, stride=self.stride)
		# 	self.resize_original = True
		# #input = [F.to_tensor(i) for i in input]
		# #input = torch.stack(input)
		# input = to_tensor(input, normalize=True)
		# input = input.to(self.device)
		return input

	def forward(self, input: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			input (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			pred (Tensor):
				Predictions.
		"""
		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			img_size = self.shape[2]
		else:
			img_size = self.shape[0]

		# DEBUG:
		# print(dir(self.model))
		# print(self.model.overrides)
		# sys.exit()

		pred = self.model(
			input,
			imgsz   = img_size,
			conf    = self.min_confidence,
			iou     = self.nms_max_overlap,
			classes = self.allowed_ids,
			augment = True,
			verbose = False,
		)
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
		# NOTE: Create Detection objects
		instances = []
		# DEBUG:
		# print("******")
		# for result in pred:
		# 	# detection
		# 	result.boxes.xyxy  # box with xyxy format, (N, 4)
		# 	result.boxes.xywh  # box with xywh format, (N, 4)
		# 	result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
		# 	result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
		# 	result.boxes.conf  # confidence score, (N, 1)
		# 	result.boxes.cls  # cls, (N, 1)
		# print("******")

		for idx, result in enumerate(pred):
			inst = []
			xyxys = result.boxes.xyxy.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			clses = result.boxes.cls.cpu().numpy()
			for bbox_xyxy, conf, cls in zip(xyxys, confs, clses):
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

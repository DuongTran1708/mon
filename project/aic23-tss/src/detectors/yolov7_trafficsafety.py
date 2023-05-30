# ==================================================================== #

# ==================================================================== #
import os
import sys
from collections import OrderedDict
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor

from core.factory.builder import DETECTORS
from core.io.filedir import is_yaml_file, is_torch_saved_file
from core.objects.instance import Instance
from core.utils.bbox import scale_bbox_xyxy
from core.utils.image import is_channel_first
from detectors import BaseDetector


# NOTE: add model YOLOv5 source to here
sys.path.append('src/detectors/yolov7')

from detectors.yolov7.utils.general import non_max_suppression, check_img_size
from detectors.yolov7.models.experimental import attempt_load
from detectors.yolov7.utils.datasets import letterbox

__all__ = [
	"YOLOv7"
]

# MARK: - YOLOv7

@DETECTORS.register(name="yolov7_trafficsafety")
class YOLOv7(BaseDetector):
	"""YOLOv7 detector model.
	"""

	# MARK: Magic Functions

	def __init__(self,  name: str = "yolov7_trafficsafety", *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		# Load model
		self.load_model()

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[2]
		else:
			self.img_size = self.shape[0]
		self.stride = int(self.model.stride.max())  # model stride
		self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size

		# Move to device
		self.model.to(self.device).eval()
		self.should_resize = True  # Rescale image from model layer (768) to original image size

	# MARK: Detection

	def load_model(self):
		"""Pipeline to load the model.
		"""
		current_dir = os.path.dirname(os.path.abspath(__file__))  # "...detector/yolov7"

		# NOTE: Simple check
		if self.weights is None or self.weights == "":
			print("No weights file has been defined!")
			raise ValueError

		# NOTE: Get path to weight file
		# self.weights = os.path.join(current_dir, "weights", self.weights)
		# if not is_torch_saved_file(self.weights):
		# 	raise FileNotFoundError

		# NOTE: Get path to model variant's config
		model_config = os.path.join(current_dir, "yolov7/cfg/deploy", f"{self.variant}.yaml")
		if not is_yaml_file(model_config):
			raise FileNotFoundError

		# NOTE: Define model and load weights
		# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
		self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model

	# MARK: Instance

	def detect(
			self,
			indexes: np.ndarray,
			images: np.ndarray
	) -> list:
		"""Detect objects in the images.

		Args:
			indexes ():
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
		inputs = self.preprocess(images=images)
		# NOTE: Forward
		pred  = self.forward(inputs=inputs)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes = indexes,
			images  = images,
			inputs  = inputs,
			pred    = pred
		)
		# NOTE: Suppression
		instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: Union[Tensor, np.ndarray]) -> Tensor:
		"""Prepare the model's input for the forward pass.

		Convert to Tensor, resize, change dims to [CHW] and normalize.

		Override this function if you want a custom preparation pipeline.

		Args:
			images (Tensor or np.array):
				The list of np.array images of shape [BHWC]. If the images is of Tensor type, we assume it has already been normalized.

		Returns:
			inputs (Tensor):
				The prepared images [BCHW] with B=1.
		"""
		inputs = None
		if isinstance(images, np.ndarray) or isinstance(images, list):
			# if images.shape[2] != self.dims[2]:
			# 	images             = ops.padded_resize_image(images=images, size=self.dims[1:3])
			# 	self.should_resize = True
			# images = np.array(images)
			inputs = []
			for image in images:
				input = letterbox(image, self.img_size, stride=self.stride)[0]
				inputs.append(F.to_tensor(pic=input))
			inputs = torch.stack(inputs)
		if torch.is_tensor(inputs) and len(inputs.size()) == 3:
			inputs = inputs.unsqueeze(0)

		return inputs.to(self.device)

	def forward(
			self,
			inputs: Union[Tensor, np.ndarray]
	) -> Union[List[Instance], List[List[Instance]]]:
		"""Define the forward pass logic of the ``model``.

		Args:
			frame_indexes (List[int]):
				The list of image indexes in the video.
			inputs (Tensor or np.array):
				The list of np.array images of shape [BHWC]. If the images is of Tensor type, we assume it has already been normalized.

		Returns:
			batch_detections (list):
				A list of ``Instance``.
				A list of ``Instance`` in batch.
		"""
		# NOTE: Forward input
		batch_predictions = self.model(inputs, augment=False)[0]
		batch_predictions = non_max_suppression(
			batch_predictions,
			self.min_confidence,
			self.nms_max_overlap
		)

		return batch_predictions

	def postprocess(
			self,
			indexes: np.ndarray,
			images : np.ndarray,
			inputs : Tensor,
			pred   : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].
			inputs (Tensor):
				Input image of shape [B, C, H, W].
			pred (Tensor):
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Rescale image from model layer (768) to original image size
		if self.should_resize:
			for predictions in pred:
				det_bbox_xyxy = predictions[:, :4].cpu().detach()
				predictions[:, :4] = scale_bbox_xyxy(
					xyxy       = det_bbox_xyxy,
					image_size = inputs.shape[2:],
					new_size   = images[0].shape[:2]
				).round()

		# DEBUG:
		# print("**************")
		# print(inputs.shape)
		# print(images[0].shape)
		# print("**************")

		# NOTE: Create Instance objects
		batch_detections = []
		for idx, predictions in enumerate(pred):
			inst = []
			for *xyxy, conf, cls in predictions:
				bbox_xyxy = np.array([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()], np.int32)
				confident = float(conf)
				class_id  = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)

				# DEBUG:
				# print("*******")
				# print(inputs.shape)
				# print(images[0].shape)
				# print(bbox_xyxy)
				# print(images.shape)
				# print(inputs.shape)
				# print("*******")

				inst.append(
					Instance(
						frame_index = indexes[0] + idx,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label
					)
				)
			batch_detections.append(inst)

		return batch_detections

# MARK: - Utils


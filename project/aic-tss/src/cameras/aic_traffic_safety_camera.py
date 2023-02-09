#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import uuid
import glob
from timeit import default_timer as timer
from typing import Union

import pickle
import cv2
import torch
import numpy as np
from tqdm import tqdm

from core.data.class_label import ClassLabels
from core.io.filedir import is_basename
from core.io.filedir import is_json_file
from core.io.filedir import is_stem
from core.utils.bbox import bbox_xyxy_to_cxcywh_norm
from core.utils.rich import console
from core.utils.constants import AppleRGB
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.video import is_video_file
from core.io.video import VideoLoader
from core.factory.builder import CAMERAS
from core.factory.builder import DETECTORS
from detectors.base import BaseDetector
from configuration import (
	data_dir,
	config_dir
)
from cameras.base import BaseCamera


# MARK: - AICVehicleCountingCamera

# noinspection PyAttributeOutsideInit
@CAMERAS.register(name="aic_traffic_safety_camera")
class AICTrafficSafetyCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			class_labels : Union[ClassLabels,       dict],
			detector     : Union[BaseDetector,      dict],
			data_loader  : dict,
			data_writer  : Union[FrameWriter,       dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			*args, **kwargs
	):
		"""

		Args:
			dataset (str):
				Dataset name. It is also the name of the directory inside
				`data_dir`.
			subset (str):
				Subset name. One of: [`dataset_a`, `dataset_b`].
			name (str):
				Camera name. It is also the name of the camera's config
				files.
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		self.process      = process
		self.verbose      = verbose

		self.data_cfg        = data
		self.detector_cfg    = detector
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer

		self.init_dirs()
		self.init_class_labels(class_labels=class_labels)
		self.init_data_writer(data_writer_cfg=self.data_writer_cfg)

		if self.process["function_dets"]:
			self.init_detector(detector=detector)

		self.start_time = None
		self.pbar       = None

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.

		Returns:

		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(self.root_dir, self.data_writer_cfg["dst"])
		self.video_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])

	def init_class_labels(self, class_labels: Union[ClassLabels, dict]):
		"""Initialize class_labels.

		Args:
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
		"""
		if isinstance(class_labels, ClassLabels):
			self.class_labels = class_labels
		elif isinstance(class_labels, dict):
			file = class_labels["file"]
			if is_json_file(file):
				self.class_labels = ClassLabels.create_from_file(file)
			elif is_basename(file):
				file              = os.path.join(self.root_dir, file)
				self.class_labels = ClassLabels.create_from_file(file)
		else:
			file              = os.path.join(self.root_dir, f"class_labels.json")
			self.class_labels = ClassLabels.create_from_file(file)
			print(f"Cannot initialize class_labels from {class_labels}. "
				  f"Attempt to load from {file}.")

	def init_detector(self, detector: Union[BaseDetector, dict]):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		console.log(f"Initiate Detector.")
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_data_loader(self, data_loader_cfg: dict):
		"""Initialize data loader.

		Args:
			data_loader_cfg (dict):
				Data loader object or a data loader's config dictionary.
		"""
		if self.process["function_dets"]:
			self.data_loader = sorted(glob.glob(os.path.join(self.video_dir, self.data_cfg["type"])))

	def check_and_create_folder(self, attr, data_writer_cfg: dict):
		"""CHeck and create the folder to store the result

		Args:
			attr (str):
				the type of function/saving/creating
			data_writer_cfg (dict):
				configuration of camera
		Returns:
			None
		"""
		path = os.path.join(self.outputs_dir, f"{data_writer_cfg[attr]}")
		if not os.path.isdir(path):
			os.makedirs(path)
		data_writer_cfg[attr] = path

	def init_data_writer(self, data_writer_cfg: dict):
		"""Initialize data writer.

		Args:
			data_writer_cfg (FrameWriter, dict):
				Data writer object or a data writer's config dictionary.
		"""
		# NOTE: save detections txt
		data_writer_cfg["dst_crop_pkl"] = f'{data_writer_cfg["dst_crop_pkl"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dst_crop_pkl", data_writer_cfg=data_writer_cfg)

	# MARK: Run

	def run_detection(self):
		"""Run detection model

		Returns:

		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.detector_cfg["batch_size"]
		self.init_data_loader(data_loader_cfg=self.data_loader_cfg)

		# NOTE: run detection
		pbar = tqdm(total=len(self.data_loader), desc="Detection process: ")
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for video_path in self.data_loader:
				# Init parameter
				basename       = os.path.basename(video_path)
				basename_noext = os.path.splitext(basename)[0]
				height_img    , width_img = None, None
				index_image    = 0
				out_dict       = []

				video_loader = VideoLoader(data=video_path, batch_size=self.data_loader_cfg["batch_size"])
				pbar_video = tqdm(total=len(video_loader), desc=f"Video {basename}:")

				# NOTE: run each video
				for images, indexes, _, _ in video_loader:
					# NOTE: pre process
					# if finish loading
					if len(indexes) == 0:
						break

					# get size of image
					if height_img is None:
						height_img, width_img, _ = images[0].shape

					# NOTE: Detect batch of instances
					batch_instances = self.detector.detect(
						indexes=indexes, images=images
					)

					# NOTE: Write the detection result
					for index_b, batch in enumerate(batch_instances):
						image_draw = images[index_b].copy()
						index_image       += 1
						name_index_image  = f"{index_image:06d}"

						for index_in, instance in enumerate(batch):
							name_index_in = f"{index_in:08d}"
							bbox_xyxy     = [int(i) for i in instance.bbox]

							# NOTE: crop the bounding box
							crop_image = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

							result_dict = {
								'video_name' : basename_noext,
								'frame_id'   : name_index_image,
								'crop_img'   : crop_image,
								'bbox'       : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
								'class_id'   : instance.class_label["train_id"],
								'conf'       : instance.confidence
							}
							out_dict.append(result_dict)

					pbar_video.update(len(indexes))

				# NOTE: save pickle
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer_cfg['dst_crop_pkl'], basename_noext)}.pkl", 'wb')
				)

				# Post process
				del video_loader
				pbar_video.close()
				pbar.update(1)
		pbar.close()

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: run detection
		if self.process["function_dets"]:
			self.run_detection()
			self.detector.clear_model_memory()
			self.detector = None

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		cv2.destroyAllWindows()
		self.stop_time = timer()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		elapsed_time = timer() - self.start_time
		if self.verbose:
			# cv2.imshow(self.name, result)
			cv2.waitKey(1)

	# MARK: Visualize

	def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.

		Returns:
			drawing (np.ndarray):
				Drawn canvas.
		"""
		return drawing

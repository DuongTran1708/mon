#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import uuid
from timeit import default_timer as timer
from typing import Union

import cv2
import torch
import numpy as np
from tqdm import tqdm

from torchkit.core.data import ClassLabels
from torchkit.core.file import is_basename
from torchkit.core.file import is_json_file
from torchkit.core.file import is_stem
from torchkit.core.type import is_list_of
from torchkit.core.vision import AppleRGB
from torchkit.core.vision import FrameLoader
from torchkit.core.vision import FrameWriter
from torchkit.core.vision import is_video_file
from projects.tss.src.builder import CAMERAS
from projects.tss.src.builder import DETECTORS
from projects.tss.src.detectors import BaseDetector
from projects.tss.src.io import AICCountingWriter
from projects.tss.src.trackers import BaseTracker
from projects.tss.utils import data_dir
from .base import BaseCamera


# MARK: - AICVehicleCountingCamera

# noinspection PyAttributeOutsideInit
@CAMERAS.register(name="aic_vehicle_counting_camera")
class AICVehicleCountingCamera(BaseCamera):
	"""AIC Counting Camera implements the functions for Multi-Class
	Multi-Movement Vehicle Counting (MMVC).

	Attributes:
		id_ (int, str):
			Camera's unique ID.
		dataset (str):
			Dataset name. It is also the name of the directory inside
			`data_dir`. Default: `None`.
		subset (str):
			Subset name. One of: [`dataset_a`, `dataset_b`].
		name (str):
			Camera name. It is also the name of the camera's config files.
			Default: `None`.
		class_labels (ClassLabels):
			Classlabels.
		rois (list[ROI]):
			List of ROIs.
		mois (list[MOI]):
			List of MOIs.
		detector (BaseDetector):
			Detector model.
		tracker (BaseTracker):
			Tracker object.
		moving_object_cfg (dict):
			Config dictionary of moving object.
		data_loader (FrameLoader):
			Data loader object.
		data_writer (FrameWriter):
			Data writer object.
		result_writer (AICCountingWriter):
			Result writer object.
		verbose (bool):
			Verbosity mode. Default: `False`.
		save_image (bool):
			Should save individual images? Default: `False`.
		save_video (bool):
			Should save video? Default: `False`.
		save_results (bool):
			Should save results? Default: `False`.
		root_dir (str):
			Root directory is the full path to the dataset.
		configs_dir (str):
			`configs` directory located inside the root directory.
		rmois_dir (str):
			`rmois` directory located inside the root directory.
		outputs_dir (str):
			`outputs` directory located inside the root directory.
		video_dir (str):
			`video` directory located inside the root directory.
		mos (list):
			List of current moving objects in the camera.
		start_time (float):
			Start timestamp.
		pbar (tqdm):
			Progress bar.
	"""

	# MARK: Magic Functions

	def __init__(
			self,
			dataset      : str,
			subset       : str,
			name         : str,
			class_labels : Union[ClassLabels,       dict],
			detector     : Union[BaseDetector,      dict],
			moving_object: dict,
			data_loader  : Union[FrameLoader,       dict],
			data_writer  : Union[FrameWriter,       dict],
			result_writer: Union[AICCountingWriter, dict],
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			save_image   : bool            = False,
			save_video   : bool            = False,
			save_results : bool            = True,
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
			rois (list[ROI], dict):
				List of ROIs or a config dictionary.
			mois (list[MOI], dict):
				List of MOIs or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			tracker (BaseTracker, dict):
				Tracker object or a tracker's config dictionary.
			moving_object (dict):
				Config dictionary of moving object.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			result_writer (AICCountingWriter, dict):
				Result writer object or a result writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
			save_image (bool):
				Should save individual images? Default: `False`.
			save_video (bool):
				Should save video? Default: `False`.
			save_results (bool):
				Should save results? Default: `False`.
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		self.subset            = subset
		self.moving_object_cfg = moving_object
		self.verbose           = verbose
		self.save_image        = save_image
		self.save_video        = save_video
		self.save_results      = save_results

		self.init_dirs()
		self.init_class_labels(class_labels=class_labels)
		self.init_detector(detector=detector)
		self.init_data_loader(data_loader=data_loader)
		self.init_data_writer(data_writer=data_writer)
		self.init_result_writer(result_writer=result_writer)

		self.start_time = None
		self.pbar       = None

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs."""
		self.root_dir    = os.path.join(data_dir, self.dataset)
		self.configs_dir = os.path.join(self.root_dir, "configs")
		self.rmois_dir   = os.path.join(self.root_dir, "rmois")
		self.outputs_dir = os.path.join(self.root_dir, "outputs")
		self.video_dir   = os.path.join(self.root_dir, self.subset)

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
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_data_loader(self, data_loader: Union[FrameLoader, dict]):
		"""Initialize data loader.

		Args:
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
		"""
		if isinstance(data_loader, FrameLoader):
			self.data_loader = data_loader
		elif isinstance(data_loader, dict):
			data = data_loader.get("data", "")
			if is_video_file(data):
				data_loader["data"] = data
			elif is_basename(data):
				data_loader["data"] = os.path.join(self.video_dir, f"{data}")
			elif is_stem(data):
				data_loader["data"] = os.path.join(
					self.video_dir, f"{data}.mp4"
				)
			else:
				data_loader["data"] = os.path.join(
					self.video_dir, f"{self.name}.mp4"
				)
			self.data_loader = FrameLoader(**data_loader)
		else:
			raise ValueError(f"Cannot initialize data loader with"
							 f" {data_loader}.")

	def init_data_writer(self, data_writer: Union[FrameWriter, dict]):
		"""Initialize data writer.

		Args:
			data_writer (FrameWriter, dict):
				Data writer object or a data writer's config dictionary.
		"""
		if isinstance(data_writer, FrameWriter):
			self.data_writer = data_writer
		elif isinstance(data_writer, dict):
			dst = data_writer.get("dst", "")
			if is_video_file(dst):
				data_writer["dst"] = dst
			elif is_basename(dst):
				data_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}")
			elif is_stem(dst):
				data_writer["dst"] = os.path.join(
					self.outputs_dir, f"{dst}.mp4"
				)
			else:
				data_writer["dst"] = os.path.join(
					self.outputs_dir, f"{self.name}.mp4"
				)
			data_writer["save_image"] = self.save_image
			data_writer["save_video"] = self.save_video
			self.data_writer = FrameWriter(**data_writer)

	def init_result_writer(self, result_writer: Union[AICCountingWriter, dict]):
		"""Initialize data writer.

		Args:
			result_writer (AICCountingWriter, dict):
				Result writer object or a result writer's config dictionary.
		"""
		if isinstance(result_writer, AICCountingWriter):
			self.result_writer = result_writer
		elif isinstance(result_writer, dict):
			dst = result_writer.get("dst", "")
			if os.path.isfile(dst):
				result_writer["dst"] = dst
			elif is_basename(dst):
				result_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}")
			elif is_stem(dst):
				result_writer["dst"] = os.path.join(
					self.outputs_dir, f"{dst}.txt"
				)
			else:
				result_writer["dst"] = os.path.join(
					self.outputs_dir, f"{self.name}.txt"
				)
			result_writer["camera_name"] = result_writer.get(
				"camera_name", self.name
			)
			self.result_writer = AICCountingWriter(**result_writer)

	# MARK: Run

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: phai them cai nay khong la bi memory leak
		with torch.no_grad():

			for images, indexes, files, rel_paths in self.data_loader:
				if len(indexes) == 0:
					break

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				self.pbar.update(len(indexes))  # Update pbar

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.mos        = []
		self.pbar       = tqdm(total=len(self.data_loader), desc=f"{self.name}")
		self.start_time               = timer()
		self.result_writer.start_time = self.start_time

		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		if self.save_results:
			self.result_writer.dump()

		self.mos = []
		self.pbar.close()
		cv2.destroyAllWindows()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		elapsed_time = timer() - self.start_time
		result       = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.verbose:
			cv2.imshow(self.name, result)
			cv2.waitKey(1)
		if self.save_video:
			self.data_writer.write_frame(image=result)



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
		# NOTE: Draw ROI
		[roi.draw(drawing=drawing) for roi in self.rois]
		# NOTE: Draw MOIs
		[moi.draw(drawing=drawing) for moi in self.mois]
		# NOTE: Draw Vehicles
		[gmo.draw(drawing=drawing) for gmo in self.mos]
		# NOTE: Draw frame index
		fps  = self.data_loader.index / elapsed_time
		text = (f"Frame: {self.data_loader.index}: "
				f"{format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)")
		font = cv2.FONT_HERSHEY_SIMPLEX
		org  = (20, 30)
		cv2.rectangle(img=drawing, pt1= (10, 0), pt2=(600, 40),
					  color=AppleRGB.BLACK.value, thickness=-1)
		cv2.putText(
			img       = drawing,
			text      = text,
			fontFace  = font,
			fontScale = 1.0,
			org       = org,
			color     = AppleRGB.WHITE.value,
			thickness = 2
		)
		return drawing

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import uuid
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
from core.utils.constants import AppleRGB
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.video import is_video_file
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
			dataset      : str,
			subset       : str,
			name         : str,
			class_labels : Union[ClassLabels,       dict],
			detector     : Union[BaseDetector,      dict],
			data_loader  : Union[FrameLoader,       dict],
			data_writer  : Union[FrameWriter,       dict],
			process      : dict,
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
		self.process      = process
		self.verbose      = verbose
		self.save_image   = save_image
		self.save_video   = save_video
		self.save_results = save_results

		self.init_dirs()
		self.init_class_labels(class_labels=class_labels)
		self.init_detector(detector=detector)
		self.init_data_loader(data_loader=data_loader)
		self.init_data_writer(data_writer=data_writer)

		self.start_time = None
		self.pbar       = None

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.

		Returns:

		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(self.root_dir, "outputs")
		self.video_dir   = os.path.join(self.root_dir, "videos")

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

	# MARK: Run

	def run_detection(self):
		"""Run detection model

		Returns:

		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.detector_cfg["batch_size"]
		self.init_data_loader(data_loader=self.data_loader_cfg)

		# NOTE: run detection
		pbar = tqdm(total=len(self.data_loader), desc=f"Detection: {self.name}")
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			height_img, width_img = None, None
			index_image           = -1
			out_dict              = dict()

			for images, indexes, _, _ in self.data_loader:
				# NOTE: if finish loading
				if len(indexes) == 0:
					break

				# NOTE: get size of image
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

					if self.process["save_dets_txt"]:
						with open(f'{self.data_writer["dst_label"]}/{name_index_image}.txt', 'w') as f_write:
							pass

					for index_in, instance in enumerate(batch):
						name_index_in = f"{index_in:08d}"
						bbox_xyxy     = instance.bbox

						# NOTE: avoid out of image bound
						# if int(bbox_xyxy[0]) < 0 or \
						# 		int(bbox_xyxy[1]) < 0 or \
						# 		int(bbox_xyxy[2]) > image_draw.shape[1] - 1 or \
						# 		int(bbox_xyxy[3]) > image_draw.shape[0] - 1:
						# 	print(bbox_xyxy)
						# 	continue

						# NOTE: small than 1000, removed, base on rules
						# if abs((bbox_xyxy[3] - bbox_xyxy[1]) * (bbox_xyxy[2] - bbox_xyxy[0])) < 1000:
						# 	continue

						crop_image    = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

						# NOTE: write crop object image
						if self.process["save_dets_crop"]:
							dets_crop_name = f"{name_index_image}_{name_index_in}"
							self.data_writer_dets_crop.write_frame(crop_image, dets_crop_name)

							if self.process["save_dets_pkl"]:
								out_dict[dets_crop_name] = {
									'bbox'   : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
									'frame'  : name_index_image,
									'id'     : name_index_in,
									'imgname': f"{dets_crop_name}.png",
									'class'  : instance.class_label["train_id"],
									'conf'   : instance.confidence
								}

						# NOTE: write txt
						if self.process["save_dets_txt"]:
							bbox_cxcywh_norm = bbox_xyxy_to_cxcywh_norm(bbox_xyxy, height_img, width_img)
							with open(f'{self.data_writer["dst_label"]}/{name_index_image}.txt', 'a') as f_write:
								f_write.write(f'{instance.class_label["train_id"]} '
											  f'{instance.confidence} '
											  f'{bbox_cxcywh_norm[0]} '
											  f'{bbox_cxcywh_norm[1]} '
											  f'{bbox_cxcywh_norm[2]} '
											  f'{bbox_cxcywh_norm[3]}\n')

						if self.process["save_dets_img"]:
							instance.draw(image_draw, bbox=True, score=True)

					# DEBUG: show detection result
					# cv2.imshow("result", images[index_b])
					# cv2.waitKey(1)

					# NOTE: write result image
					if self.process["save_dets_img"]:
						self.data_writer_dets_debug.write_frame(image_draw, name_index_image)

				# NOTE: get feature of all crop images

				pbar.update(len(indexes))  # Update pbar

			if self.process["save_dets_pkl"]:
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer['dst_crop_pkl'], self.name)}_dets_crop.pkl", 'wb')
				)

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
		self.pbar       = tqdm(total=len(self.data_loader), desc=f"{self.name}")
		self.start_time = timer()

		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
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

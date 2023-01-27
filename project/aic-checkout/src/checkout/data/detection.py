#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store results from detectors.
Attribute includes: bounding box, confident score, class, uuid, ...
"""

from __future__ import annotations

__all__ = [
	"Detection",
]

from timeit import default_timer as timer

import cv2
import numpy as np

import mon
from checkout.typing import UIDType
from mon.core.typing import Ints


# region Detection

class Detection:
	"""The detection object. Wrap the raw prediction from detectors.
	
	Attributes:
		uid: An unique ID. Defaults to None.
		roi_uid: The unique ID of the ROI containing the detection.
		box: A bounding box in (x1, y1, x2, y2) format.
		polygon: A list of points representing an instance mask. Defaults to
			None.
		confidence: A confidence score. Defaults to None.
		classlabel: A :class:`mon.Classlabel` object. Defaults to None.
		frame_index: The frame index of a detection. Defaults to None.
		timestamp: The creating time of the current object.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		uid        : UIDType,
		roi_uid    : UIDType,
		box        : np.ndarray,
		polygon    : np.ndarray | None = None,
		confidence : float      | None = None,
		classlabel : dict       | None = None,
		frame_index: int        | None = None,
		timestamp  : float             = timer(),
	):
		super().__init__()
		self.uid         = uid
		self.roi_uid     = roi_uid
		self.box         = box
		self.polygon     = polygon
		self.confidence  = confidence
		self.classlabel  = classlabel
		self.frame_index = frame_index
		self.timestamp   = timestamp
		
	@property
	def box_cxcyrh(self):
		return mon.box_xyxy_to_cxcyrh(box=self.box)

	@property
	def box_center(self):
		return mon.get_box_center(box=self.box)
	
	@property
	def box_tl(self):
		"""The box's top left corner."""
		return self.box[0:2]
		
	def draw(
		self,
		drawing: np.ndarray,
		box    : bool        = False,
		polygon: bool        = False,
		label  : bool        = True,
		color  : Ints | None = None
	) -> np.ndarray:
		"""Draw the current object on the :param:`drawing`."""
		color = color \
			or (self.classlabel["color"] if self.classlabel is not None
			    else (255, 255, 255))
		
		if box:
			cv2.rectangle(
				img       = drawing,
				pt1       = (self.box[0], self.box[1]),
				pt2       = (self.box[2], self.box[3]),
				color     = color,
				thickness = 2
			)
		
		if polygon:
			pts = self.polygon.reshape((-1, 1, 2))
			cv2.polylines(
				img       = drawing,
				pts       = pts,
				isClosed  = True,
				color     = color,
				thickness = 2,
			)
		
		if label:
			font = cv2.FONT_HERSHEY_SIMPLEX
			org  = (self.box_tl[0] + 5, self.box_tl[1])
			cv2.putText(
				img       = drawing,
				text      = self.classlabel["name"],
				fontFace  = font,
				fontScale = 1.0,
				org       = org,
				color     = color,
				thickness = 2
			)
		return drawing

# endregion
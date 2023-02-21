#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for retail checkout cameras."""

from __future__ import annotations

__all__ = [
    "CheckoutCamera",
]

import uuid
from timeit import default_timer as timer
from typing import Any

import cv2
import numpy as np

import mon
from mon.foundation import math
from supr import data, detect, io, rmoi, track
from supr.camera.base import Camera
from supr.globals import DETECTORS, MovingState, TRACKERS


# region CheckoutCamera

class CheckoutCamera(Camera):
    """Retail checkout camera.
    
    See Also: :class:`supr.camera.base.Camera`.
    """
    
    def __init__(
        self,
        root         : mon.Path,
        subset       : str,
        name         : str,
        rois         : Any,
        mois         : Any,
        classlabels  : Any,
        num_classes  : int | None,
        image_loader : Any,
        image_writer : Any,
        result_writer: Any,
        detector     : Any,
        tracker      : Any,
        moving_object: Any,
        id_          : int | str = uuid.uuid4().int,
        save_image   : bool      = False,
        save_video   : bool      = False,
        save_result  : bool      = True,
        verbose      : bool      = False,
        *args, **kwargs
    ):
        super().__init__(
            id_          = id_,
            root         = root,
            subset       = subset,
            name         = name,
            image_loader = image_loader,
            image_writer = image_writer,
            save_image   = save_image,
            save_video   = save_video,
            save_result  = save_result,
            verbose      = verbose
        )
        self.rois              = rmoi.ROI.from_value(value=rois)
        self.mois              = rmoi.MOI.from_value(value=mois)
        self.classlabels       = mon.ClassLabels.from_value(value=classlabels)
        self.num_classes       = num_classes or len(self.classlabels) \
                                 if self.classlabels is not None else 0
        self.result_writer     = result_writer
        self.detector          = detector
        self.tracker           = tracker
        self.moving_object_cfg = moving_object
        self.moving_objects    = []
        self.start_time        = None
        self.init_moving_object()
        
    @property
    def result_writer(self) -> io.ProductCountingWriter:
        return self._result_writer
    
    @result_writer.setter
    def result_writer(self, result_writer: Any):
        if not self.save_result:
            self._result_writer = None
        elif isinstance(result_writer, io.ProductCountingWriter):
            self._result_writer = result_writer
        elif isinstance(result_writer, dict):
            destination = mon.Path(result_writer.get("destination", None))
            if destination.is_dir():
                destination = destination / f"{self.name}.txt"
            elif destination.is_basename() or destination.is_stem():
                destination = self.result_dir / f"{destination}.txt"
            if not destination.is_txt_file(exist=False):
                raise ValueError(
                    f"destination must be a valid path to a .txt file, but got "
                    f"{destination}."
                )
            result_writer["destination"] = destination
            self._result_writer = io.ProductCountingWriter(**result_writer)
        else:
            raise ValueError(
                f"Cannot initialize result writer with {result_writer}."
            )
    
    @property
    def detector(self):
        return self._detector
    
    @detector.setter
    def detector(self, detector: Any):
        if isinstance(detector, detect.Detector):
            self._detector = detector
        elif isinstance(detector, dict):
            detector["classlabels"] = self.classlabels
            self._detector = DETECTORS.build(**detector)
        else:
            raise ValueError(f"Cannot initialize detector with {detector}.")
    
    @property
    def tracker(self):
        return self._tracker
    
    @tracker.setter
    def tracker(self, tracker: Any):
        if isinstance(tracker, track.Tracker):
            self._tracker = tracker
        elif isinstance(tracker, dict):
            self._tracker = TRACKERS.build(**tracker)
        else:
            raise ValueError(f"Cannot initialize tracker with {tracker}.")
    
    def init_moving_object(self):
        cfg = self.moving_object_cfg
        data.Product.min_traveled_distance = cfg["min_traveled_distance"]
        data.Product.min_entering_distance = cfg["min_entering_distance"]
        data.Product.min_hit_streak        = cfg["min_hit_streak"]
        data.Product.max_age               = cfg["max_age"]
        data.Product.min_touched_landmarks = cfg["min_touched_landmarks"]
        data.Product.max_untouches_age     = cfg["max_untouches_age"]
    
    def on_run_start(self):
        """Called at the beginning of run loop."""
        self.moving_objects = []
        self.start_time     = timer()
        self.result_writer.start_time = self.start_time
        mon.mkdirs(
            paths    = [self.subset_dir, self.result_dir],
            parents  = True,
            exist_ok = True
        )
        if self.verbose:
            cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
            
    def run(self):
        """Main run loop."""
        self.on_run_start()

        with mon.get_progress_bar() as pbar:
            for images, indexes, files, rel_paths in pbar.track(
                self.image_loader,
                total       = self.image_loader.batch_len(),
                description = f"[bright_yellow]{self.name}"
            ):
                if len(indexes) == 0 or images is None:
                    break
                    
                # Detect
                batch_instances = self.detector.detect(indexes=indexes, images=images)
                
                # Get ROIs
                for idx, instances in enumerate(batch_instances):
                    roi_ids = [rmoi.get_roi_for_box(bbox=i.bbox, rois=self.rois) for i in instances]
                    for i, roi_id in zip(instances, roi_ids):
                        i.roi_id = roi_id
                        
                # Track
                for idx, instances in enumerate(batch_instances):
                    self.tracker.update(instances=instances)
                    self.moving_objects: list[data.Product] = self.tracker.tracks
                    
                    # Update moving objects' moving state
                    for mo in self.moving_objects:
                        mo.update_moving_state(rois=self.rois)
                        if mo.is_confirmed:
                            mo.timestamp = math.floor(
                                (mo.current.frame_index - self.tracker.min_hits)
                                / self.image_loader.fps
                            )    
                        
                    # Count
                    countable = [o for o in self.moving_objects if o.is_to_be_counted]
                    if self.save_result:
                        self.result_writer.append_results(products=countable)
                    for mo in countable:
                        mo.moving_state = MovingState.COUNTED
                    self.run_step_end(image=images[idx])
                
        self.on_run_end()
        
    def run_step_end(self, image: np.ndarray):
        """Perform some postprocessing operations when a run step end."""
        if not (self.verbose or self.save_image or self.save_video):
            return
        elapsed_time = timer() - self.start_time
        image        = self.draw(image=image, elapsed_time=elapsed_time)
        if self.save_video:
            self.image_writer.write(image=image)
        if self.verbose:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, image)
            cv2.waitKey(1)
            
    def on_run_end(self):
        """Called at the end of run loop."""
        if self.save_result:
            self.result_writer.write_to_file()
        self.mos = []
        if self.verbose:
            cv2.destroyAllWindows()
    
    def draw(self, image: np.ndarray, elapsed_time: float) -> np.ndarray:
        """Visualize the results on the image.

        Args:
            image: Drawing canvas.
            elapsed_time: Elapsed time per iteration.
        """
        # NOTE: Draw ROI
        [r.draw(image=image) for r in self.rois]
        # NOTE: Draw MOIs
        [m.draw(image=image) for m in self.mois]
        # NOTE: Draw Products
        [o.draw(image=image) for o in self.moving_objects]
        # NOTE: Draw frame index
        index = self.image_loader.index
        fps   = index / elapsed_time
        text  = f"Frame: {index}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
        cv2.rectangle(
            img       = image,
            pt1       = (10,   0),
            pt2       = (600, 40),
            color     = (0, 0, 0),
            thickness = -1
        )
        cv2.putText(
            img       = image,
            text      = text,
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1.0,
            org       = (20, 30),
            color     = (255, 255, 255),
            thickness = 2
        )
        return image
    
# endregion

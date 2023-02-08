#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
import sys
from timeit import default_timer as timer
from time import perf_counter

import yaml

from core.utils.rich import console
from cameras import AICTrafficSafetyCamera

from configuration import (
	data_dir,
	load_config
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--config", default="c041.yaml",
	help="Config file for each camera. Final path to the config file "
		 "is: tss/data/[dataset]/configs/[config]/"
)
parser.add_argument(
	"--detection", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--verbose", action='store_true', help="Should visualize the images."
)

Camera = AICTrafficSafetyCamera


# MARK: - Main Function

def main():
	# NOTE: Start timer
	process_start_time = perf_counter()
	camera_start_time  = perf_counter()

	# NOTE: Parse camera config
	args        = parser.parse_args()
	config_path = os.path.join(data_dir, args.dataset, "configs", args.config)
	camera_cfg  = load_config(config_path)

	# DEBUG: print camera config
	# print(camera_cfg)

	# Update value from args
	camera_cfg["dataset"]      = args.dataset
	camera_cfg["subset"]       = args.subset
	camera_cfg["verbose"]      = args.verbose
	camera_cfg["process"]      = {
		"function_dets" : args.detection,  # Detection
		"save_dets_crop": True,
		"save_dets_pkl" : True,
	}

	# NOTE: add merge feat config if needed
	camera_cfg["merge_feat"] = None
	config_path = os.path.join(data_dir, args.dataset, "configs", args.config_feat_merge)
	with open(config_path) as f:
		camera_cfg["featuremerger"] = yaml.load(f, Loader=yaml.FullLoader)

	# NOTE: Define camera
	camera           = Camera(**camera_cfg)
	camera_init_time = perf_counter() - camera_start_time

	# NOTE: Process
	camera.run()

	# NOTE: End timer
	total_process_time = perf_counter() - process_start_time
	console.log(f"Total processing time: {total_process_time} seconds.")
	console.log(f"Camera init time: {camera_init_time} seconds.")
	console.log(f"Actual processing time: "
				f"{total_process_time - camera_init_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()

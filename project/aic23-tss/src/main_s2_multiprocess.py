#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from timeit import default_timer as timer
from time import perf_counter

import yaml

from core.utils.rich import console
from cameras import AICTrafficSafetyCameraS2Multiprocess

from configuration import (
	data_dir,
	config_dir,
	load_config
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--config", default="c041.yaml",
	help="Config file for each camera. Final path to the config file."
)
parser.add_argument(
	"--dataset", default="aic23_trafficsafety",
	help="Dataset to run on."
)
parser.add_argument(
	"--detection", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--identification", action='store_true', help="Should run identification."
)
parser.add_argument(
	"--write_final", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--verbose", action='store_true', help="Should visualize the images."
)

Camera = AICTrafficSafetyCameraS2Multiprocess


# MARK: - Main Function

def run_node(_, index_node):
	args = parser.parse_args()
	config_path = os.path.join(config_dir, args.config)

	camera_cfg  = load_config(config_path)

	# Update value from args
	camera_cfg["dataset"]       = args.dataset
	camera_cfg["verbose"]       = args.verbose
	camera_cfg["process"]       = {
		"function_dets"         : args.detection,       # Detection
		"function_identify"     : args.identification,  # Identification
		"function_write_final"  : args.write_final,     # Writing final results.
	}
	camera_cfg["process_index"] = index_node

	# NOTE: Define camera
	camera           = Camera(**camera_cfg)

	# NOTE: Process
	camera.run()


def main():
	# NOTE: Start timer
	process_start_time = perf_counter()

	# NOTE: Parse camera config
	args        = parser.parse_args()
	config_path = os.path.join(config_dir, args.config)
	camera_cfg = load_config(config_path)

	processes = []

	# NOTE: Define processes
	for index_node in range(int(camera_cfg['data']['process_num'])):
		processes.append(multiprocessing.Process(target=run_node, args=([], index_node)))

	# NOTE: Start processes
	for process in processes:
		process.start()

	# NOTE: Wait all processes stop
	for process in processes:
		process.join()

	# NOTE: End timer
	total_process_time = perf_counter() - process_start_time
	console.log(f"Total processing time: {total_process_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()

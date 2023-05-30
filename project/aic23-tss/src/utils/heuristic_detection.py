#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle
import sys
import os
import glob
from enum import Enum
import shutil
import random
from operator import itemgetter, attrgetter

import multiprocessing
import threading

from tqdm import tqdm
import numpy as np
import cv2

from utils.classes_aic23_track5 import *


# NOTE: HEURISTIC FOR MOTORBIKE------------------------------------------------

def heuristic_motorbike(crop_dict):
	"""Run the heuristic for motorbike

	Args:
		crop_dict(dict):

	Returns:
		results_dict(dict):
	"""
	# initial value
	results_dict = [crop_dict]
	bbox_ori     = crop_dict['bbox']

	# fine new bounding box
	x_0_1 = float(bbox_ori[0])
	y_0_1 = float(bbox_ori[1])
	x_0_2 = float(bbox_ori[2])
	y_0_2 = float(bbox_ori[3])
	x_0_c = (x_0_1 + x_0_2) / 2
	y_0_c = (y_0_1 + y_0_2) / 2
	y_1_2 = y_0_c
	ratio = 1.2
	y_1_1 = max(0.0, y_1_2 - (abs(x_0_2 - x_0_1) / 1.2))
	x_1_1 = x_0_c - (((y_0_2 - y_0_1) / 2) / ratio)
	x_1_2 = x_0_c + (((y_0_2 - y_0_1) / 2) / ratio)

	# if abs(x_1_2 - x_1_1) >= abs(y_1_2 - y_1_1):  # scale horizontal for drivers
	# 	ratio = 1.2
	# 	x_1_1 = x_0_c - ((y_0_2 - y_0_1) / 2) / ratio
	# 	x_1_2 = x_0_c + ((y_0_2 - y_0_1) / 2) / ratio

	bbox_result = (int(x_1_1), int(y_1_1), int(x_1_2), int(y_1_2))

	for train_id in range(1, 3):
		# Add driver with/without helmet
		result_dict_temp = {
			'video_name': crop_dict['video_name'],
			'frame_id'  : crop_dict['frame_id'],
			'crop_img'  : crop_dict['crop_img'],
			'bbox'      : bbox_result,
			'class_id'  : train_id,  # train_id
			'id'        : train_id + 1,
			'conf'      : crop_dict['conf'],
			'width_img' : crop_dict['width_img'],
			'height_img': crop_dict['height_img']
		}
		results_dict.append(result_dict_temp)

	return results_dict


# NOTE: HEURISTIC FOR DRIVER AND PASSENGER-------------------------------------


def add_motorbike(crop_dict):
	# initial value
	results_dict = []
	bbox_ori     = crop_dict['bbox']

	# fine new bounding box
	x_0_1 = float(bbox_ori[0])
	y_0_1 = float(bbox_ori[1])
	x_0_2 = float(bbox_ori[2])
	y_0_2 = float(bbox_ori[3])
	x_0_c = (x_0_1 + x_0_2) / 2
	y_0_c = (y_0_1 + y_0_2) / 2
	y_1_1 = y_0_c
	ratio = 1.2
	y_1_2 = min(crop_dict['height_img'], y_1_1 + (abs(x_0_2 - x_0_1) / ratio))
	x_1_1 = x_0_c - (((y_0_2 - y_0_1) / 2) / ratio)
	x_1_2 = x_0_c + (((y_0_2 - y_0_1) / 2) / ratio)

	bbox_result = (int(x_1_1), int(y_1_1), int(x_1_2), int(y_1_2))

	# Add motorbike
	result_dict_temp = {
		'video_name': crop_dict['video_name'],
		'frame_id'  : crop_dict['frame_id'],
		'crop_img'  : crop_dict['crop_img'],
		'bbox'      : bbox_result,
		'class_id'  : 0,  # train_id of motorbike
		'id'        : 1,
		'conf'      : crop_dict['conf'],
		'width_img' : crop_dict['width_img'],
		'height_img': crop_dict['height_img']
	}
	results_dict.append(result_dict_temp)

	return results_dict


def add_driver_or_passenger(crop_dict):
	# initial value
	results_dict = []

	# check id for add passenger:
	if int(crop_dict['class_id']) == 0:  # if id la motorbike
		return results_dict
	if int(crop_dict['class_id']) % 2 == 0:
		max_id = int(crop_dict['class_id']) + 1
	else:
		max_id = int(crop_dict['class_id']) + 2

	# if max_id > 7 more than classes number
	max_id = 7 if max_id > 7 else max_id

	for train_id in range(max_id):
		if train_id == 0:  # motorbike id
			continue
		if train_id == int(crop_dict['class_id']):  # same as refer object
			continue

		# Add driver and passengers:
		result_dict_temp = {
			'video_name': crop_dict['video_name'],
			'frame_id'  : crop_dict['frame_id'],
			'crop_img'  : crop_dict['crop_img'],
			'bbox'      : crop_dict['bbox'],
			'class_id'  : train_id,  # train_id of motorbike
			'id'        : train_id + 1,
			'conf'      : crop_dict['conf'],
			'width_img' : crop_dict['width_img'],
			'height_img': crop_dict['height_img']
		}
		results_dict.append(result_dict_temp)

	return results_dict


def heuristic_person(crop_dict):
	"""Run the heuristic for driver and passengers

	Args:
		crop_dict(dict):

	Returns:
		results_dict(dict):
	"""
	# initial value
	results_dict = [crop_dict]
	bbox_ori     = crop_dict['bbox']

	# NOTE: add motorbike
	result_dict_temp = add_motorbike(crop_dict)
	for result_dict in result_dict_temp:
		results_dict.append(result_dict)

	# NOTE: add high lever driver or passenger
	result_dict_temp = add_driver_or_passenger(crop_dict)
	for result_dict in result_dict_temp:
		results_dict.append(result_dict)

	return results_dict


def heuristic_objecst(det_crop):
	classes_7 = get_list_7_classses()
	out_dict = []

	# Initial information
	crop_dict = {
		'video_name': det_crop['video_name'],
		'frame_id'  : det_crop['frame_id'],
		'crop_img'  : det_crop['crop_img'],
		'bbox'      : det_crop['bbox'],
		'class_id'  : det_crop['class_id'],  # train_id
		'id'        : det_crop['id'],
		'conf'      : det_crop['conf'],
		'width_img' : det_crop['width_img'],
		'height_img': det_crop['height_img']
	}

	if crop_dict['id'] == 1:  # if the id object is motorbike
		results_dict = heuristic_motorbike(crop_dict)
	else:
		results_dict = heuristic_person(crop_dict)

	for result_dict in results_dict:
		out_dict.append(result_dict)

	return out_dict


def main():
	pass

if __name__ == "__main__":
	main()

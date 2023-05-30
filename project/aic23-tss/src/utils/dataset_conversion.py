#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import pickle
from enum import Enum
import shutil
import random
from operator import itemgetter, attrgetter

import multiprocessing
import threading

from tqdm import tqdm
import numpy as np
import cv2

import albumentations as A

from classes_aic23_track5 import *


# NOTE: SMALL UTILITIES -------------------------------------------------------


def make_dir(path):
	"""Make dir"""
	if not os.path.isdir(path):
		os.makedirs(path)


class AppleRGB(Enum):
	"""Apple's 12 RGB colors."""

	RED    = (255, 59 , 48)
	GREEN  = (52 , 199, 89)
	BLUE   = (0  , 122, 255)
	ORANGE = (255, 149, 5)
	BROWN  = (162, 132, 94)
	PURPLE = (88 , 86 , 214)
	TEAL   = (90 , 200, 250)
	INDIGO = (85 , 190, 240)
	BLACK  = (0  , 0  , 0)
	PINK   = (255, 45 , 85)
	WHITE  = (255, 255, 255)
	GRAY   = (128, 128, 128)
	YELLOW = (255, 204, 0)

# NOTE: VIDEOS - IMAGES EXTRACTION---------------------------------------------


def extract_video(basename_noext, video_path, folder_image_out):
	cam = cv2.VideoCapture(video_path)
	index = 0

	while True:

		index = index + 1

		# reading from frame
		ret, frame = cam.read()

		if ret:
			# writing the extracted images
			image_path = os.path.join(folder_image_out, f"{basename_noext}{index:05d}.jpg")
			cv2.imwrite(image_path, frame)
		else:
			break

	# Release all space and windows once done
	cam.release()


def extract_videos():
	# Initial parameters
	folder_root = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"

	# Get list video
	video_paths = sorted(glob.glob(os.path.join(folder_root, "videos", "*.mp4")))

	# Extract video
	for video_path in tqdm(video_paths):
		basename = os.path.basename(video_path)
		basename_noext = os.path.splitext(basename)[0]

		# Create the output folder
		folder_image_out = os.path.join(folder_root, "images", basename_noext)
		make_dir(folder_image_out)

		# Extract one video
		extract_video(basename_noext, video_path, folder_image_out)


# NOTE: EXTRACT GROUNDTRUTH FILE------------------------------------------------


def read_groundtruth():
	gt_path = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/gt.txt"
	labels = []
	with open(gt_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			labels.append([int(word) for word in line.replace('\n', '').split(',')])
	return labels


def voc_to_yolo_format(size, box):
	dw = 1. / (size[0])
	dh = 1. / (size[1])

	x = (box[0] + box[2]) / 2.0
	y = (box[1] + box[3]) / 2.0
	w = abs(box[2] - box[0])
	h = abs(box[3] - box[1])

	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh

	return x, y, w, h


"""
### 7 classes: Minus 1 each class
1, motorbike   0
2, DHelmet     1
3, DNoHelmet   2
4, P1Helmet    3
5, P1NoHelmet  4
6, P2Helmet    5
7, P2NoHelmet  6
### 3 classes: Minus 1 each class
1, motorbike  0
2, Helmet     1
3, NoHelmet   2
### 2 classes: Minus 1 each class
1, motorbike  0
2, driver     1
### 1 class: Minus 1 each class
1, motorbike  0
"""
def extract_groundtruth(basename_noext, label, folder_label_out, size, num_class=None):
	# NOTE: sort label base on frame id, track id
	label = sorted(label, key=itemgetter(1, 2))

	# <video_id>, <frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
	frame_id   = 1
	label_temp = []
	for index in tqdm(range(len(label)), desc=f"{basename_noext}"):
		if label[index][1] == frame_id:
			label_temp.append(label[index])

		if index == len(label) - 1 or label[index + 1][1] != frame_id:
			with open(os.path.join(folder_label_out, f"{basename_noext}{frame_id:05d}.txt"), "w") as f_write:
				for line in label_temp:
					x, y, w, h = voc_to_yolo_format(size, [line[3], line[4], line[3] + line[5], line[4] + line[6]])

					# NOTE: for 7 classes, uncomment if you need to use
					if num_class == 7:
						cls = line[7] - 1
					# NOTE: for 3 classes, uncomment if you need to use
					elif num_class == 3:
						if line[7] in [2, 4, 6]:
							cls = 1
						elif line[7] in [3, 5, 7]:
							cls = 2
						else:
							cls = 0
					# NOTE: for 2 classes, uncomment if you need to use
					elif num_class == 2:
						if line[7] > 1:
							cls = 1
						else:
							cls = 0
					else:
						cls = line[7] - 1
					f_write.write(f"{cls} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n")

			label_temp = []
			if index < len(label) - 1:
				frame_id = label[index + 1][1]


def extract_groundtruths(num_class):
	# Initial parameters
	folder_root = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"

	# Get list video
	video_paths = sorted(glob.glob(os.path.join(folder_root, "videos", "*.mp4")))

	# get groundtruth
	labels = read_groundtruth()

	# Extract video
	for video_path in tqdm(video_paths):
		basename = os.path.basename(video_path)
		basename_noext = os.path.splitext(basename)[0]

		# Create the output folder
		folder_label_out = os.path.join(folder_root, f"labels_{num_class}_classes", basename_noext)
		make_dir(folder_label_out)

		# get width, height video
		cam    = cv2.VideoCapture(video_path)
		width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
		height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
		cam.release()

		# get label for one video
		label = [line for line in labels if line[0] == int(basename_noext)]

		# Extract one video
		extract_groundtruth(basename_noext, label, folder_label_out, (width, height), num_class)


# NOTE: EXTRACT PICKLE FILE----------------------------------------------------


def extract_crop_image(out_dicts, folder_image_ou, folder_label_ou):
	crop_id    = -1
	crop_dicts = []
	out_dicts  = sorted(out_dicts, key=itemgetter('crop_id'))
	for out_dict in out_dicts:
		if crop_id != int(out_dict['crop_id']):

			if crop_id > -1:
				if len(crop_dicts) > 1:  # filter the number of bounding box in crop image
					crop_img        = crop_dicts[0]['crop_img']
					h_img, w_img, _ = crop_img.shape
					basename_crop = f"{int(crop_dicts[0]['video_name']):03d}_{int(crop_dicts[0]['frame_id']):05d}_{crop_id:05d}"

					# save crop image
					cv2.imwrite(os.path.join(folder_image_ou, f"{basename_crop}.jpg"), crop_img)

					# write label
					with open(os.path.join(folder_label_ou, f"{basename_crop}.txt"), "w") as f_write:
						for crop_dict in crop_dicts:
							bbox_xyxy = crop_dict['bbox']
							x, y, w, h = voc_to_yolo_format((w_img, h_img), bbox_xyxy)
							f_write.write(f"{crop_dict['class_id']} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n")

			crop_id    = int(out_dict['crop_id'])
			crop_dicts = [out_dict]
		elif crop_id == int(out_dict['crop_id']):
			crop_dicts.append(out_dict)


def extract_pickles():
	"""extract label from final result"""
	# initial parameters
	folder_pkl_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/outputs_s2_v8_det_v8_iden/dets_full_pkl/yolov8x6/"
	folder_img_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/images/"
	folder_ou     = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/labels_7_classes_cropped/"

	# Create the output folder
	folder_image_ou = os.path.join(folder_ou, "images")
	folder_label_ou = os.path.join(folder_ou, "labels")
	make_dir(folder_image_ou)
	make_dir(folder_label_ou)

	# get list pickle
	pkl_paths = sorted(glob.glob(os.path.join(folder_pkl_in, "*.pkl")))

	# run each pickle
	for pkl_path in tqdm(pkl_paths, desc="Extraction pickles"):
		dets_crop = pickle.load(open(pkl_path, 'rb'))
		dets_crop = sorted(dets_crop, key=itemgetter('frame_id'))

		# run each frame
		frame_id  = -1
		out_dicts = []
		for det_crop in tqdm(dets_crop, desc=f"Processing {os.path.basename(pkl_path)}"):
			if frame_id != int(det_crop['frame_id']):
				if frame_id > 0:
					if len(out_dicts) > 1:  # filter if there is any 1 object in crop image
						extract_crop_image(out_dicts, folder_image_ou, folder_label_ou)
				frame_id = int(det_crop['frame_id'])
				out_dicts = [det_crop]
			elif frame_id == int(det_crop['frame_id']):
				if float(det_crop['conf']) >= 0.1:  # filter confident score:
					if int(abs(det_crop['bbox'][2] - det_crop['bbox'][0])) >= 100  \
						and int(abs(det_crop['bbox'][3] - det_crop['bbox'][1])) >= 100:  # filter size bounding box
							out_dicts.append(det_crop)


# NOTE: COPY FILES FROM MANY FOLDERS INTO TWO FOLDERS--------------------------


def copy_all_file(num_class):
	# NOTE: initial parameters
	image_root = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images/"
	label_root = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_{num_class}_classes_filtered/"
	image_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/{num_class}_classes/train/images/"
	label_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/{num_class}_classes/train/labels/"

	# NOTE: create directory
	make_dir(image_out)
	make_dir(label_out)

	# NOTE: Get list
	list_label = glob.glob(os.path.join(label_root, "*/*.txt"))

	# NOTE: Copy all files
	for index, lbl_path in enumerate(tqdm(list_label)):

		if index % 1 > 0:
		# if index % 5 > 0:
			continue

		# get base information
		basename       = os.path.basename(lbl_path)
		basename_noext = os.path.splitext(basename)[0]

		# check image exist
		img_path = glob.glob(os.path.join(image_root, f"*/{basename_noext}.jpg"))
		if len(img_path) == 0:
			continue

		# copy file
		img_path = img_path[0]
		img_new  = os.path.join(image_out, f"{basename_noext}.jpg")
		lbl_new  = os.path.join(label_out, f"{basename_noext}.txt")
		shutil.copyfile(img_path, img_new)
		shutil.copyfile(lbl_path, lbl_new)


# NOTE: GROUP MORE LABELS INTO ONE LABEL-------------------------------------


def find_intersect(box1, box2):
	dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
	dy = min(box1[3], box2[3]) - max(box1[1], box2[1])

	if (dx >= 0) and (dy >= 0):
		return dx * dy
	return 0.0


def group_label_many_into_one(labels):
	"""

	Args:
		labels (np.ndarray): the ungrouped labels

	Returns:
		labels (np.ndarray) : the grouped labels
		motor_Drivers (dict): the drivers with their posiblle motor on the labels
	"""
	labels_temp    = []
	motor_drivers  = {}

	# NOTE: surf all driver
	# find the motor the driver belong
	# if the driver in not on any motor, we convert it into motor also
	for index, label in enumerate(labels):
		list_motor   = []
		list_overlap = []

		if label[0] != 0:  # if it is not motor
			box1 = [
				label[1] - (label[3] / 2),
				label[2] - (label[4] / 2),
				label[1] + (label[3] / 2),
				label[2] + (label[4] / 2)
			]   # because of YOLO format

			for index_sub, label_sub in enumerate(labels):

				if index_sub != index:  # it is not the same object
					if label_sub[0] == 0:  # if it is motor
						box2 = [
							label_sub[1] - (label_sub[3] / 2),
							label_sub[2] - (label_sub[4] / 2),
							label_sub[1] + (label_sub[3] / 2),
							label_sub[2] + (label_sub[4] / 2)
						]
						overlap = find_intersect(box1, box2)
						if overlap > 0.0:  # if they overlap
							list_motor.append(index_sub)
							list_overlap.append(overlap)

			if len(list_motor) == 0:  # if the driver in not on any motor, we convert it into motor also
				labels_temp.append([ele if index_sub > 0 else 0 for index_sub, ele in enumerate(label)])
			else:  # add it into list for post process
				index_motor = list_motor[np.argmax(list_overlap)]  # get the index of motor in labels

				if index_motor not in motor_drivers:  # check do we have the motor in list
					motor_drivers[index_motor] = []

				motor_drivers[index_motor].append(index)

	# NOTE: group driver into motor
	for key, values in motor_drivers.items():
		# Get value from motor
		x_min, y_min, x_max, y_max = [], [], [], []
		x_min.append(labels[key][1] - (labels[key][3] / 2))
		y_min.append(labels[key][2] - (labels[key][4] / 2))
		x_max.append(labels[key][1] + (labels[key][3] / 2))
		y_max.append(labels[key][2] + (labels[key][4] / 2))

		# Get all value from driver
		for value in values:
			x_min.append(labels[value][1] - (labels[value][3] / 2))
			y_min.append(labels[value][2] - (labels[value][4] / 2))
			x_max.append(labels[value][1] + (labels[value][3] / 2))
			y_max.append(labels[value][2] + (labels[value][4] / 2))

		# find the proper value
		x_min = min(x_min)
		y_min = min(y_min)
		x_max = max(x_max)
		y_max = max(y_max)

		# add to final list
		labels_temp.append([
			0,
			(x_max + x_min) / 2,
			(y_max + y_min) / 2,
			abs(x_max - x_min),
			abs(y_max - y_min),
		])

	# NOTE: for any motor we have not checked so far
	for index, label in enumerate(labels):
		if label[0] == 0:  # if it is motor
			if index not in motor_drivers:
				labels_temp.append(label)

	return labels_temp, motor_drivers


def group_labels_many_into_one():
	"""Group all driver and motorbike into one label :: motorbike
		Only run this function after convert`		 the label from 7 classes into 2 classes
	"""
	# Initial parameter
	folder_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_7_classes_filtered/"
	folder_ou = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_1_classes_filtered/"

	# DEBUG:
	classes_7 = get_list_7_classses()
	classes_n = set()

	# Get list label
	labels_path = sorted(glob.glob(os.path.join(folder_in, "*/*.txt")))
	dir_list    = os.listdir(folder_in)

	# Create list dir
	for dir_path in dir_list:
		make_dir(os.path.join(folder_ou, dir_path))

	# Grouping
	for label_path_in in tqdm(labels_path):
		# Read label
		labels_temp = []
		with open(label_path_in, 'r') as f_open:
			lines = f_open.readlines()
			for line in lines:
				labels_temp.append([float(word) for word in line.replace('\n', '').split(' ')])

		# Group one label
		labels, motor_drivers = group_label_many_into_one(labels_temp)

		# DEBUG:
		# print(labels_temp)
		# print(motor_drivers)
		for index_motor, list_drivers in motor_drivers.items():
			name_temp = ""
			if len(list_drivers) > 1:
				# print(index_motor, list_drivers)
				pass

			for index_driver in list_drivers:
				name_temp = name_temp + str(classes_7[int(labels_temp[index_driver][0])])
			# print("****")
			# print(list_drivers)
			# print(name_temp)
			classes_n.add(name_temp)


		# Get new name
		label_path_ou = label_path_in.replace(folder_in, folder_ou)
		with open(label_path_ou, 'w') as f_write:
			for line in labels:
				f_write.write(f"{int(line[0])} {float(line[1]):.10f} {float(line[2]):.10f} {float(line[3]):.10f} {float(line[4]):.10f}\n")

	# DEBUG:
	print(sorted(list(classes_n)))


# NOTE: DEFINE NEW LABELS BASED -----------------------------------------------


"""
### 7 classes
1, motorbike   0
2, DHelmet     1
3, DNoHelmet   2
4, P1Helmet    3
5, P1NoHelmet  4
6, P2Helmet    5
7, P2NoHelmet  6
###
"""
def define_labels_based_on_paper():
	pass


# NOTE: VISUALIZE LABEL -------------------------------------------------------


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
	"""Plots one bounding box on image img"""
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def visualization_image(img, label_path, colors, labels_name):
	# NOTE:
	h_img, w_img, _ = img.shape

	# NOTE: read label
	with open(label_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			label     = [float(word) for word in line.replace('\n', '').split(' ')]
			x_min     = label[1] - (label[3] / 2)
			y_min     = label[2] - (label[4] / 2)
			x_max     = label[1] + (label[3] / 2)
			y_max     = label[2] + (label[4] / 2)
			cls_index = int(label[0])
			box = [
				x_min * w_img,
				y_min * h_img,
				x_max * w_img,
				y_max * h_img
			]
			plot_one_box(
				x     = box,
				img   = img,
				color = colors[cls_index],
				label = labels_name[cls_index]
			)


def initial_label_name(num_class):
	labels_name = []
	if num_class == 7:
		labels_name = [
			"motorbike",
			"DHelmet",
			"DNoHelmet",
			"P1Helmet",
			"P1NoHelmet",
			"P2Helmet",
			"P2NoHelmet"
		]
	elif num_class == 3:
		labels_name = [
			"motorbike",
			"Helmet",
			"NoHelmet"
		]
	elif num_class == 2:
		labels_name = [
			"motorbike",
			"driver"
		]
	elif num_class == 1:
		labels_name = [
			"motorbike"
		]
	return labels_name


def visualization_images(_, num_classes):
	# initial parameter folder
	folder_root     = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"
	folder_image_in = folder_root + "images/"
	folder_label_in = folder_root + f"labels_{num_classes}_classes_filtered/"
	folder_image_ou = folder_root + f"visualization/images_draw_{num_classes}_classes/"

	# initial label name
	labels_name = initial_label_name(num_classes)

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# Create list dir
	dir_list = os.listdir(folder_image_in)
	for dir_path in dir_list:
		make_dir(os.path.join(folder_image_ou, dir_path))

	# get list image
	images_path = sorted(glob.glob(os.path.join(folder_image_in, "*/*.jpg")))

	# draw each image
	for image_path_in in tqdm(images_path, desc=f"Draw {num_classes} classes"):

		basename       = os.path.basename(image_path_in)
		basename_noext = os.path.splitext(basename)[0]

		# read image
		img = cv2.imread(image_path_in)

		# check label exist
		label_path = glob.glob(os.path.join(folder_label_in, f"*/{basename_noext}.txt"))
		if len(label_path) > 0:  # if it exists, draw the image
			label_path = label_path[0]
			visualization_image(img, label_path, colors, labels_name)

		# write image
		image_path_ou = image_path_in.replace(folder_image_in, folder_image_ou)
		cv2.imwrite(image_path_ou, img)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
	"""Concatenate images of different heights horizontally
		Noi anh theo chieu ngang
		hconcat_resize_min([im1, im2, im1])
	"""
	h_min = min(im.shape[0] for im in im_list)
	im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
					  for im in im_list]
	return cv2.hconcat(im_list_resize)


def visualize_comparison(num_class_l = 7, num_class_r = 1):
	# folder in
	folder_root     = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/"
	folder_image_in = f"{folder_root}images/"
	folder_left     = f"{folder_root}labels_{num_class_l}_classes_filtered/"
	folder_right    = f"{folder_root}labels_{num_class_r}_classes_filtered/"
	video_name      = f"{num_class_l}_classes_vs_{num_class_r}_classes"
	folder_image_ou = f"{folder_root}visualization/"

	# Get list of classes
	classes_l = get_list_7_classses()
	classes_r = get_list_2_classses()

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# get list images
	images_path = sorted(glob.glob(os.path.join(folder_image_in, "*/*.jpg")))

	# define output video
	img_w   = 1920
	img_h   = 1080
	video_w = img_w * 2
	video_h = img_h
	fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
	fps     = 10
	video   = cv2.VideoWriter(os.path.join(folder_image_ou, f"{video_name}.mp4"), fourcc, fps, (video_w, video_h))

	# concatenate video
	for img_path in tqdm(images_path, desc=f"Draw comparison {num_class_l} vs {num_class_r}"):
		# Get label
		lbl_path_l = img_path.replace(folder_image_in, folder_left).replace(".jpg", ".txt")
		if not os.path.exists(lbl_path_l):
			continue

		lbl_path_r = img_path.replace(folder_image_in, folder_right).replace(".jpg", ".txt")
		if not os.path.exists(lbl_path_r):
			continue

		# read image
		img_ori = cv2.imread(img_path)
		img_l   = img_ori.copy()
		img_r   = img_ori.copy()

		# draw label
		visualization_image(img_l, lbl_path_l, colors, classes_l)
		visualization_image(img_r, lbl_path_r, colors, classes_r)

		# get image and draw
		img_re = hconcat_resize_min([img_l, img_r])
		cv2.resize(img_re, (video_w, video_h))

		# write video
		label = os.path.basename(img_path)
		tl = round(0.002 * (img_re.shape[0] + img_re.shape[1]) / 2) + 1
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c1 = (0, 50)
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.putText(img_re, label, (c1[0], c1[1] + c2[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
		video.write(img_re)

	cv2.destroyAllWindows()
	video.release()


# NOTE: FILTER BOUNDING BOX ---------------------------------------------------


def filter_boundingbox_video(video_name, labels):
	"""Main algorithm for filtering"""
	# NOTE: for video 007
	if video_name == "007":
		labels_temp = []
		img_w = 1920
		img_h = 1080
		for label in labels:
			x_center = float(label[1]) * img_w
			y_center = float(label[2]) * img_h
			if 45 < x_center < 615 and 54 < y_center < 100:
				pass
			else:
				labels_temp.append(label)
		labels = labels_temp

	if video_name == "039" or video_name == "058":
		labels_temp = []
		img_w = 1920
		img_h = 1080
		for label in labels:
			x_center = float(label[1]) * img_w
			y_center = float(label[2]) * img_h
			if 1377 < x_center < 1616 and 954 < y_center < 1017:
				pass
			else:
				labels_temp.append(label)
		labels = labels_temp
	return labels


def filter_boundingbox_videos(num_class):
	# Initialize parameter
	folder_root_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_{num_class}_classes/"
	folder_root_ou = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_{num_class}_classes_filtered/"

	# Get list directory (actually list video)
	list_dir       = os.listdir(folder_root_in)

	for dir_name in list_dir:
		folder_label_in = os.path.join(folder_root_in, dir_name)

		# check it is path of directory or not
		if not os.path.isdir(folder_label_in):
			continue

		# initialize values
		folder_label_ou = os.path.join(folder_root_ou, dir_name)
		labels_path     = sorted(glob.glob(os.path.join(folder_label_in, "*.txt")))

		# create directory
		make_dir(folder_label_ou)

		# Surf all label
		for label_path_in in tqdm(labels_path, desc=f"Process {dir_name} of {num_class} classes"):
			# Read label
			labels = []
			with open(label_path_in, 'r') as f_open:
				lines = f_open.readlines()
				for line in lines:
					labels.append([word for word in line.replace('\n', '').split(' ')])

			# filtering
			labels = filter_boundingbox_video(dir_name, labels)

			# Write label
			label_path_ou = label_path_in.replace(folder_root_in, folder_root_ou)
			with open(label_path_ou, "w") as f_write:
				for label in labels:
					f_write.write(f"{label[0]} {float(label[1]):.10f} {float(label[2]):.10f} "
								  f"{float(label[3]):.10f} {float(label[4]):.10f}\n")


# NOTE: CREATE CROP DATASET----------------------------------------------------


def group_label_many_into_crop_image(labels):
	"""

	Args:
		labels (np.ndarray): the ungrouped labels

	Returns:
		labels (np.ndarray) : the grouped labels
		motor_Drivers (dict): the drivers with their posiblle motor on the labels
	"""
	labels_temp    = []
	# store index of driver and passenger with motor
	motor_drivers  = {}
	# store label of driver and passenger with motor, in which we make sure motor has driver or passenger
	motor_labels   = {}

	# NOTE: surf all driver
	# find the motor the driver belong
	# if the driver in not on any motor, we convert it into motor also
	for index, label in enumerate(labels):
		list_motor   = []
		list_overlap = []

		if label[0] != 0:  # if it is not motor
			box1 = [
				label[1] - (label[3] / 2),
				label[2] - (label[4] / 2),
				label[1] + (label[3] / 2),
				label[2] + (label[4] / 2)
			]   # because of YOLO format

			for index_sub, label_sub in enumerate(labels):

				if index_sub != index:  # it is not the same object
					if label_sub[0] == 0:  # if it is motor
						box2 = [
							label_sub[1] - (label_sub[3] / 2),
							label_sub[2] - (label_sub[4] / 2),
							label_sub[1] + (label_sub[3] / 2),
							label_sub[2] + (label_sub[4] / 2)
						]
						overlap = find_intersect(box1, box2)
						if overlap > 0.0:  # if they overlap
							list_motor.append(index_sub)
							list_overlap.append(overlap)

			if len(list_motor) == 0:  # if the driver in not on any motor, we convert it into motor also
				labels_temp.append([ele if index_sub > 0 else 0 for index_sub, ele in enumerate(label)])
			else:  # add it into list for post process
				index_motor = list_motor[np.argmax(list_overlap)]  # get the index of motor in labels

				if index_motor not in motor_drivers:  # check do we have the motor in list
					motor_drivers[index_motor] = []
					motor_labels[index_motor]  = [labels[index_motor]]

				motor_drivers[index_motor].append(index)
				motor_labels[index_motor].append(label)

	# NOTE: group driver into motor
	for key, values in motor_drivers.items():
		# Get value from motor
		x_min, y_min, x_max, y_max = [], [], [], []
		x_min.append(labels[key][1] - (labels[key][3] / 2))
		y_min.append(labels[key][2] - (labels[key][4] / 2))
		x_max.append(labels[key][1] + (labels[key][3] / 2))
		y_max.append(labels[key][2] + (labels[key][4] / 2))

		# Get all value from driver
		for value in values:
			x_min.append(labels[value][1] - (labels[value][3] / 2))
			y_min.append(labels[value][2] - (labels[value][4] / 2))
			x_max.append(labels[value][1] + (labels[value][3] / 2))
			y_max.append(labels[value][2] + (labels[value][4] / 2))

		# find the proper value
		x_min = min(x_min)
		y_min = min(y_min)
		x_max = max(x_max)
		y_max = max(y_max)

		# add to final list
		labels_temp.append([
			0,
			(x_max + x_min) / 2,
			(y_max + y_min) / 2,
			abs(x_max - x_min),
			abs(y_max - y_min),
		])
		# 9999999 is class the cover of motor
		motor_labels[key].append([
			999999,
			(x_max + x_min) / 2,
			(y_max + y_min) / 2,
			abs(x_max - x_min),
			abs(y_max - y_min),
		])

	# NOTE: for any motor we have not checked so far
	# which mean the motor isolate and dont have any driver
	for index, label in enumerate(labels):
		if label[0] == 0:  # if it is motor
			if index not in motor_drivers:
				labels_temp.append(label)
				motor_drivers[index] = []

	return labels_temp, motor_drivers, motor_labels


def scaleup_bbox(bbox_xyxy, height_img, width_img, ratio, padding):
	"""Scale up 1.2% or +-40

	Args:
		bbox_xyxy:
		height_img:
		width_img:

	Returns:

	"""
	cx = 0.5 * bbox_xyxy[0] + 0.5 * bbox_xyxy[2]
	cy = 0.5 * bbox_xyxy[1] + 0.5 * bbox_xyxy[3]
	w = abs(bbox_xyxy[2] - bbox_xyxy[0])
	w = min(w * ratio, w + padding)
	h = abs(bbox_xyxy[3] - bbox_xyxy[1])
	h = min(h * ratio, h + padding)
	bbox_xyxy[0] = int(max(0, cx - 0.5 * w))
	bbox_xyxy[1] = int(max(0, cy - 0.5 * h))
	bbox_xyxy[2] = int(min(width_img - 1, cx + 0.5 * w))
	bbox_xyxy[3] = int(min(height_img - 1, cy + 0.5 * h))
	return bbox_xyxy


def voc_to_yolo_format(size, box):
	dw = 1. / (size[0])
	dh = 1. / (size[1])

	x = (box[0] + box[2]) / 2.0
	y = (box[1] + box[3]) / 2.0
	w = abs(box[2] - box[0])
	h = abs(box[3] - box[1])

	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh

	return x, y, w, h


def extract_cover_crop_image(img, motor_labels, img_w, img_h, basename_noext):
	imgs_labels = []
	index_motor = -1
	for key, values in motor_labels.items():
		objects = []

		# NOTE: get value of cover crop image
		for value in values:
			# get value of bounding box
			cls = int(value[0])
			bbox_xywhn = value[1:]
			bbox_xyxy = [
				int(bbox_xywhn[0] * img_w - (bbox_xywhn[2] * img_w / 2.0)),
				int(bbox_xywhn[1] * img_h - (bbox_xywhn[3] * img_h / 2.0)),
				int(bbox_xywhn[0] * img_w + (bbox_xywhn[2] * img_w / 2.0)),
				int(bbox_xywhn[1] * img_h + (bbox_xywhn[3] * img_h / 2.0))
			]
			if cls == 999999:  # if this is the cover
				bbox_xyxy = scaleup_bbox(bbox_xyxy, img_h, img_w, ratio=1.5, padding=40)
				crop_image = img[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
				cover_objs = {
					'class_id' : cls,
					'bbox'     : bbox_xyxy,
					'crop_img' : crop_image,
				}
			else:
				objects.append({
					'class_id': cls,
					'bbox'    : bbox_xyxy
				})

		# NOTE: create new label from cover image
		cover_h, cover_w, _ = cover_objs['crop_img'].shape
		labels              = []
		for obj in objects:
			bbox_xyxy  = [
				obj['bbox'][0] - cover_objs['bbox'][0],
				obj['bbox'][1] - cover_objs['bbox'][1],
				obj['bbox'][2] - cover_objs['bbox'][0],
				obj['bbox'][3] - cover_objs['bbox'][1],
			]
			x, y, w, h = voc_to_yolo_format((cover_w, cover_h), bbox_xyxy)
			labels.append([obj['class_id'], x, y, w, h])

		# NOTE: add to list motor with persons
		index_motor = index_motor + 1
		imgs_labels.append({
			'name'      : f"{basename_noext}{index_motor:06d}",
			'crop_img'  : crop_image,
			'labels'    : labels,
			'width_img' : cover_w,
			'height_img': cover_h
		})

	return imgs_labels


def create_crop_dataset():
	"""create the crop dataset 7 cropped classes
	"""
	# Initial parameter
	folder_image_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images/"
	folder_label_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_7_classes_filtered/"
	folder_ou       = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_7_classes_cropped_filtered/"

	# Get list label
	labels_path = sorted(glob.glob(os.path.join(folder_label_in, "*/*.txt")))

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# define output folder
	folder_label_ou      = os.path.join(folder_ou, "labels")
	folder_image_ou      = os.path.join(folder_ou, "images")
	folder_image_draw_ou = os.path.join(folder_ou, "images_draw")
	make_dir(folder_label_ou)
	make_dir(folder_image_ou)
	make_dir(folder_image_draw_ou)

	# Creating
	for label_path_in in tqdm(labels_path):
		basename        = os.path.basename(label_path_in)
		basename_noext  = os.path.splitext(basename)[0]
		img_path        = label_path_in.replace(folder_label_in, folder_image_in).replace(".txt", ".jpg")

		if not os.path.exists(img_path):
			continue

		img             = cv2.imread(img_path)
		img_h, img_w, _ = img.shape

		# Read label
		labels_temp = []
		with open(label_path_in, 'r') as f_open:
			lines = f_open.readlines()
			for line in lines:
				labels_temp.append([float(word) for word in line.replace('\n', '').split(' ')])

		# Group one label
		_, _, motor_labels = group_label_many_into_crop_image(labels_temp)

		# extract crop image and its labels
		imgs_labels = extract_cover_crop_image(img, motor_labels, img_w, img_h, basename_noext)

		for img_labels in imgs_labels:
			name_file = img_labels['name']
			crop_img  = img_labels['crop_img']
			labels    = img_labels['labels']

			# write image
			cv2.imwrite(os.path.join(folder_image_ou, f"{name_file}.jpg"), crop_img)

			# write label
			with open(os.path.join(folder_label_ou, f"{name_file}.txt"), 'w') as f_write:
				# print(labels)
				# print("****")
				for line in labels:
					f_write.write(
						f"{int(line[0])} {float(line[1]):.10f} {float(line[2]):.10f} {float(line[3]):.10f} {float(line[4]):.10f}\n")


# NOTE: CREATE AUGMENTATION DATASET----------------------------------------------


def create_multiscale_dataset():
	"""create the crop dataset 7 cropped classes with multiscale
	"""
	# Initial parameter
	folder_root     = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5"
	folder_image_in = f"{folder_root}/labels_7_classes_cropped_filtered/images/"
	folder_label_in = f"{folder_root}/labels_7_classes_cropped_filtered/labels/"
	folder_ou       = f"{folder_root}/labels_7_classes_cropped_filtered_multiscale/"
	ori_res         = 320.0
	new_res         = [128.0, 192.0, 256.0, 320.0, 384.0, 448.0, 512.0, 640.0]

	# Get list image
	images_path = sorted(glob.glob(os.path.join(folder_image_in, "*.jpg")))

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# define output folder
	folder_label_ou = os.path.join(folder_ou, "labels")
	folder_image_ou = os.path.join(folder_ou, "images")
	make_dir(folder_label_ou)
	make_dir(folder_image_ou)

	# Multiscaling
	for image_path_in in tqdm(images_path):
		basename       = os.path.basename(image_path_in)
		basename_noext = os.path.splitext(basename)[0]
		label_path_in  = image_path_in.replace(folder_image_in, folder_label_in).replace(".jpg", ".txt")

		if not os.path.exists(label_path_in):
			continue

		img             = cv2.imread(image_path_in)
		img_h, img_w, _ = img.shape

		# create multi scale
		for index, res in enumerate(new_res):
			img_path_new = os.path.join(folder_image_ou, f"{basename_noext}{index:03d}.jpg")
			lbl_path_new = os.path.join(folder_label_ou, f"{basename_noext}{index:03d}.txt")

			# create image
			img_temp = np.copy(img)
			img_temp = cv2.resize(
				img_temp,
				(int(float(img_w) / ori_res * new_res[index]),
				 int(float(img_h) / ori_res * new_res[index])),
				interpolation=cv2.INTER_CUBIC
			)
			cv2.imwrite(img_path_new, img_temp)

			# copy label
			shutil.copyfile(label_path_in, lbl_path_new)


def read_label(label_path):
	bboxes = []
	with open(label_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			bboxes.append([float(word) if index > 0 else int(word) for index, word in enumerate(line.replace('\n', '').split(' '))])

	return np.array(bboxes)


def show_bbox_yolo(img, labels) :
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches
	# Create figure and axes
	fig, ax = plt.subplots()

	# Display the image
	ax.imshow(img)

	for label in labels:
	# if len(labels) > 0 :
		# Create a Rectangle patch
		dw = img.shape[1]
		dh = img.shape[0]

		x1 = (label[0] - label[2] / 2) * dw
		y1 = (label[1] - label[3] / 2) * dh
		w = label[2] * dw
		h = label[3] * dh
		rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')

		# Add the patch to the Axes

		ax.add_patch(rect)
	plt.show()


def create_augmentation_dataset():
	"""create the crop dataset 7 cropped classes with augmentation
	"""
	# initial parameter
	folder_in        = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/1_classes/train/"
	folder_image_in  = f"{folder_in}/images/"
	folder_label_in  = f"{folder_in}/labels/"
	folder_ou        = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/1_classes_augmentation/train/"
	folder_image_ou  = f"{folder_ou}/images/"
	folder_label_ou  = f"{folder_ou}/labels/"
	number_new_image = 5

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# create directory
	make_dir(folder_image_ou)
	make_dir(folder_label_ou)

	# define augmentation
	transform = A.Compose([
		# A.Blur(blur_limit=50, p=0.1),
		A.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_CUBIC),
		A.MedianBlur(blur_limit=21, p=0.1),
		A.ToGray(p=0.3),
		A.RandomBrightnessContrast(p=0.5),
		A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1),
		A.ShiftScaleRotate(rotate_limit=20, p=1, border_mode=cv2.BORDER_CONSTANT),
		A.HorizontalFlip(p=1),
	],	bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

	# get list image
	images_path = sorted(glob.glob(os.path.join(folder_image_in, "*.jpg")))

	for image_path_in in tqdm(images_path):
		basename       = os.path.basename(image_path_in)
		basename_noext = os.path.splitext(basename)[0]
		label_path_in  = image_path_in.replace(folder_image_in, folder_label_in).replace(".jpg", ".txt")

		if not os.path.exists(label_path_in):
			continue

		img             = cv2.imread(image_path_in)
		img_h, img_w, _ = img.shape

		# copy original to new folder
		index_image    = 0
		image_path_new = os.path.join(folder_image_ou, f"{basename_noext}{index_image:04d}.jpg")
		label_path_new = os.path.join(folder_label_ou, f"{basename_noext}{index_image:04d}.txt")
		shutil.copyfile(image_path_in, image_path_new)
		shutil.copyfile(label_path_in, label_path_new)

		# run augmentation
		labels = read_label(label_path_in)

		if len(labels) < 1:
			continue

		for index_image in range(1, number_new_image):
			# bboxes = labels[:, 1:]
			# print(bboxes)
			# bboxes[bboxes > 1.0] = 1.0
			# print(bboxes)
			# sys.exit()
			# print(labels)
			# print(labels[:, 1:])
			# print(labels[:, 0])
			# print("***********")

			# transform
			transformed        = transform(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
			transformed_image  = transformed['image']
			transformed_bboxes = transformed['bboxes']

			# create new path
			image_path_new = os.path.join(folder_image_ou, f"{basename_noext}{index_image:04d}.jpg")
			label_path_new = os.path.join(folder_label_ou, f"{basename_noext}{index_image:04d}.txt")

			# write image
			cv2.imwrite(image_path_new, transformed_image)

			# write label
			with open(label_path_new, "w") as f_write:
				for cls, bbox in zip(labels[:, 0], transformed_bboxes):
					f_write.write(f"{int(cls)} {bbox[0]:.10f} {bbox[1]:.10f} {bbox[2]:.10f} {bbox[3]:.10f}\n")

			# DEBUG: run 1 transform
			# print(len(transformed_image))
			# print(len(transformed_bboxes))
			# print(transformed_image.shape)
			# print(transformed_bboxes)
			# print(labels)
			# show_bbox_yolo(transformed_image, transformed_bboxes)
			# break

		# DEBUG: run 1 image
		# break


# NOTE: MAIN ------------------------------------------------------------------


def main():
	# NOTE: extract image from video
	# extract_videos()

	# NOTE: extract label from one ground truth text file [7, 3, 2, 1]
	# extract_groundtruths(1)

	# NOTE: extract label from final result
	# extract_pickles()

	# NOTE: Filter bounding box
	# filter_boundingbox_videos(7)
	# filter_boundingbox_videos(3)
	# filter_boundingbox_videos(2)

	# NOTE: group class into only motor
	# group_labels_many_into_one()

	# NOTE: copyfile from many folders into one folder
	# copy_all_file(7)
	# copy_all_file(3)
	# copy_all_file(2)
	# copy_all_file(1)

	# NOTE: visualization by drawing ground truth
	# Get 00100147, 00200198 to comparison
	# visualization_images(7)
	# processes = []
	# # num_classes = [7, 3, 2, 1]   # draw many file at ones
	# num_classes = [2]   # draw many file at ones
	# # Define processes
	# for num_class in num_classes:
	# 	processes.append(multiprocessing.Process(target=visualization_images, args=([], num_class)))
	#
	# # Start processes
	# for process in processes:
	# 	process.start()
	#
	# # Wait all processes stop
	# for process in processes:
	# 	process.join()

	# NOTE: visualization for comparision between two label.
	# visualize_comparison(
	# 	num_class_l = 7,
	# 	num_class_r = 2
	# )

	# NOTE: create the crop dataset 7 cropped classes
	# create_crop_dataset()

	# NOTE: create the crop dataset 7 cropped classes with multiscale
	# create_multiscale_dataset()

	# NOTE: create the crop dataset 7 cropped classes with augmentation
	create_augmentation_dataset()
	pass


if __name__ == "__main__":
	main()

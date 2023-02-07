#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


# NOTE: EXTRACT GROUNTRUTH FILE------------------------------------------------


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
		cam = cv2.VideoCapture(video_path)
		width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
		height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
		cam.release()

		# get label for one video
		label = [line for line in labels if line[0] == int(basename_noext)]

		# Extract one video
		extract_groundtruth(basename_noext, label, folder_label_out, (width, height), num_class)


# NOTE: COPY FILES FROM MANY FOLDERS INTO TWO FOLDERS--------------------------


def copy_all_file(num_class):
	# NOTE: initial parameters
	image_root = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images/"
	label_root = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/labels_{num_class}_classes_filtered/"
	image_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/{num_class}_classes/val/images/"
	label_out  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/yolo_format/{num_class}_classes/val/labels/"

	# NOTE: create directory
	make_dir(image_out)
	make_dir(label_out)

	# NOTE: Get list
	list_label = glob.glob(os.path.join(label_root, "*/*.txt"))

	# NOTE: Copy all files
	for index, lbl_path in enumerate(tqdm(list_label)):

		# if index % 1 > 0:
		if index % 5 > 0:
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


# Concatenate images of different heights horizontally
#  Noi anh theo chieu ngang
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
	# hconcat_resize_min([im1, im2, im1])
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


# NOTE: MAIN ------------------------------------------------------------------


def main():
	# NOTE: extract image from video
	# extract_videos()

	# NOTE: extract label from one ground truthtext file [7, 3, 2, 1]
	# extract_groundtruths(1)
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

	# NOTE: visualization by drawing
	# # Get 00100147, 00200198 to comparison
	# visualization_images(7)
	# processes = []
	# num_classes = [7, 3, 2, 1]   # draw many file at ones
	# # num_classes = [7]   # draw many file at ones
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
	# pass

	# NOTE: visualization for comparision between two label.
	visualize_comparison(
		num_class_l = 7,
		num_class_r = 2
	)


if __name__ == "__main__":
	main()

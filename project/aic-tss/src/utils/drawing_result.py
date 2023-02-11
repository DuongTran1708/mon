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

from classes_aic23_track5 import *

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--draw_final", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--draw_pickle", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--path_final",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/final_result_s1.txt",
	help="Path to pickle folder."
)
parser.add_argument(
	"--path_pickle_in",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/dets_crop_pkl/yolov8x6/",
	help="Path to pickle folder."
)
parser.add_argument(
	"--path_video_in",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/videos/",
	help="Path to output folder."
)
parser.add_argument(
	"--path_video_out",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/dets_crop_pkl_debug/",
	help="Path to output folder."
)


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


# NOTE: VISUALIZE FINAL RESULT ------------------------------------------------


def read_result(gt_path):

	labels = []
	with open(gt_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			labels.append([int(word) for word in line.replace('\n', '').split(',')])
	return labels


def draw_final_video(video_path_in, video_path_ou, label_video, colors, labels_name):

	# Read original information of input video
	cam         = cv2.VideoCapture(video_path_in)
	width       = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
	height      = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
	fps         = cam.get(cv2.CAP_PROP_FPS)
	index_frame = 0

	# generate new video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(video_path_ou, fourcc, fps, (width, height))

	while True:
		index_frame = index_frame + 1

		# reading from frame
		ret, frame = cam.read()
		if ret:
			label_image = [label for label in label_video if label[1] == index_frame]
			if len(label_image) >  0:
				for label in label_image:
					cls_index = int(label[6])
					box = [
						label[2],
						label[3],
						label[2] + label[4],
						label[3] + label[5]
					]
					plot_one_box(
						x     = box,
						img   = frame,
						color = colors[cls_index],
						label = labels_name[cls_index - 1]
					)

			# writing the extracted images
			video.write(frame)
		else:
			break

	# Release all space and windows once done
	cam.release()
	video.release()


def draw_final_result(args):
	# initiate parameters
	folder_out = args.path_video_out
	folder_in  = args.path_video_in

	# create output folder
	make_dir(folder_out)

	# get result
	result_path = args.path_final
	labels = read_result(result_path)

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# get list of label
	labels_name = get_list_7_classses()

	# Get list video
	video_paths = sorted(glob.glob(os.path.join(folder_in, "*.mp4")))

	# draw videos
	for video_path in tqdm(video_paths, desc="Drawing final result"):
		basename = os.path.basename(video_path)
		basename_noext = os.path.splitext(basename)[0]
		video_index = int(basename_noext)
		video_path_ou = os.path.join(folder_out, basename)

		# draw one video
		label_video = [label for label in labels if label[0] == video_index]
		draw_final_video(video_path, video_path_ou, label_video, colors, labels_name)

		# DEBUG: run 1 video
		# break


# NOTE: VISUALIZE PICKLE RESULT -----------------------------------------------


def draw_pickle_video(video_path_in, video_path_ou, pkl_path, colors, labels_name):
	# read pkl
	dets_pkl = pickle.load(open(pkl_path, 'rb'))

	# read video in
	cam = cv2.VideoCapture(video_path_in)
	width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
	height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
	fps = cam.get(cv2.CAP_PROP_FPS)
	index_frame = 0

	# generate new video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video  = cv2.VideoWriter(video_path_ou, fourcc, fps, (width, height))

	while True:
		index_frame = index_frame + 1

		# reading from frame
		ret, frame = cam.read()
		if ret:
			label_image = [label for label in dets_pkl if int(label["frame_id"]) == index_frame]

			if len(label_image) >  0:
				for label in label_image:
					box = [
						label["bbox"][0],
						label["bbox"][1],
						label["bbox"][2],
						label["bbox"][3]
					]
					plot_one_box(
						x     = box,
						img   = frame,
						color = colors[int(label["id"])],
						label = labels_name[int(label["class_id"])]
					)

			# writing the extracted images
			video.write(frame)
		else:
			break

	# Release all space and windows once done
	video.release()
	cam.release()


def draw_pickles(args):
	# get the parameters
	path_pickle_in = args.path_pickle_in
	path_video_out = args.path_video_out
	path_video_in  = args.path_video_in

	# create output folder
	make_dir(path_video_out)

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# get list of label
	labels_name = get_list_7_classses()

	# get list pkl
	pkl_paths = sorted(glob.glob(os.path.join(path_pickle_in, "*.pkl")))

	# draw videos
	for pkl_path in tqdm(pkl_paths, desc="Drawing pickle result"):
		basename = os.path.basename(pkl_path)
		basename_noext = os.path.splitext(basename)[0]
		video_index = int(basename_noext)
		video_path_in = os.path.join(path_video_in, f"{basename_noext}.mp4")
		video_path_ou = os.path.join(path_video_out, f"{basename_noext}.mp4")

		# draw one video
		draw_pickle_video(video_path_in, video_path_ou, pkl_path, colors, labels_name)


# NOTE: MAIN ------------------------------------------------------------------


def main():
	args = parser.parse_args()
	if args.draw_final:
		draw_final_result(args)
	elif args.draw_pickle:
		draw_pickles(args)


if __name__ == "__main__":
	main()

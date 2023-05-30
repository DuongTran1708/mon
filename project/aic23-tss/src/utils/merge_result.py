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

# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--results",
	default=[],
	nargs='+',
	help="List of file result."
)


# MARK: - Args


def main():
	# NOTE: Parse
	args = parser.parse_args()

	print(f"Merging {args.results}")


if __name__ == "__main__":
	main()

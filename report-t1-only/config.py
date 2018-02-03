import numpy as np
from easydict import EasyDict as edict
from os import path as osp

config = edict()

# running mode: train or test
config.gtpath = '../data/ECODSEdatasetGT'

# Task 1
config.task1 = edict()
config.task1.minContourArea = 50  # minimum area, in pixel, for polygons.

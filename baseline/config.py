import numpy as np
from easydict import EasyDict as edict
from os import path as osp

config = edict()

# running mode: train or test
config.datapath = '../data/ECODSEdataset'
config.teamname = 'Baseline'
config.outputpath = '../income/%s' % config.teamname # output of baseline

# Task 1
config.task1 = edict()
config.task1.minContourArea = 10  # minimum area, in m*m, for polygons.
config.task1.maxContourArea = 10000  # maximum area, in m*m, for polygons.




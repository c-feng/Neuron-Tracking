import os
import sys
import _init_paths


import numpy as np
from lib.datasets.point_dataset import PointDataset
from utils.show3d_balls import showpoints, ncolors
import train_utils.train_utils as train_utils

data_root = "/home/fcheng/Neuron/pointnet/data/real_data/"
pointdata = PointDataset(data_root, mode="train", augment=True)
# pointdata_a = PointDataset(data_root, mode="train", augment=True)

for i in range(pointdata.__len__()):

    point_set, p_labels = pointdata.readData(i)
    
    print(pointdata.dataNames[i])
    print(point_set.shape, np.max(point_set), np.min(point_set), "\n")
    # labels = np.unique(p_labels)
    # print(labels)
    # colors = ncolors(len(labels))

    # p_labels_order = np.zeros(len(point_set), dtype=int)
    # for cnt, l in enumerate(labels):
    #     p_labels_order[p_labels==l] = cnt


    # showpoints(point_set, c_gt=colors[p_labels_order], normalizecolor=False, ballradius=2)
    # showpoints(point_set_a, c_gt=colors[p_labels_order], normalizecolor=False, ballradius=2)



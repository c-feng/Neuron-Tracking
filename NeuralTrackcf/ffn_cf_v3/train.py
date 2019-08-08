import os
import numpy as np
import argparse

import torch
import torch.nn as nn

TRAIN_DATA_PATH = "/media/fcheng/synthetic/SyntheticData/chaos/300_5_2_4_6/"
REAL_DATA_PATH = "/media/jjx/Biology/data/data_modified/"
INIT_WEIGHT = "/media/fcheng/NeuralTrackcf/logs/logs_9/model_best.pth.tar"
working_dir = osp.dirname(osp.abspath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description="FFN Network")

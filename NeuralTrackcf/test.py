import os
import os.path as osp
import sys
import numpy as np
from skimage.external import tifffile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inference import Inference, DF_Inference
from model import SegFFNNet
from dataset import DatasetTest, ToTensor, Compose, DatasetRealTest
from neuralTrack.utils.metrics import batch_multi_soft_metric, batch_soft_metric

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# OUTPUT_DIR = "eval/fov64_DF"
OUTPUT_DIR = "eval/realData/fov64_DF_finetune"
# OUTPUT_DIR = "eval/realData/fov64_DF"
TEST_DATA_PATH = "/media/fcheng/synthetic/SyntheticData/chaos/300_5_2_4_6/"
REAL_TEST_DATA_PATH = "/media/jjx/Biology/data/data_modified/"
working_dir = osp.dirname(osp.abspath(__file__))
# MODEL_PATH = osp.join(working_dir, "logs/logs_9/model_best.pth.tar")
# MODEL_PATH = osp.join(working_dir, "logs/fov64_DF/model_best.pth.tar")
# MODEL_PATH = osp.join(working_dir, "logs/realData/tryParallel/model_best.pth.tar")
# MODEL_PATH = osp.join(working_dir, "logs/realData/fov72/model_best.pth.tar")
# MODEL_PATH = osp.join(working_dir, "logs/realData/fov64_DF/model_best.pth.tar")
MODEL_PATH = osp.join(working_dir, "logs/realData/fov64_DF_finetune/model_best.pth.tar")
PATCH_SHAPE = [64, 64, 64]
# FOV_SHAPE = [41, 41, 41]
# FOV_SHAPE = [57, 57, 57]
FOV_SHAPE = [64, 64, 64]

device = torch.device("cuda")
path = osp.join(working_dir, OUTPUT_DIR)
mkdir_if_not_exist(path)

# model = SegFFNNet(in_dims=1)
model = SegFFNNet(in_dims=1, ins_dims=5)
model = model.to(device)

# infer = Inference(model, MODEL_PATH, PATCH_SHAPE, FOV_SHAPE, path)
infer = DF_Inference(model, MODEL_PATH, PATCH_SHAPE, FOV_SHAPE, path)

t = [ToTensor()]
# dataset = DatasetTest(TEST_DATA_PATH, transform=Compose(t))
dataset = DatasetRealTest(REAL_TEST_DATA_PATH, transform=Compose(t))

for i in range(1):
    img, gt = dataset[i]
    name = dataset.datanames[i]
    print(img.shape, gt.shape)

    print("Processing {} ...".format(name))
#     ins_mask = infer.forward(np.squeeze(img.cpu().numpy()))
    ins_mask = infer.forward(np.squeeze(img.cpu().numpy()), np.squeeze(gt.cpu().numpy()))
#     ins_mask = infer.forward_gt(np.squeeze(img.cpu().numpy()), np.squeeze(gt.cpu().numpy()))

    tifffile.imsave(osp.join(path, name+"_pred.tif"), ins_mask[:300, :300, :300].astype(np.float16))

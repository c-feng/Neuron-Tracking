from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# 0. basic config
__C.TAG = 'default'


# general training config
__C.TRAIN = edict()

# __C.TRAIN.LR = 0.002
__C.TRAIN.LR = 0.5
__C.TRAIN.LR_CLIP = 0.00001
# __C.TRAIN.DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]
__C.TRAIN.DECAY_STEP_LIST = [300, 500, 800]
__C.TRAIN.LR_DECAY = 0.5
__C.TRAIN.LR_WARMUP = True
__C.TRAIN.WARMUP_MIN = 0.0002
__C.TRAIN.WARMUP_EPOCH = 5

__C.TRAIN.BN_MOMENTUM = 0.9
__C.TRAIN.BN_DECAY = 0.5
__C.TRAIN.BNM_CLIP = 0.01
# __C.TRAIN.BN_DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]
__C.TRAIN.BN_DECAY_STEP_LIST = [300, 500, 800]

__C.TRAIN.OPTIMIZER = 'sgd'
__C.TRAIN.WEIGHT_DECAY = 0.0  # "L2 regularization coeff [default: 0.0]"
__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.MOMS = [0.95, 0.85]
__C.TRAIN.DIV_FACTOR = 10.0
__C.TRAIN.PCT_START = 0.4

__C.TRAIN.GRAD_NORM_CLIP = 1.0

# Loss param
__C.LOSS = edict()
__C.LOSS.DELTA_V = 0.05
__C.LOSS.DELTA_D = 1.5
__C.LOSS.PARAM_VAR = 1.
__C.LOSS.PARAM_DIST = 1.
__C.LOSS.PARAM_REG = 0.001

# Net
__C.NET = edict()
__C.NET.SA_NPOINTS = [16384, 4096, 1024, 256]
__C.NET.RADIIS = [[0.05, 0.1, 0.5], [0.1, 0.5, 1.], [0.5, 1.0, 2.0], [1.0, 2.0, 4.0]]
__C.NET.NSAMPLES = [[16, 16, 32], [16, 16, 32], [16, 16, 32], [16, 16, 32], [16, 16, 32]]
__C.NET.MLPS = [[[16, 16, 32], [16, 16, 32], [32, 32, 64]],
                [[64, 64, 128], [64, 64, 128], [64, 96, 128]],
                [[128, 196, 256], [128, 196, 256], [128, 196, 256]],
                [[256, 256, 512], [256, 256, 512], [256, 384, 512]] ]
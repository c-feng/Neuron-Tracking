import os
import os.path as osp
import sys
import numpy as np
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import horovod.torch as hvd

from neuralTrack.utils.serialization import load_checkpoint, save_checkpoint, read_json
from neuralTrack.loss.loss_utils import VoxLoss
from neuralTrack.loss.diceLoss import dice_cross_entropy_loss
from neuralTrack.utils.metrics import batch_multi_soft_metric, batch_soft_metric
from neuralTrack.utils.logging import Logger, TFLogger
from neuralTrack.utils.serialization import load_checkpoint, save_checkpoint, read_json

from model import SegFFNNet
from trainer import SegFFNNetTrainer
from evaluator import SegFFNNetEvaluator
from dataset import DatasetTrain, Compose, ToTensor, DatasetEvaluate, DatasetReal
from utils.transcheckpoint import transcheckpoint

TRAIN_DATA_PATH = "/media/fcheng/synthetic/SyntheticData/chaos/300_5_2_4_6/"
REAL_DATA_PATH = "/media/jjx/Biology/data/data_modified/"
INIT_WEIGHT = "/media/fcheng/NeuralTrackcf/logs/logs_9/model_best.pth.tar"
working_dir = osp.dirname(osp.abspath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"]="4,6"

parser = argparse.ArgumentParser(description="Softmax loss classification")
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                                 default=osp.join(working_dir, 'logs/realData/tryParallel'))
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--init', type=str, default=INIT_WEIGHT, metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--train_data', type=str, default=TRAIN_DATA_PATH)
parser.add_argument('--eval_data', type=str, default=TRAIN_DATA_PATH)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--fov_shape', type=list, default=[41,41,41])
parser.add_argument("--sample_shape", type=list, default=[72, 72, 72])

args = parser.parse_args()

device = torch.device("cuda")

best_res = 0

# Horovod: initialize library
hvd.init()

# Horovod: pin GPU to local rank.
torch.cuda.set_device(hvd.local_rank())

model = SegFFNNet(in_dims=1)
model = model.to(device)

if args.init:
    # checkpoint = load_checkpoint(args.init)
    checkpoint = transcheckpoint(args.init)
    print("lens of checkpoint:", len(checkpoint['state_dict']))
    model.load_state_dict(checkpoint['state_dict'])

if args.resume:
    # checkpoint = load_checkpoint(args.resume)
    checkpoint = transcheckpoint(args.init)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    args.start_epoch = start_epoch
    best_res = checkpoint['best_res']
    print("=> Start epoch {}  best res {:.1%}"
          .format(start_epoch, best_res))

param_groups = model.parameters()
param_groups = filter(lambda p: p.requires_grad, param_groups)
# Horovod: scale learning rate by the number of GPUs.
optimizer = torch.optim.RMSprop(param_groups, lr = args.lr * hvd.size(), \
        alpha = 0.9, eps = 1e-4, weight_decay = 0.0001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,25,35,50,70],gamma = 0.5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[25,40,50,75,105],gamma = 0.5)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, name_parameters=model.named_parameters())


criterion_func = partial(dice_cross_entropy_loss, rate = 0.5)
criterion = VoxLoss(criterion_func, with_weight = True)
metric_func = partial(batch_multi_soft_metric, num_classes = 2, bg = 0, thres = 1) 

trainer = SegFFNNetTrainer(model, criterion, args.logs_dir, metric_func, args.fov_shape, device=device)
evaluator = SegFFNNetEvaluator(model, criterion, metric_func, args.fov_shape, device=device)

sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
train_tfLogger = TFLogger(osp.join(args.logs_dir, 'train'))
eval_tfLogger = TFLogger(osp.join(args.logs_dir, 'eval'))


# t = [ToTensor()]
# train_data = DatasetTrain(args.train_data, transform=Compose(t), sample_shape=args.sample_shape)
# # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
# #                           num_workers=args.num_workers)

# eval_data = DatasetEvaluate(args.eval_data, transform=Compose(t), sample_shape=args.sample_shape)
# eval_data.reset_Sample()
# eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=True,
#                           num_workers=args.num_workers)

t = [ToTensor()]
train_data = DatasetReal(REAL_DATA_PATH, mode="train", transform=Compose(t), sample_shape=args.sample_shape)
# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
#                           num_workers=args.num_workers)
train_data.reset_Sample()
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=hvd.size(),
                                                                rank=hvd.rank())

eval_data = DatasetReal(REAL_DATA_PATH, mode="eval", transform=Compose(t), sample_shape=args.sample_shape)
eval_data.reset_Sample()
eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data, num_replicas=hvd.size(),
                                                               rank=hvd.rank())
eval_loader = DataLoader(eval_data, batch_size=1, sampler=eval_sampler,
                          num_workers=2)


for epoch in range(args.start_epoch, 150):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    
    train_data.reset_Sample()
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers)
    trainer.train(epoch, train_loader, optimizer, lr, 1, train_tfLogger)

    if (epoch+1)%2 ==0:
        print("now is tesing the model on the eval data:")
        step = len(train_loader) *(epoch+1)
        res = evaluator.evaluate(eval_loader,step,1,eval_tfLogger)

        is_best = res > best_res
        best_res = max(res,best_res)
        save_checkpoint({
                        #'state_dict': model.module.state_dict(),
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'best_res': best_res,},
                        is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        print('\n * Finished epoch {:3d} ! res now is: {:5.1%}  best: {:5.1%}{}\n'. 
              format(epoch, res, best_res, ' *' if is_best else ''))

train_tfLogger.close()
eval_tfLogger.close()
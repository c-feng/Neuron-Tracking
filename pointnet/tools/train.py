import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import os
import argparse
import logging

import _init_paths
# from lib.net.neural_ins import NeuralPointNet
from lib.net.neural_ins_syn import NeuralPointNet
import lib.net.train_functions as train_functions
from lib.datasets.point_dataset import PointDataset
from lib.config import cfg
# from lib.config_one_data import cfg
import train_utils.train_utils as train_utils

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--batch_size", type=int, default=4, required=False, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=False, help="Number of epochs to train for")
parser.add_argument('--workers', type=int, default=16, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', type=str, default=None, help='whether to use multiple gpu')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument('--train_with_eval', action='store_true', default=False, help='whether to train with evaluation')
args = parser.parse_args()

if args.mgpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def create_dataloader(logger):
    # DATA_PATH = "/home/fcheng/Neuron/pointnet/data/real_data_FPS_15000/"
    DATA_PATH = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/"

    # create dataloader
    train_set = PointDataset(data_root=DATA_PATH, mode="train", augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch)

    if args.train_with_eval:
        test_set = PointDataset(data_root=DATA_PATH, mode="test", augment=False)
        test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                                 num_workers=args.workers, shuffle=True, collate_fn=test_set.collate_batch)
    else:
        test_loader = None
    
    return train_loader, test_loader

def create_optimizer(model):

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError

    return optimizer

def create_scheduler(model, optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)


    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler

def train():
    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    
    # tensorboard log
    tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard'))

    # create dataloader & network & optimizer
    train_loader, test_loader = create_dataloader(logger)
    model = NeuralPointNet(input_channels=0)
    optimizer = create_optimizer(model)

    if args.mgpus is not None and len(args.mgpus) > 2:
        model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        it, start_epoch = train_utils.load_checkpoint(pure_model, optimizer, filename=args.ckpt, logger=logger)
        last_epoch = start_epoch + 1

    lr_scheduler, bnm_scheduler = create_scheduler(model, optimizer, total_steps=len(train_loader)*args.epochs, 
                                                   last_epoch=last_epoch)

    if cfg.TRAIN.LR_WARMUP and cfg.TRAIN.OPTIMIZER != 'adam_onecycle':
        lr_warmup_scheduler = train_utils.CosineWarmupLR(optimizer, T_max=cfg.TRAIN.WARMUP_EPOCH * len(train_loader),
                                                      eta_min=cfg.TRAIN.WARMUP_MIN)
    else:
        lr_warmup_scheduler = None
    
    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(
                model,
                train_functions.model_fn_neural_ins_decorator(),
                optimizer,
                ckpt_dir=ckpt_dir,
                lr_scheduler=lr_scheduler,
                bnm_scheduler=bnm_scheduler,
                model_fn_eval=train_functions.model_fn_neural_ins_decorator(),
                tb_log=tb_log,
                eval_frequency=5,
                lr_warmup_scheduler=lr_warmup_scheduler,
                warmup_epoch=cfg.TRAIN.WARMUP_EPOCH,
                grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP
                )

    trainer.train(
            it,
            start_epoch,
            args.epochs,
            train_loader,
            test_loader,
            ckpt_save_interval=args.ckpt_save_interval,
            lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle')
    )

    logger.info('**********************End training**********************')

if __name__ == "__main__":
    train()



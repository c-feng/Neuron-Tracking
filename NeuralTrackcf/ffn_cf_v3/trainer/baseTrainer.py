import time
import os
import numpy as np

import torch

from ..utils.meters import AverageMeter
from ..utils.vis_utils import trans3Dto2D

class BaseTrainer():
    def __init__(self, model, criterion, metric_func=None, logs_path=None, device="cuda"):
        self.device = torch.device(device)
        self.model = model
        self.criterion = criterion
        self.metric_func = metric_func
        self.logs_path = logs_path

    def train(self, epoch, data_loader, optimizer, current_lr=0., print_freq=1, tfLogger=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for i, input in enumerate(data_loader):
            start = time.time()

            img, gts = self._parse_data(input)
            batch_size = img.size(0)

            data_time.update(time.time() - start)

            loss, metrics, vis = self._forward(img, gts)
            
            dices,precs,recalls = metrics
            losses.update(loss.item(),batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start)

            start = time.time()
            if (i + 1) % print_freq == 0:
                print('epoch:[{}][{}/{}]  '
                      'batch_time:{:.3f}s({:.3f})  '
                      'data:{:.3f}s({:.3f})  '
                      'Loss:{:.3f}({:.3f})'
                    .format(epoch,i+1,len(data_loader),
                        batch_time.val,batch_time.avg,
                        data_time.val,data_time.avg,
                        losses.val,losses.avg,
                        ))
            
            if (i+1) % (print_freq*10) == 0:
                if tfLogger is not None:
                    step = epoch*len(data_loader) + (i+1)
                    infos = {"lr":current_lr,
                             "loss":loss.item()}
                    
                    for dice in dices:
                        infos[dice.infos] = dice.avg
                    for prec in precs:
                        infos[prec.infos] = prec.avg
                    for recall in recalls:
                        infos[recall.infos] = recall.avg

                    for tag,value in infos.items():
                        tfLogger.scalar_summary(tag, value, step)
            
            if (i+1) % (print_freq*20) == 0:
                if tfLogger is not None:
                    vis_info = {}
                    vis_info["full_seg"] = vis[0]
                    vis_info["full_seg_gt"] = vis[1]
                    vis_info["fov_seg"] = vis[2]
                    vis_info["fov_seg_gt"] = vis[3]

                    for tag, value in vis_info.items():
                        tfLogger.image_summary(tag, trans3Dto2D(value), step)

        print("Training Finished in Epoch {}".format(epoch))
        
        for dice, prec, recall in zip(dices, precs, recalls):
            print("{} {:.2%} {} {:.2%} {} {:.2%}".format(\
                    dice.infos, dice.avg,\
                    prec.infos, prec.avg,\
                    recall.infos, recall.avg))
        print("\n")
        
    def _parse_data(self, input):
        NotImplemented
    
    def _forward(self, img, gts):
        NotImplemented


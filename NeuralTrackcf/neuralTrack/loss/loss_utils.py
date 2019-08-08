import numpy as np

from skimage.morphology import dilation, cube, ball
from skimage.measure import label

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diceLoss import dice_loss, dice_cross_entropy_loss


class VoxLoss(nn.Module):
    def __init__(self,criterion_func ,weight=None, with_weight = True):
        super(VoxLoss,self).__init__()
        self.weight = weight
        self.with_weight = with_weight
        self.criterion_func = criterion_func

    def forward(self,input,target):
        device = input.device
        weight = self.weight

        if self.with_weight:
            if weight is not None:
                weight = torch.Tensor(weight).to(device)
            else:
                num_classes = input.size(1)
                weight = [1 -  (target == i).float().sum() / target.numel()  + 1e-3 for i in range(num_classes) ]
                #print(weight)
                weight = torch.Tensor(weight).to(device)
            loss = self.criterion_func(input, target, weight)
        else:
            loss = self.criterion_func(input, target)

        return loss 

class EndsLoss(nn.Module):
    def __init__(self,criterion_func ,weight=None, with_weight = True):
        super(EndsLoss,self).__init__()
        self.weight = weight
        self.with_weight = with_weight
        self.criterion_func = criterion_func

    def forward(self,input,target):
        device = input.device
        weight_ends = self.weight
        
        input_ends = input
        target_ends = target

        vox_mask = target > 0

        input_ends_sel = input_ends.transpose(0,1).\
                            masked_select(vox_mask).\
                            reshape(input_ends.size(1),-1).\
                            unsqueeze(0)
        target_ends_sel = target_ends.\
                            masked_select(vox_mask).\
                            unsqueeze(0) 

        if self.with_weight:
            if weight_ends is not None:
                weight_ends = torch.Tensor(weight_ends).to(device)

            else:
                num_classes = input_ends.size(1)
                weight_ends = [ 1 - (target_ends_sel == i).float().sum() / target_ends_sel.numel() + 1e-5 \
                        for i in range(num_classes) ]
                weight_ends = torch.Tensor(weight_ends).to(device)
            loss_ends = self.criterion_func(input_ends_sel, target_ends_sel, weight_ends)

        else:

            loss_ends = self.criterion_func(input_ends_sel, target_ends_sel)
        return loss_ends

class VoxEndsLoss(nn.Module):
    def __init__(self, vox_criterion, ends_criterion, weight_vox=None, \
            weight_ends=None, with_weight = True):
        super(VoxEndsLoss,self).__init__()

        self.weight_vox = weight_vox
        self.weight_ends = weight_ends
        self.with_weight = with_weight

        self.vox_criterion = vox_criterion
        self.ends_criterion = ends_criterion

    def forward(self, input_vox, input_ends, target_vox, target_ends):
        device = input_vox.device
        weight_vox = self.weight_vox
        weight_ends = self.weight_ends
        
        #_, vox_mask = torch.max(input_vox, 1) 
        vox_mask = target_vox > 0

        input_ends_sel = input_ends.transpose(0,1).\
                            masked_select(vox_mask).\
                            reshape(input_ends.size(1),-1).\
                            unsqueeze(0)
        target_ends_sel = target_ends.\
                            masked_select(vox_mask).\
                            unsqueeze(0) 
        #print(target_ends_sel.shape)
        if self.with_weight:
            if weight_vox is not None:
                weight_vox = torch.Tensor(weight_vox).to(device)
            else:
                num_classes = input_vox.size(1)
                weight_vox = [ 1 - (target_vox == i).float().sum() / target_vox.numel() + 1e-5 \
                        for i in range(num_classes) ]
                weight_vox = torch.Tensor(weight_vox).to(device)

            if weight_ends is not None:
                weight_ends = torch.Tensor(weight_ends).to(device)
            else:
                num_classes = input_ends.size(1)
                
                weight_ends = [ 1 - (target_ends_sel == i).float().sum() / target_ends_sel.numel() + 1e-5 \
                        for i in range(num_classes) ]
                weight_ends = torch.Tensor(weight_ends).to(device)

            #print(weight_vox)
            #print(weight_ends)
            loss_vox = self.vox_criterion(input_vox, target_vox, weight_vox)
            #loss_ends = self.ends_criterion(input_ends, target_ends, weight_ends)
            #print(weight_ends)

            loss_ends = self.ends_criterion(input_ends_sel, target_ends_sel, weight_ends)
            #loss_ends = self.ends_criterion(input_ends_sel, target_ends_sel)
        else:
            loss_vox = self.vox_criterion(input_vox, target_vox)
            loss_ends = self.ends_criterion(input_ends_sel, target_ends_sel)

        #loss = loss_vox
        #loss = loss_ends
        loss = loss_vox + loss_ends
        #print("loss_vox", loss_vox)
        #print("loss_ends", loss_ends)
        return loss

class VoxEndsEmbeddingLoss(nn.Module):
    def __init__(self, vox_criterion, ends_criterion, emb_criterion, weight_vox=None, \
            weight_ends=None, with_weight = True, region_thres = 20):
        super(VoxEndsEmbeddingLoss,self).__init__()

        self.weight_vox = weight_vox
        self.weight_ends = weight_ends
        self.with_weight = with_weight

        self.emb_criterion = emb_criterion

        self.voxEndsLoss = VoxEndsLoss(vox_criterion, ends_criterion, \
                weight_vox, weight_ends, with_weight)

    def forward(self, input_vox, input_ends, input_emb, input_seeds, \
            target_vox, target_ends, target_ins):

        device = input_vox.device
        voxEnds_loss = self.voxEndsLoss(input_vox, input_ends, target_vox, target_ends)
        
         



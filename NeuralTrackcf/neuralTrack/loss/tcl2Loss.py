import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

import numpy as np
from skimage.morphology import dilation, ball


def calculate_means(pred, gt, conf = None):
    """pred: f * h * w * d
       gt: h * w * d
       conf: h * w* d or None"""
    ins = torch.unique(gt[gt > 0])
    #print(ins)
    means = []
    #print(pred.size())
    for i in ins:
        n_loc = torch.masked_select(pred, gt == i).reshape(pred.size(0), -1)
        if conf == None:
            mean = torch.mean(n_loc, dim = 1)
        else:
            n_loc_conf = torch.masked_select(conf, gt == i)
            mean = torch.sum(n_loc_conf[None] * n_loc, dim = 1) / torch.sum(n_loc_conf)
        means.append(mean)
    means = torch.stack(means, 0)#means n_ins * f
    return means, ins

def tcl_loss(inputs, targets, margin = 5, confs = None, norm = 2):
    """ inputs: B * f * h * w * d
        targets: B * h * w * d
    """
    loss_ = 0.0
    if confs == None:
        for input, target in zip(inputs, targets):
            loss_ += calculate_n_ins_tcl(input, target, margin, None, norm)
    else: 
        for input, target, conf in zip(inputs, targets, confs):
            loss_ += calculate_n_ins_tcl(input, target, margin, conf, norm)
        
    loss_ = loss_ / inputs.size(0)
    return loss_

def calculate_n_ins_tcl(pred, gt, margin, conf = None, norm = 2):
    """ preds: f * h * w * d
        gt: h * w * d
    """
    loss = 0.0
    means, ins = calculate_means(pred, gt, conf)
    n_ins = len(ins)

    if n_ins == 1:
        mask_n = gt == 0
        mean_n = torch.mean(torch.masked_select(pred, mask_n).reshape(pred.size(0), -1), dim = 1)[:, None]# f * 1

        #mask_n_array = np.logical_and(mask_p_array == 0, dilation(mask_p_array > 0, ball(3)))
        #mask_n = torch.from_numpy(mask_n_array.astype(np.uint8)).to(mask_p.device)
        #mean_n = torch.mean(torch.masked_select(pred, mask_n).\
        #        reshape(pred.size(0), -1), dim = 1)[:,None]

        mask_p = gt > 0
        loss += calculate_tcl(pred, mask_p, means[0], mean_n, margin, conf, norm)
        return loss

    for i, label in enumerate(ins):
        mask_p = gt == label
        #mask_n = (gt > 0) & (gt != i)
        mean_p = means[i]
        mean_n = torch.cat([means[:i], means[i+1:]], dim = 0).t()#f * (n - 1)
        loss += calculate_tcl(pred, mask_p, mean_p, mean_n, margin, conf, norm)        
    return loss/ n_ins

def calculate_tcl(pred, mask, mean_p, mean_n, margin, conf = None, norm = 2):
    """ mask: h * w * d
        pred: f * h * w * d 
        mean_p: f
        mean_n: f * (n - 1)
        conf: h * w * d or None
    """
    n_loc = torch.masked_select(pred, mask).reshape(pred.size(0), -1)#f * n_loc
    
    dist_p = torch.norm(n_loc - mean_p[:, None], norm, 0)
    dist_n = torch.norm(n_loc[:,:,None] - mean_n[:, None], norm, 0)#n_loc * (n - 1)
    dist_n_min, _ = torch.min(dist_n, dim = 1)#f * n_loc
    if conf == None:
        loss = torch.mean(torch.clamp(dist_p + margin - dist_n_min, min = 0.0 ))
    else:
        n_loc_conf = torch.masked_select(conf, mask) # n_loc
        loss = torch.sum(n_loc_conf[None] * \
                    torch.clamp(dist_p + margin - dist_n_min, min = 0.0 )) / \
                    torch.sum(n_loc_conf)
    return loss



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

def tcl_loss(inputs, targets, confs = None, margin = 5, norm = 2):
    """ inputs: B * f * h * w * d
        targets: B * h * w * d
    """
    loss_ = 0.0
    if confs == None:
        for input, target in zip(inputs, targets):
            loss_ += calculate_n_ins_tcl(input, target, None, margin, norm)
    else: 
        for input, target, conf in zip(inputs, targets, confs):
            loss_ += calculate_n_ins_tcl(input, target, conf, margin, norm)
        
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
        mask_p = gt == ins[0]
        mask_p_array = mask_p.cpu().numpy()
        mask_n_array = np.logical_and(mask_p_array == 0, dilation(mask_p_array > 0, ball(3)))
        mask_n = torch.from_numpy(mask_n_array.astype(np.uint8)).to(mask_p.device)
        loss += calculate_tcl(pred, mask_p, mask_n, means[0], margin, norm)
        return loss

    for i, mean in zip(ins, means):
        mask_p = gt == i
        mask_n = (gt > 0) & (gt != i)
        loss += calculate_tcl(pred, mask_p, mask_n, mean, margin, norm)        
    return loss/ n_ins

def calculate_tcl(pred, mask_p, mask_n, mean, margin, norm = 2):
    """ mask_p: h * w * d
        mask_n: h * w * d
        pred: f * h * w * dxy
        mean: f
    """
    loc_p = torch.masked_select(pred, mask_p).reshape(pred.size(0), -1)#f * loc_p
    loc_n = torch.masked_select(pred, mask_n).reshape(pred.size(0), -1)#f * loc_n
    
    dist_p = torch.norm(loc_p - mean[:, None], norm, 0)
    dist_n = torch.norm(loc_p - mean[:, None], norm, 0)
    
    dist_p_max = torch.max(dist_p)
    dist_n_min = torch.min(dist_n)
    loss = torch.clamp(dist_p_max + margin - dist_n_min, min = 0.0 )
    return loss



'''def calculate_distance(pred, target, means, ins, margin, norm=2):
    """ means: (n_ins, f)
    """
    dist = 0.
    for idx, i in enumerate(ins):
        n_loc = torch.masked_select(pred, target==i).reshape(pred.size(0), -1)
        d_i = 0.
        for j in range(n_loc.size(1)):
            d = torch.norm(n_loc[:, j][:, None] - torch.t(means), norm, 0) ** 2
            d_p = d[idx]
            mask = torch.ones(means.size(0))
            mask[idx] = 0
            d_n = torch.masked_select(d, mask>0)
            if d_n.numel():
                d_n = torch.min(d_n)
            else: d_n = 0.
            d_i += torch.clamp(d_p + margin - d_n, min=0.)
        dist = dist + d_i

    return dist

def cal_tclLoss(input, target, margin):
    """ inputs: f x h x w x d
        targets: h x w x d
    """
    means, ins = calculate_means(input, target)
    loss = calculate_distance(input, target, means, ins, margin)
    return loss


class TCLoss(nn.Module):
    def __init__(self, margin=5):
        super(TCLoss, self).__init__()
        self._margin = margin
    
    def forward(self, inputs, targets):
        """ inputs: N x f x h x w x d
            targets: N x h x w x d
        """
        loss_ = 0.
        for input, target in zip(inputs, targets):
            loss_ += cal_tclLoss(input, target, self._margin)
        
        loss_ = loss_ / inputs.size(0)
        return loss_'''

if __name__ == "__main__":
    a = torch.randn((1, 64, 30000))
    b = torch.randint(1, 10, size=(1, 30000,))
    # a = torch.Tensor([[0, 1, 0],
    #                   [1, 2, 0]])
    # b = torch.IntTensor([1, 2, 1])
    print(a)
    print(b)
    # c = cal_tclLoss(a, b, 2)
    cter = TCLoss(margin=2)
    c = cter(a, b)
    print(c)
    print("-"*10)

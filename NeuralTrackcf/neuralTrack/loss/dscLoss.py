import numpy as np

import torch
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

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

def calculate_variance_term(pred, gt, ins, means, delta_v, conf = None, norm=2):
    """pred: f * h * w * d
       gt: h * w * d
       means: n_ins * f
       conf: h * w * d or None"""
    var_term = 0.0
    for i, mean in zip(ins, means):
        n_loc = torch.masked_select(pred, gt == i).reshape(pred.size(0), -1)
        if conf == None:
            _var = torch.mean(\
                    torch.clamp(torch.norm(n_loc - mean[:, None], norm, 0) - delta_v, min = 0.0)**2)
        else:
            n_loc_conf = torch.masked_select(conf, gt == i)
            _var = torch.sum(n_loc_conf[None] * torch.clamp(\
                        torch.norm(n_loc - mean[:, None], norm, 0) - delta_v, min = 0.0)**2, dim = 1)/\
                        torch.sum(n_loc_conf)

        var_term += _var
    return var_term / len(ins)

def calculate_distance_term(means, delta_d, norm = 2):
    """means: n_ins * f"""
    dist_term = 0.0
    n_ins = means.size(0)

    if  n_ins <= 1:return dist_term
    means_1 = means[:,None]
    means_2 = means[None]

    diff = means_1 - means_2 # n * n * f
    _norm = torch.norm(diff, norm, 2)
    
    margin = 2 * delta_d * (1.0 - torch.eye(means.size(0)))
    margin = margin.to(means.device)

    dist_term = torch.sum(
            torch.clamp(margin - _norm, min=0.0)**2
            )
    dist_term = dist_term/(n_ins * (n_ins - 1))
    return dist_term

def calculate_regularization_term(means, norm):
    """means: n_instances, n_filters"""
    #reg_term = 0.0
    reg_norm = torch.mean(torch.norm(means, norm, 1))
    return reg_norm

def calculate_discriminative_loss(pred, gt, delta_v, delta_d, conf = None, norm = 2):
    alpha = beta = 1.0
    gamma = 0.001
    
    means, ins = calculate_means(pred, gt, conf)
    var_term = calculate_variance_term(pred, gt, ins, means, delta_v, conf, norm)
    dist_term = calculate_distance_term(means, delta_d, norm)
    reg_term = calculate_regularization_term(means, norm)
    
    loss = alpha * var_term + beta * dist_term + gamma * reg_term  
    return loss

def dsc_loss(inputs, targets, confs = None, delta_v = 0.5, delta_d = 2, norm = 2):
    """inputs: B * f * h * w * d
        targets: B * h * w * d
        confs: B * h * w * d or None"""
    loss_ = 0.0
    if confs == None:
        for input, target in zip(inputs, targets):
            loss_ += calculate_discriminative_loss(input, target, delta_v, delta_d, None, norm)
    else:
        for input, target, conf in zip(inputs, targets, confs):
            loss_ += calculate_discriminative_loss(input, target, delta_v, delta_d, conf, norm)

    loss_ = loss_ / inputs.size(0)
    return loss_

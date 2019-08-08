import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MultiLabelBinarizer

def dice_loss_cal(input, target, weight = None):
    '''input B * Num Classes * H * W * D
        target B * H * W * D '''
    smooth = 1.
    loss = 0.
    n_classes = input.size(1)
    for c in range(n_classes):
        iflat = input[:, c]
        tflat = (target == c).float()
        intersection = (iflat * tflat).sum()

        if weight is not None:
            w = weight[c]
            loss += w*(1 - ((2. * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth)))
        else:
            loss += 1 - ((2. * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth))

    return loss

def dice_loss(input, target, weight = None):
    prob_ = F.softmax(input, dim = 1)
    loss = dice_loss_cal(prob_, target, weight)
    return loss 

def dice_cross_entropy_loss(input, target, weight = None, rate = 0.5):
    loss = rate * F.cross_entropy(input, target, weight)
    loss += dice_loss(input, target, weight)
    return loss 


if __name__ == "__main__":
    device = torch.device("cuda")
    input = torch.rand((2,2,100,100,100))
    target = torch.randint(0,2,(2,100,100,100),dtype = torch.long)
    loss = dice_loss(input.to(device),target.to(device))
    print(loss)

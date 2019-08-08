import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

EPS = 1e-10

class FFNLoss(nn.Module):
    """ loss = sum(log(vi)mi - log(1 - vi)(1 - mi))
    """
    def __init__(self, use_gpu=True, eps=1e-15):
        super(FFNLoss, self).__init__()
        self.use_gpu = use_gpu
        self.eps = eps

    def forward(self, predictions, targets):
        pred = torch.squeeze(predictions, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1-self.eps)
        # print(torch.max(predictions), torch.max(targets))
        return torch.sum( -torch.log(pred)*targets - torch.log(1.0-pred)*(1.0-targets) )

class sigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,reduction='elementwise_mean'):
        super(sigmoidCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction 
    def forward(self,input,target):
        input = torch.unsqueeze(input, dim=1)
        scores = torch.log(torch.cat([1-input,input], dim=1))
        return F.nll_loss( scores, target, weight=self.weight, ignore_index=self.ignore_index )

class FocalLoss_(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target, alpha):
        if isinstance(alpha,(float,int)): alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): alpha = torch.Tensor(alpha)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target = (target > 0.9).long()

        # logpt = F.log_softmax(input)
        if input.size(1) == 1:
            # torch.stack([1.-input, input], dim=2)
            logpt = torch.stack([F.logsigmoid(1-input), F.logsigmoid(input)], dim=1)
            logpt = torch.squeeze(logpt, 2)
        else:
            logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if alpha is not None:
            if alpha.type()!=input.data.type():
                alpha = alpha.type_as(input.data)
            at = alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()
        self.size_average = size_average
        print ("FOCAL LOSS: ", gamma)

    def forward(self, input, target, alpha):
        target = target.float()
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        #target = target.view(-1,1)
        # input = self.sigmoid(input)
        input = torch.sigmoid(input)
        target = target.view(-1,1)
        target = target.float()
        # pt = input * target + (1 - input) * (1 - target)
        # pt = input ** target + (1 - input) ** (1 - target)
        pt = (input + EPS) * (0.05 + target * torch.where(target > 0.5, torch.Tensor([1]), torch.Tensor([-1]))) + (1 - input + EPS) * ( 0.95 + target * torch.where(target > 0.5, torch.Tensor([-1]), torch.Tensor([1])))
        # pt = ((input + EPS) ** target) * ((1 - input + EPS) ** (1 - target))
        logpt = pt.log()

        # at = (1 - alpha) * target + (alpha) * (1 - target)
        # at = alpha * (target - 0.05) + (1.) * (0.95 - target)
        at = torch.where(target>0.5, torch.Tensor([alpha]), torch.Tensor([1.]))
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
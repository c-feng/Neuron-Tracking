import torch
import torch.nn as nn
import numpy as np
from .diceLoss import DiceLossPlusCrossEntrophy

class DfnLoss(nn.Module):
    def __init__(self,weight = None,size_average = True):
        super(DfnLoss,self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.crossEntrophy = nn.CrossEntropyLoss(weight)
        self.mseLoss = nn.MSELoss()

    def forward(self,inputs,targets):
        directField,score = inputs
        directField_gt,skeleton_gt = targets
        #skeleton_gt = torch.LongTensor(np.array(targets[:,-1],dtype = np.int))
        #skeleton_gt = targets[:,-1].type(torch.long)
        #skeleton_gt = torch.cat([1 - skeleton_gt,skeleton_gt],dim = 1)
        
        mseLoss = self.mseLoss(directField,directField_gt) 
        crossEntrophy = self.crossEntrophy(score,skeleton_gt)
        #print("mseLoss {:.4f} CrossEntropyLoss {:.4f}".format(mseLoss,crossEntrophy))
        loss = mseLoss + crossEntrophy 
        #loss =  crossEntrophy 
        return loss

class DfnLossPlusDice(nn.Module):
    def __init__(self,weight = None,size_average = True):
        super(DfnLossPlusDice,self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.crossEntrophy = DiceLossPlusCrossEntrophy(weight)
        self.mseLoss = nn.MSELoss()

    def forward(self,inputs,targets):
        directField,score = inputs
        directField_gt,skeleton_gt = targets
        #skeleton_gt = torch.LongTensor(np.array(targets[:,-1],dtype = np.int))
        #skeleton_gt = targets[:,-1].type(torch.long)
        #skeleton_gt = torch.cat([1 - skeleton_gt,skeleton_gt],dim = 1)
        
        mseLoss = self.mseLoss(directField,directField_gt) 
        crossEntrophy = self.crossEntrophy(score,skeleton_gt)
        #print("mseLoss {:.4f} CrossEntropyLoss {:.4f}".format(mseLoss,crossEntrophy))
        loss = mseLoss + crossEntrophy 
        #loss =  crossEntrophy 
        return loss


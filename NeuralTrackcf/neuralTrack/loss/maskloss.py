import torch
from ..utils.hungarian import softIoU, StableBalancedMaskedBCE
from .diceLoss import DiceLossPlusCrossEntrophy
import torch.nn as nn


class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred, sw):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1)
        # costs = torch.mean(torch.masked_select(costs,sw.byte()))
        costs = torch.masked_select(costs,sw.byte())
        if costs.nelement() != 0:
            loss = torch.mean(costs)
        else:
            loss = costs
        #loss = torch.mean(costs)
        return loss

class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred,sw):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight)
        costs = costs[sw.byte()]
        if costs.nelement() != 0:
            loss = torch.mean(costs)
        else:
            loss = costs
        return loss
class BCEWithLogitsLoss(nn.Module):
    def __init__(self,balance_weight=None):
        super(BCEWithLogitsLoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight)
        if costs.nelement() != 0:
            loss = torch.mean(costs)
        else:
            loss = costs
        return loss

class softIoULoss(nn.Module):
    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred, sw):
        costs = softIoU(y_true,y_pred)
        costs = costs[sw.byte()]

        if costs.nelement() != 0:
            loss = torch.mean(costs)
        else:
            loss = costs
        return loss
        
class MaskedMixedLoss(nn.Module):
    def __init__(self,balance_weight = None):
        super(MaskedMixedLoss,self).__init__()
        self.balance_weight = balance_weight 
        self.diceloss = DiceLossPlusCrossEntrophy(self.balance_weight)
        #self.diceloss = softIoULoss()
        #self.stopLoss = BCEWithLogitsLoss()
        self.stopLoss = nn.BCEWithLogitsLoss()
    def forward(self,y_true,sw_mask,y_pred,stops_pred,sw):
        diceloss = self.diceloss(y_pred,y_true)
        print(sw_mask.float())
        print(stops_pred>0)
        stopLoss = self.stopLoss(stops_pred.float(),sw_mask.float())
        print(diceloss,stopLoss)
        loss = diceloss + stopLoss
        return loss
if __name__ == "__main__":
    a = torch.randint(0,2,(2,2,4,4,4))
    b = torch.randint(0,2,(2,2,4,4,4))
    c = torch.randint(0,2,(2,2))
    d = torch.randint(0,2,(2,2))
    
    e = MaskedMixedLoss()
    loss = e(a,c,b,d,c)
    #print(loss)

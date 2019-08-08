import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class sigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,reduction='elementwise_mean'):
        super(sigmoidCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction 
    def forward(self,input,target):
        input = torch.unsqueeze(input,dim = 1)
        scores = torch.log(torch.cat([1-input,input],dim =1))
        return F.nll_loss( scores, target, weight=self.weight, ignore_index=self.ignore_index ) 


if __name__ == "__main__":
    device = torch.device("cuda")
    
    #input = torch.rand(2,2).to(device)
    input = torch.tensor([[0.7,0.2],[0.6,0.2]]).to(device)
    #input2 = 
    temp = torch.unsqueeze(input,dim = 1)
    input2 = torch.cat([1-temp,temp],dim =1)
    target = torch.tensor([[0,1],[1,0]]).to(device)
    weight = torch.tensor([0.3,0.7]).to(device)
    criterion = sigmoidCrossEntropyLoss(weight)
    print(input,input2)
    print(criterion(input,target))

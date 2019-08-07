import torch
import torch.nn as nn

from neural_seg import NeuralSeg
from neural_ins import NeuralPointNet


class NeuralNet(nn.Module):
    def __init__(self, in_dims, pc_channels, use_xyz=True, mode="seg_ins", joint=True):
        super.__init__()

        self.mode = mode
        self.joint = joint

        self.neuralSeg = NeuralSeg(in_dims)
        self.pointIns = NeuralPointNet(pc_channels, use_xyz)
    
    def forward(self, x):
        if self.mode == "seg":
            output = {}
            full_seg, _ = self.neuralSeg(x)
            output{"full_seg"} = full_seg

            if self.joint:
                points = trans_seg2pc(full_seg)
                ins = self.pointIns(points)
                output{"ins"} = ins

        elif self.mode == "ins":
            output = self.pointIns(x)

        return output

def trans_seg2pc(full_seg):

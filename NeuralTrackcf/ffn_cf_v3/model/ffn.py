import torch
import torch.nn as nn
from functools import partial

from netutils import VoxCABHead, SegHead, FuseNet, VoxResnet

class FFNNet(nn.Module):
    def __init__(self, in_dims=1, ins_dims=3):
        super(FFNNet, self).__init__()
        self.featureHead = VoxCABHead(in_dims)
        self.segHead = SegHead(64 * 3)
        self.fuseNet = FuseNet(64 * 3)
        self.ffn = VoxResnet(in_dims=ins_dims, num_classes=2)
    
    def forward(self):
        """
            imgs: B * 1 * h * w * d
            gts: B * h * w * d
        """
        pass

import torch
import torch.nn as nn

from neuralTrack.models.voxResnet import VoxRes, RRB, CAB
from neuralTrack.models.emb import VoxCABHead, SegHead, EmbHead, MatchHead
from ffn_cf_v2.models.voxResnet import VoxResnet

# class SegFFNNet(nn.Module):
#     def __init__(self, in_dims, ins_dims=3):
#         super(SegFFNNet, self).__init__()
#         self.featureHead = VoxCABHead(in_dims)
#         self.segHead = SegHead(64 * 3)
#         self.fuseNet = FuseNet(64 * 3)
#         self.ffn = VoxResnet(in_dims=ins_dims, num_classes=2)
    
#     def forward(self, x):
#         batch_size = x.size(0)  # [N, 96, 96, 96]

#         rrb = torch.cat(self.featureHead(x), dim=1)

#         full_seg = self.segHead(rrb)

#         return full_seg

class SegFFNNet(nn.Module):
    def __init__(self, in_dims, ins_dims=3):
        super(SegFFNNet, self).__init__()
        self.featureHead = VoxCABHead(in_dims)
        self.segHead = SegHead(64 * 3)
        self.ffn = VoxResnet(in_dims=ins_dims, num_classes=2)
    
    def forward(self, x):
        batch_size = x.size(0)  # [N, 96, 96, 96]

        rrb = torch.cat(self.featureHead(x), dim=1)

        full_seg = self.segHead(rrb)

        return full_seg

class FuseNet(nn.Module):
    def __init__(self, feature_dims=64*3):
        super(FuseNet, self).__init__()
        self.conv1 = nn.Conv3d(feature_dims, 1, 1, 1, 0)

    def forward(self, x):
        f = self.conv1(x)
        # f = torch.cat((seg, f), dim=1)

        return f

class SegFFNNetEmb(nn.Module):
    def __init__(self, in_dims, emb_dims=32):
        super(SegFFNNetEmb, self).__init__()
        self.featureHead = VoxCABHead(in_dims)
        self.segHead = SegHead(64 * 3)
        self.embhead = EmbHead(in_dims=64*3, emd_dims=emb_dims)
        self.matchHead = MatchHead(in_dims=emb_dims)
        self.fuseNet = FuseNet(64 * 3)
        self.ffn = VoxResnet(in_dims=3+1, num_classes=2)
    
    def forward(self, x):
        batch_size = x.size(0)
        rrb = torch.cat(self.featureHead(x), dim=1)

        full_seg = self.segHead(rrb)

        return full_seg

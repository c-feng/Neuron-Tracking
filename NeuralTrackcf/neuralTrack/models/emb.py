from functools import partial

import torch

import numpy as np
import torch.nn as nn

from torch.nn import init

from .voxResnet import VoxRes, RRB, CAB 
from ..utils.match_utils import batch_emb_match

class EmbSeg(nn.Module):
    def __init__(self, in_dims, emb_dims = 32):
        super(EmbSeg, self).__init__()
        self.featureHead = VoxCABHead(in_dims)
        self.segHead = SegHead(64 * 3)
        self.embHead = EmbHead(64 * 3, emb_dims)
        self.matchHead = MatchHead(emb_dims * 2 + 1)
        self.insHead = SegHead(3 + emb_dims)
        
    def forward(self, img, prev_img, prev_pred, cur_anchor):
        rrb = torch.cat(self.featureHead(img), dim = 1)
        seg = self.segHead(rrb)
        emb = self.embHead(rrb)

        cur_match = self.matchHead(emb, emb, cur_anchor)

        prev_rrb = torch.cat(self.featureHead(prev_img), dim = 1)
        prev_emb = self.embHead(prev_rrb) 

        prev_match = self.matchHead(emb, prev_emb, prev_pred)
        
        
        fs =  torch.cat([cur_match, prev_match, cur_pred[:,None].float(), emb] , dim = 1)
        ins = self.insHead(fs)
        return ins, seg, cur_match, prev_match, emb

class ScoreEmbSeg(nn.Module):
    def __init__(self, in_dims, emb_dims = 32):
        super(ScoreEmbSeg, self).__init__()
        self.featureHead = VoxCABHead(in_dims)
        self.segHead = SegHead(64 * 3)
        self.embHead = EmbHead(64 * 3, emb_dims)
        self.matchHead = MatchHead(emb_dims * 2 + 1)
        self.insHead = SegHead(3 + emb_dims)
        self.juncHead = SegHead(emb_dims)
        
    def forward(self, img, prev_img, prev_pred, cur_pred):
        rrb = torch.cat(self.featureHead(img), dim = 1)
        seg = self.segHead(rrb)

        emb = self.embHead(rrb)
        cur_match = self.matchHead()
        prev_rrb = torch.cat(self.featureHead(prev_img), dim = 1)
        prev_emb = self.embHead(prev_rrb) 
        prev_match = self.matchHead(torch.cat(\
                [emb, prev_emb, prev_pred[:,None].float()], dim = 1))
        

        fs =  torch.cat([cur_match, prev_match, cur_pred[:,None].float(), emb] , dim = 1)
        ins = self.insHead(fs)
        junc = self.juncHead(emb)
        return ins, seg, junc, cur_match, prev_match, emb

class MatchHead(nn.Module):
    def __init__(self, in_dims ):
        super(MatchHead, self).__init__()
        self.feature = VoxRes(in_dims)
        self.dims_align = nn.Conv3d(64, 1, 1, 1, 0)
    def forward(self, cur_emb, prev_emb, prev_pred):
        input_ = torch.cat(\
                [cur_emb, prev_emb, prev_pred[:,None].float()], dim = 1)

        feature = self.feature(input_)
        match_map = torch.sigmoid(self.dims_align(feature))
        return match_map

class EmbHead(nn.Module):
    def __init__(self, in_dims , emd_dims = 32):
        super(EmbHead, self).__init__()
        self.feature = VoxRes(in_dims)
        self.dims_align = nn.Conv3d(64, emd_dims, 1, 1, 0)
    def forward(self, input_):
        feature = self.feature(input_)
        emb = self.dims_align(feature)
        return emb

class SegHead(nn.Module):
    def __init__(self, in_dims):
        super(SegHead, self).__init__()
            
        self.conv_1a = nn.Conv3d(in_dims,32,3,1,1)
        self.bn_1a = nn.BatchNorm3d(32)
        self.relu = nn.LeakyReLU(1e-5, inplace=True)
        self.conv_1b = nn.Conv3d(32,32,3,1,1)
        self.bn_1b = nn.BatchNorm3d(32)
        self.conv_1c = nn.Conv3d(32,64,3,2,1)

        self.res_2 = VoxRes(64)
        self.res_3 = VoxRes(64)

        self.upsample = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier = nn.Conv3d(64,2,1,1)
    def forward(self, input_):
        x = input_
        
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        x = self.conv_1b(x)
        
        x = self.bn_1b(x)
        x = self.relu(x)
        x = self.conv_1c(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.upsample(x)
        x = self.classifier(x)
        return x 


class VoxCABHead(nn.Module):
    def __init__(self,in_dims):
        super(VoxCABHead,self).__init__()
        self.in_dims = in_dims
        self.conv_1a = nn.Conv3d(in_dims,32,3,1,1)
        self.bn_1a = nn.BatchNorm3d(32)
        self.relu = nn.LeakyReLU(1e-5, inplace=True)

        self.conv_1b = nn.Conv3d(32,32,3,1,1)
        self.bn_1b = nn.BatchNorm3d(32)
        
        self.conv_1c = nn.Conv3d(32,64,3,2,1)

        self.res_2 = VoxRes(64)
        self.res_3 = VoxRes(64)
        self.bn_3 = nn.BatchNorm3d(64)

        self.conv_4 = nn.Conv3d(64,64,3,2,1)

        self.res_5 = VoxRes(64)

        self.res_6 = VoxRes(64)

        self.bn_6 = nn.BatchNorm3d(64)

        self.conv_7 = nn.Conv3d(64,64,3,2,1)

        self.res_8 = VoxRes(64)

        self.res_9 = VoxRes(64)
        

        self.rrb_0_0 = RRB(64)
        self.cab_0 = CAB(64,64)
        self.rrb_0_1 = RRB(64)
        self.upsample_0 = nn.ConvTranspose3d(64,64,2,2,0)

        self.rrb_1_0 = RRB(64)
        self.cab_1 = CAB(64,64)
        self.rrb_1_1 = RRB(64)
        self.upsample_1_0 = nn.ConvTranspose3d(64,64,2,2,0)
        #self.upsample_1_1 = nn.Upsample(scale_factor = 2)
        self.upsample_1_1 = partial(nn.functional.interpolate, scale_factor = 2)

        self.rrb_2_0 = RRB(64)
        self.cab_2 = CAB(64,64)
        self.rrb_2_1 = RRB(64)
        self.upsample_2_0 = nn.ConvTranspose3d(64,64,2,2,0)
        #self.upsample_2_1 = nn.Upsample(scale_factor = 4)
        self.upsample_2_1 = partial(nn.functional.interpolate, scale_factor = 4)

        self.reset_params()

    def forward(self,img):
        x = img
        #print(x.view(-1)[:50])    
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        x = self.conv_1b(x)
        #print(x.view(-1)[:50])    
        
        x = self.bn_1b(x)
        x = self.relu(x)
        x = self.conv_1c(x)
        x = self.res_2(x)
        x = self.res_3(x)

        rrb_0_0 = self.rrb_0_0(x)

        x = self.bn_3(x)
        x = self.relu(x)
        x = self.conv_4(x)
        x = self.res_5(x)
        x = self.res_6(x)

        rrb_1_0 = self.rrb_1_0(x)

        x = self.bn_6(x)
        x = self.relu(x)
        x = self.conv_7(x)
        x = self.res_8(x)
        x = self.res_9(x)
        
        gf = torch.mean(x.view(x.size(0),x.size(1),-1),dim = -1)
        gf = gf[:,:,None,None,None].expand(-1,-1,x.size(2),x.size(3),x.size(4))

        rrb_2_0 = self.rrb_2_0(x)
        rrb_2_0 = self.cab_2(rrb_2_0,gf)
        rrb_2_1 = self.rrb_2_1(rrb_2_0)
        #print(rrb_2_1.size())
        rrb_2 = self.upsample_2_0(rrb_2_1)

        #print(rrb_2.size(), rrb_1_0.size())
        rrb_1_0 = self.cab_1(rrb_1_0,rrb_2) 
        rrb_1_1 = self.rrb_1_1(rrb_1_0)
        rrb_1 = self.upsample_1_0(rrb_1_1)

        rrb_0_0 = self.cab_0(rrb_0_0,rrb_1)
        rrb_0_1 = self.rrb_0_1(rrb_0_0)
        rrb_0 = self.upsample_0(rrb_0_1)

        rrb_2 = self.upsample_2_1(rrb_2)
        rrb_1 = self.upsample_1_1(rrb_1) 
        return rrb_0, rrb_1, rrb_2 
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = VoxCAB(1,2)
    model = model.to(device)
    output_0,output_1,output_2 = model(img)
    print(output_0.size())

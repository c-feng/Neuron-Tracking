import torch

import numpy as np
import torch.nn as nn

from torch.nn import init

def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.LeakyReLU(1e-5)]
            else:
                layers += [conv3d, nn.LeakyReLU(1e-5)]
        in_channels = v
    return nn.Sequential(*layers)

class VoxRes(nn.Module):
    def __init__(self,cfg):
        super(VoxRes,self).__init__()
        self.features = make_layers(cfg)
        conv3d = nn.Conv3d(cfg[0],cfg[-1],kernel_size = 1)
        batch_norm = nn.BatchNorm3d(cfg[-1])
        self.dims_align = nn.Sequential(conv3d,batch_norm,nn.LeakyReLU(1e-5))
        self.relu = nn.LeakyReLU(1e-5)
        self.reset_params()
    def forward(self,x):
        #print(x.size())
        x =  self.features(x) + self.dims_align(x)
        #print(x.size())
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()
class DFN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(DFN,self).__init__()
        self.vox_0 = make_layers([in_channels,32,32])
        #self.maxPool_0 = nn.MaxPool3d(2,2)
        self.maxPool_0 = nn.Sequential(nn.Conv3d(32,64,3,2,1),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        

        self.voxRes_1 = VoxRes([64,64,64])
        self.voxRes_2 = VoxRes([64,64,64])
        
        self.maxPool_2 = nn.Sequential(nn.Conv3d(64,64,3,2,1),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))

        self.voxRes_3 = VoxRes([64,64,64])
        self.voxRes_4 = VoxRes([64,64,64])
        self.maxPool_4 = nn.Sequential(nn.Conv3d(64,64,3,2,1),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        
        self.voxRes_5 = VoxRes([64,64,64])
        self.voxRes_6 = VoxRes([64,64,64])

        self.side_upsample_6 = nn.Sequential(nn.ConvTranspose3d(64,64,2,2,0),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        self.side_output_6 = nn.Sequential(nn.Conv3d(64,2,1,1,0),nn.ConvTranspose3d(2,2,8,8,0))

        self.side_filter_4 = nn.Sequential(nn.Conv3d(64,64,1,1,0),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        self.side_upsample_4 = nn.Sequential(nn.ConvTranspose3d(64,64,2,2,0),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        self.side_output_4 = nn.Sequential(nn.Conv3d(64,2,1,1,0),nn.ConvTranspose3d(2,2,4,4,0)) 

        self.side_filter_2 = nn.Sequential(nn.Conv3d(64,64,1,1,0),\
                nn.BatchNorm3d(64),nn.LeakyReLU(1e-5))
        self.side_output_2 = nn.Sequential(nn.Conv3d(64,2,1,1,0),nn.ConvTranspose3d(2,2,2,2,0)) 
        
        self.side_output = nn.Conv3d(6,2,1,1,0)

        self.reset_params()
    def forward(self,x):
        x_0 = self.vox_0(x)
        x_1 = self.maxPool_0(x_0)
        x_1 = self.voxRes_1(x_1)
        x_2 = self.voxRes_2(x_1)
        x_3 = self.maxPool_2(x_2)
        x_3 = self.voxRes_3(x_3)
        x_4 = self.voxRes_4(x_3)
        x_5 = self.maxPool_4(x_4)
        x_5 = self.voxRes_5(x_5)
        x_6 = self.voxRes_6(x_5)
        
        f_4 = self.side_upsample_6(x_6) + self.side_filter_4(x_4)
        f_2 = self.side_upsample_4(f_4) + self.side_filter_2(x_2)
        
        s_6 = self.side_output_6(x_6)
        s_4 = self.side_output_4(f_4)
        s_2 = self.side_output_2(f_2)
        
        s = self.side_output(torch.cat([s_6,s_4,s_2],dim = 1))
        #s = s_6 + s_4 + s_2
        return s,s_6,s_4,s_2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = DFN(1,40)
    model = model.to(device)
    outputs = model(img)
    print(outputs[0][0].size(),outputs[0][-1].size())

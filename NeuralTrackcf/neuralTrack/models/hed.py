import torch

import numpy as np
import torch.nn as nn

from torch.nn import init

class VoxRes(nn.Module):
    def __init__(self,in_dims):
        super(VoxRes,self).__init__()
        self.in_dims = in_dims
        self.bn_1 = nn.BatchNorm3d(in_dims)
        self.conv_1 = nn.Conv3d(in_dims,64,3,1,1)
        self.bn_2 = nn.BatchNorm3d(64)
        self.conv_2 =nn.Conv3d(64,64,3,1,1)
        self.relu = nn.LeakyReLU(1e-5)
    def forward(self,img):
        x = img
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x += img
        return x
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

class VoxResnet(nn.Module):
    def __init__(self,in_dims,num_classes):
        super(VoxResnet,self).__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes 
        self.conv_1a = nn.Conv3d(in_dims,32,3,1,1)
        self.bn_1a = nn.BatchNorm3d(32)
        self.relu = nn.LeakyReLU(1e-5)

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
        

        self.upsample_0 = nn.Conv3d(32,num_classes,3,1,1)
        #self.upsample_0 = nn.Upsample()
        self.classifier_0 = nn.Conv3d(num_classes,num_classes,1,1,0)

        self.upsample_1 = nn.ConvTranspose3d(64,num_classes,2,2,0)
        #self.upsample_1 = nn.Upsample(scale_factor = 2)
        self.classifier_1 = nn.Conv3d(num_classes,num_classes,1,1,0)
        #self.classifier_1 = nn.Conv3d(64,num_classes,1,1,0)

        self.upsample_2 = nn.ConvTranspose3d(64,num_classes,4,4,0)
        #self.upsample_2 = nn.Upsample((4,4,4))
        #self.upsample_2 = nn.Upsample(scale_factor = 4)
        self.classifier_2 = nn.Conv3d(num_classes,num_classes,1,1,0)
        #self.classifier_2 = nn.Conv3d(64,num_classes,1,1,0)

        self.upsample_3 = nn.ConvTranspose3d(64,num_classes,8,8,0)
        #self.upsample_3 = nn.Upsample((8,8,8))
        #self.upsample_3 = nn.Upsample(scale_factor = 8)
        self.classifier_3 = nn.Conv3d(num_classes,num_classes,1,1,0)
        #self.classifier_3 = nn.Conv3d(64,num_classes,1,1,0)
        self.fuse = nn.Conv3d(num_classes*4,num_classes,1,1,0)
    def forward(self,img):
        x = img
        
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        
        x = self.conv_1b(x)
        output_0 = self.upsample_0(x)
        output_0 = self.classifier_0(output_0)

        x = self.bn_1b(x)
        x = self.relu(x)

        x = self.conv_1c(x)

        x = self.res_2(x)

        x = self.res_3(x)
        output_1 = self.upsample_1(x)
        output_1 = self.classifier_1(output_1)

        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_4(x)

        x = self.res_5(x)
        
        x = self.res_6(x)
        output_2 = self.upsample_2(x)
        output_2 = self.classifier_2(output_2)

        x = self.bn_6(x)
        x = self.relu(x)
        
        x = self.conv_7(x)

        x = self.res_8(x)

        x = self.res_9(x)
        output_3 = self.upsample_3(x)
        output_3 = self.classifier_3(output_3)

        #output = output_0 + output_1 + output_2 + output_3
        output = torch.cat([output_0,output_1,output_2,output_3],dim = 1)
        output = self.fuse(output)
        return output,output_0,output_1,output_2,output_3 

if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = VoxResnet(1,2)
    model = model.to(device)
    output,output_0,output_1,output_2,output_3 = model(img)
    print(output.size(),output_0.size())

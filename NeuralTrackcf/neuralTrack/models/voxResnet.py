import torch

import numpy as np
import torch.nn as nn

from torch.nn import init

class EndsNet(nn.Module):
    def __init__(self, in_dims, num_classes):
        super(EndsNet,self).__init__()
        self.feature = VoxRes(in_dims)
        self.classifier = nn.Conv3d(64,num_classes,1,1,0)
    def forward(self, input_):
        feature = self.feature(input_)
        output_ = self.classifier(feature)
        return output_

class VoxRes(nn.Module):
    def __init__(self,in_dims, out_dims = 64):
        super(VoxRes,self).__init__()
        self.in_dims = in_dims
        self.bn_1 = nn.BatchNorm3d(in_dims)
        self.conv_1 = nn.Conv3d(in_dims,64,3,1,1)
        self.bn_2 = nn.BatchNorm3d(64)
        self.conv_2 =nn.Conv3d(64,out_dims,3,1,1)
        self.relu = nn.LeakyReLU(1e-5)
        self.dim_align = nn.Conv3d(in_dims, out_dims, 1, 1, 0)
        self.reset_params()
    def forward(self,img):
        x = img
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x += self.dim_align(img)
        return x
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

class RRB(nn.Module):
    def __init__(self,in_dims):
        super(RRB,self).__init__()
        self.in_dims = in_dims
        self.bn = nn.BatchNorm3d(in_dims)

        self.conv_1 = nn.Conv3d(in_dims,in_dims,1,1,0)
        self.conv_2 = nn.Conv3d(in_dims,in_dims,3,1,1)
        self.conv_3 =nn.Conv3d(in_dims,in_dims,3,1,1)
        self.relu = nn.LeakyReLU(1e-5)

        self.reset_params()
    def forward(self,img):
        img = self.conv_1(img)
        x = self.conv_2(img)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x += img

        x = self.relu(x)
        return x
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

class CAB(nn.Module):
    def __init__(self,in_dims_0,in_dims_1):
        super(CAB,self).__init__()

        self.fc_0 = nn.Linear(in_dims_0 + in_dims_1, in_dims_0 + in_dims_1)
        self.fc_1 = nn.Linear(in_dims_0 + in_dims_1, in_dims_0)
        self.relu = nn.LeakyReLU(1e-5)
        self.sigmoid = nn.Sigmoid()

        self.reset_params()
    def forward(self,f0,f1):
        f = torch.cat([f0, f1], dim = 1)
        f = torch.mean(f.view(f.size(0),f.size(1),-1), dim = -1)
        f = self.fc_0(f)
        f = self.relu(f)
        f = self.fc_1(f)
        w = self.sigmoid(f)
        f0_w = f0 * w[:,:,None,None,None]
        f = f0_w + f1
        return f

    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
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
        self.reset_params()
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
        
        #print(output_0.size(),output_1.size(),output_2.size(),output_3.size())
        output = output_0 + output_1 + output_2 + output_3

        return output,output_0,output_1,output_2,output_3 
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

class VoxRRB(nn.Module):
    def __init__(self,in_dims,num_classes):
        super(VoxRRB,self).__init__()
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
        

        self.rrb_0_0 = RRB(64)
        self.rrb_0_1 = RRB(64)
        self.upsample_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_0 = nn.Conv3d(64,num_classes,1,1,0)

        self.rrb_1_0 = RRB(64)
        self.rrb_1_1 = RRB(64)
        self.upsample_1_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_1 = nn.Conv3d(64,num_classes,1,1,0)
        self.upsample_1_1 = nn.ConvTranspose3d(num_classes,num_classes,2,2,0)

        self.rrb_2_0 = RRB(64)
        self.upsample_2_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_2 = nn.Conv3d(64,num_classes,1,1,0)
        self.upsample_2_1 = nn.ConvTranspose3d(num_classes,num_classes,4,4,0)

        #self.upsample = nn.ConvTranspose3d(64,64,2,2,0)
        #self.classifier = nn.Conv3d(64,num_classes,1,1,0)

        self.reset_params()

    def forward(self,img):
        x = img
        
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        x = self.conv_1b(x)
        
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
        
        rrb_2_0 = self.rrb_2_0(x)
        rrb_2 = self.upsample_2_0(rrb_2_0)
        output_2 = self.classifier_2(rrb_2)

        output_2 = self.upsample_2_1(output_2)

        rrb_1_0 = rrb_2 + rrb_1_0
        rrb_1_1 = self.rrb_1_1(rrb_1_0)
        rrb_1 = self.upsample_1_0(rrb_1_1)
        output_1 = self.classifier_1(rrb_1)

        output_1 = self.upsample_1_1(output_1)

        rrb_0_0 = rrb_1 + rrb_0_0
        rrb_0_1 = self.rrb_0_1(rrb_0_0)
        rrb_0 = self.upsample_0(rrb_0_1)
        output_0 = self.classifier_0(rrb_0)

        return output_0,output_1,output_2 
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()

class VoxCAB(nn.Module):
    def __init__(self,in_dims,num_classes):
        super(VoxCAB,self).__init__()
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
        

        self.rrb_0_0 = RRB(64)
        self.cab_0 = CAB(64,64)
        self.rrb_0_1 = RRB(64)
        self.upsample_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_0 = nn.Conv3d(64,num_classes,1,1,0)

        self.rrb_1_0 = RRB(64)
        self.cab_1 = CAB(64,64)
        self.rrb_1_1 = RRB(64)
        self.upsample_1_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_1 = nn.Conv3d(64,num_classes,1,1,0)
        self.upsample_1_1 = nn.ConvTranspose3d(num_classes,num_classes,2,2,0)

        self.rrb_2_0 = RRB(64)
        self.cab_2 = CAB(64,64)
        self.rrb_2_1 = RRB(64)
        self.upsample_2_0 = nn.ConvTranspose3d(64,64,2,2,0)
        self.classifier_2 = nn.Conv3d(64,num_classes,1,1,0)
        self.upsample_2_1 = nn.ConvTranspose3d(num_classes,num_classes,4,4,0)

        #self.upsample = nn.ConvTranspose3d(64,64,2,2,0)
        #self.classifier = nn.Conv3d(64,num_classes,1,1,0)

        self.reset_params()

    def forward(self,img):
        x = img
        
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        x = self.conv_1b(x)
        
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
        rrb_2 = self.upsample_2_0(rrb_2_1)

        output_2 = self.classifier_2(rrb_2)

        output_2 = self.upsample_2_1(output_2)

        #rrb_1_0 = rrb_2 + rrb_1_0
        rrb_1_0 = self.cab_1(rrb_1_0,rrb_2) 
        rrb_1_1 = self.rrb_1_1(rrb_1_0)
        rrb_1 = self.upsample_1_0(rrb_1_1)
        output_1 = self.classifier_1(rrb_1)

        output_1 = self.upsample_1_1(output_1)

        rrb_0_0 = self.cab_0(rrb_0_0,rrb_1)
        rrb_0_1 = self.rrb_0_1(rrb_0_0)
        rrb_0 = self.upsample_0(rrb_0_1)
        output_0 = self.classifier_0(rrb_0)

        return output_0,output_1,output_2 
    
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

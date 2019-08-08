import torch

import numpy as np
import torch.nn as nn

from torch.nn import init

from ..loss.discriminative import DiscriminativeLoss


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


class VoxRes(nn.Module):
    def __init__(self,in_dims):
        super(VoxRes,self).__init__()
        self.in_dims = in_dims
        self.bn_1 = nn.BatchNorm3d(in_dims)
        self.conv_1 = nn.Conv3d(in_dims,64,3,1,1)
        self.bn_2 = nn.BatchNorm3d(64)
        self.conv_2 =nn.Conv3d(64,64,3,1,1)
        self.relu = nn.LeakyReLU(1e-5)

        self.reset_params()
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

class DLF(nn.Module):
    def __init__(self,in_dims,embedding_dims = 16):
        super(DLF,self).__init__()
        self.in_dims = in_dims
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
        

        self.upsample_0 = nn.Conv3d(32,2,3,1,1)
        self.classifier_0 = nn.Conv3d(2,2,1,1,0)

        self.ins_upsample_0 = nn.Conv3d(32,embedding_dims,3,1,1)
        self.ins_classifier_0 = nn.Conv3d(embedding_dims,embedding_dims,1,1,0)

        self.upsample_1 = nn.ConvTranspose3d(64,2,2,2,0)
        self.classifier_1 = nn.Conv3d(2,2,1,1,0)

        self.ins_upsample_1 = nn.ConvTranspose3d(64,embedding_dims,2,2,0)
        self.ins_classifier_1 = nn.Conv3d(embedding_dims,embedding_dims,1,1,0)

        self.upsample_2 = nn.ConvTranspose3d(64,2,4,4,0)
        self.classifier_2 = nn.Conv3d(2,2,1,1,0)

        self.ins_upsample_2 = nn.ConvTranspose3d(64,embedding_dims,4,4,0)
        self.ins_classifier_2 = nn.Conv3d(embedding_dims,embedding_dims,1,1,0)

        self.upsample_3 = nn.ConvTranspose3d(64,2,8,8,0)
        self.classifier_3 = nn.Conv3d(2,2,1,1,0)

        self.ins_upsample_3 = nn.ConvTranspose3d(64,embedding_dims,8,8,0)
        self.ins_classifier_3 = nn.Conv3d(embedding_dims,embedding_dims,1,1,0)

        self.reset_params()
    def forward(self,img):
        x = img
        
        x = self.conv_1a(x)
        x = self.bn_1a(x)
        x = self.relu(x)
        
        x = self.conv_1b(x)
        output_0 = self.upsample_0(x)
        output_0 = self.classifier_0(output_0)
        
        ins_0 = self.ins_upsample_0(x)
        ins_0 = self.ins_classifier_0(ins_0)

        x = self.bn_1b(x)
        x = self.relu(x)

        x = self.conv_1c(x)

        x = self.res_2(x)

        x = self.res_3(x)
        output_1 = self.upsample_1(x)
        output_1 = self.classifier_1(output_1)

        ins_1 = self.ins_upsample_1(x)
        ins_1 = self.ins_classifier_1(ins_1)

        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_4(x)

        x = self.res_5(x)
        
        x = self.res_6(x)
        output_2 = self.upsample_2(x)
        output_2 = self.classifier_2(output_2)

        ins_2 = self.ins_upsample_2(x)
        ins_2 = self.ins_classifier_2(ins_2)

        x = self.bn_6(x)
        x = self.relu(x)
        
        x = self.conv_7(x)

        x = self.res_8(x)

        x = self.res_9(x)
        output_3 = self.upsample_3(x)
        output_3 = self.classifier_3(output_3)
        
        ins_3 = self.ins_upsample_3(x)
        ins_3 = self.ins_classifier_3(ins_3)
        #print(output_0.size(),output_1.size(),output_2.size(),output_3.size())
        output = output_0 + output_1 + output_2 + output_3
        ins = ins_0 + ins_1 + ins_2 + ins_3
        return output,ins 
    
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()
class FullModel(nn.Module):
    def __init__(self,model,delta_var,delta_dist):
        super(FullModel,self).__init__()
        self.model = model
        self.loss = DiscriminativeLoss(delta_var,delta_dist,2)

    def forward(self,inputs_,ins_gt,n_objects):
    #def forward(self,inputs_,ins_gt,n_objects,max_n_objects):
        outputs_ = self.model(inputs_)
        sem_seg,ins_seg = outputs_
        loss = self.loss(ins_seg,ins_gt,n_objects)
        #print(loss.size())
        return torch.unsqueeze(loss,0),sem_seg,ins_seg

if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = DLF(1)
    model = model.to(device)
    s,ins = model(img)
    print(s.size(),ins.size())

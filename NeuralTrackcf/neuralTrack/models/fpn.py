import torch

import numpy as np
import torch.nn as nn

from torch.nn import init


class FPN(nn.Module):
    def __init__(self,in_channels = 1,num_classes = 2):
        super(FPN,self).__init__()
        #self.pool = nn.MaxPool3d(2,2)
        self.stage_0 = self.make_layers([in_channels,64],[64,64])
        self.pool_0 = nn.MaxPool3d(2,2)

        self.stage_1 = self.make_layers([64,128],[128,128])
        self.pool_1 = nn.MaxPool3d(2,2)

        self.stage_2 = self.make_layers([128,256,256],[256,256,256])
        self.pool_2 = nn.MaxPool3d(2,2)

        self.stage_3 = self.make_layers([256,512,512],[512,512,512])
        self.pool_3 = nn.MaxPool3d(2,2)

        self.stage_4 = self.make_layers([512,512,512],[512,512,512])
        #self.pool_4 = nn.MaxPool3d(2,2)

        #self.stage_5 = self.make_layers([512,512,512],[512,512,512])
        
        self.side_classifier_0 = nn.Conv3d(64,num_classes,1)
        self.side_classifier_1 = nn.Conv3d(128,num_classes,1)
        self.unsample_1 = nn.ConvTranspose3d(num_classes,num_classes,2,2)
        self.side_classifier_2 = nn.Conv3d(256,num_classes,1)
        self.unsample_2 = nn.ConvTranspose3d(num_classes,num_classes,4,4)
        self.side_classifier_3 = nn.Conv3d(512,num_classes,1)
        self.unsample_3 = nn.ConvTranspose3d(num_classes,num_classes,8,8)
        self.side_classifier_4 = nn.Conv3d(512,num_classes,1)
        self.unsample_4 = nn.ConvTranspose3d(num_classes,num_classes,16,16)
       
        self.fuse_classifier = nn.Conv3d(5*num_classes,num_classes,1)
        self.logSoftmax = nn.LogSoftmax(dim = 1)
    def forward(self,img):
        features_0 = self.stage_0(img)
        side_outputs_0 = self.side_classifier_0(features_0)
        side_outputs_0 = self.logSoftmax(side_outputs_0)
        features_0 = self.pool_0(features_0)
        

        features_1 = self.stage_1(features_0)
        side_outputs_1 = self.side_classifier_1(features_1)
        side_outputs_1 = self.unsample_1(side_outputs_1)
        side_outputs_1 = self.logSoftmax(side_outputs_1)
        features_1 = self.pool_1(features_1)

        features_2 = self.stage_2(features_1)
        side_outputs_2 = self.side_classifier_2(features_2)
        side_outputs_2 = self.unsample_2(side_outputs_2)
        side_outputs_2 = self.logSoftmax(side_outputs_2)
        features_2 = self.pool_2(features_2)

        features_3 = self.stage_3(features_2)
        side_outputs_3 = self.side_classifier_3(features_3)
        side_outputs_3 = self.unsample_3(side_outputs_3)
        side_outputs_3 = self.logSoftmax(side_outputs_3)
        features_3 = self.pool_3(features_3)

        features_4 = self.stage_4(features_3)
        side_outputs_4 = self.side_classifier_4(features_4)
        side_outputs_4 = self.unsample_4(side_outputs_4)
        side_outputs_4 = self.logSoftmax(side_outputs_4)


        fused_outputs = self.fuse_classifier(torch.cat(\
                [side_outputs_0,side_outputs_1,side_outputs_2,side_outputs_3,side_outputs_4],dim = 1))
        fused_outputs = self.logSoftmax(fused_outputs)
        
        
        #fused_outputs = torch.log(fused_outputs)
        
        #side_outputs_0 = torch.log(side_outputs_0)
        #side_outputs_1 = torch.log(side_outputs_1)
        #side_outputs_2 = torch.log(side_outputs_2)
        #side_outputs_3 = torch.log(side_outputs_3)
        #side_outputs_4 = torch.log(side_outputs_4)
        return fused_outputs,side_outputs_0,side_outputs_1,side_outputs_2,side_outputs_3,side_outputs_4 
         
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()
    def make_layers(self,in_dims,out_dims):
        layers = []
        for in_dim,out_dim in zip(in_dims,out_dims):
            layers.append(nn.Conv3d(in_dim,out_dim,3,1,1))
            layers.append(nn.BatchNorm3d(out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = VGG(1,2)
    model = model.to(device)
    output,output_0,output_1,output_2,output_3,output_4 = model(img)
    print(output.size(),output_0.size())

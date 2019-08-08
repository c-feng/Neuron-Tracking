import torch
import torch.nn as nn
from torch.nn import init 
import numpy as np


class BasicFCN(nn.Module):
    def __init__(self,num_classes = 0,channels = 4,
            kernels_dim= [30,30,40,40,40,40,50,50],
            kernels_size = [3,3,3,3,3,3,3,3],
            kernels_stride = [1,1,1,1,1,1,1,1],
            kernels_pad = [1,1,1,1,1,1,1,1],
            layers_connect = [4,6,8]):
        super(BasicFCN,self).__init__()
        
        self.channels = channels 
        self.kernels_dim = kernels_dim
        self.kernels_size = kernels_size
        self.kernels_stride = kernels_stride
        self.kernels_pad = kernels_pad 

        blocks_ind = []
        blocks = nn.ModuleList()
        for i,j in enumerate(layers_connect):
            if i == 0:
                blocks_ind.append([0,j-2,"straight"])
                blocks_ind.append([j-2,j,"residual"])
            else:
                if layers_connect[i-1]< j-2:
                    blocks_ind.append([layers_connect[i-1],j-2,"straight"])
                blocks_ind.append([j-2,j,"residual"])
        for x,y,f in blocks_ind:
            if x==0:
                channels =channels
            else:
                channels = kernels_dim[x-1]
            kernel_size = kernels_size[x:y]
            kernel_stride = kernels_stride[x:y]
            kernel_dim = kernels_dim[x:y]
            kernel_pad = kernels_pad[x:y]
            
            if f == "straight":
                block = BasicBlock(channels,kernel_dim,kernel_size,kernel_stride,kernel_pad)
                blocks.append(block)
            elif f == "residual":
                block = ResBlock(channels,kernel_dim,kernel_size,kernel_stride,kernel_pad)
                blocks.append(block)
        if len(blocks) == 0:
            blocks.append(BasicBlock(channels,kernels_dim,
                    kernels_size,kernels_stride,kernels_pad))
        
        self.blocks = blocks
        self.reset_params()
        self.receptive_fields,self.receptive_strides = self.receptive_field_cal()
        self.receptive_pads = np.array(kernels_pad)*np.array(self.receptive_strides[1:])
    def forward(self,img):
        x = img 
        for block in self.blocks:
            x = block(x)
        return x 

    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def receptive_field_cal(self):
        kernels_size = self.kernels_size
        kernels_stride = self.kernels_stride 
        receptive_strides = [1]
        receptive_fileds = [1]
        for i,kernel_size in enumerate(kernels_size):
            kernel_stride = kernels_stride[i]
            receptive_stride = receptive_strides[i]*kernel_stride
            receptive_strides.append(kernel_stride)
            receptive_filed = receptive_fileds[i] + \
                (kernel_size - 1)*receptive_strides[i]
            receptive_fileds.append(receptive_filed)
        return receptive_fileds,receptive_strides     
        
class BasicBlock(nn.Module):
    def __init__(self,channels = 4,
            kernels_dim = [30,30],
            kernels_size = [3,3],
            kernels_stride = [1,1],
            kernels_pad = [1,1]
            ):
        super(BasicBlock,self).__init__()
        self.channels = channels 
        self.kernels_dim = kernels_dim
        self.kernels_size = kernels_size
        self.kernels_stride = kernels_stride
        self.kernels_pad = kernels_pad 
        self.receptive_fields,self.receptive_strides = self.receptive_field_cal()
        
        self.pReLU = nn.PReLU()
        self.CNNs,self.BNs  = self.network_build(channels,kernels_size,
                kernels_dim,kernels_stride,kernels_pad)
    def forward(self,img):
        x = img
        for CNN,BN in zip(self.CNNs,self.BNs):
            x = CNN(x)
            x = BN(x)
            x = self.pReLU(x)
        return x 
    def network_build(self,channels,kernels_size,kernels_dim,kernels_stride,kernels_pad):
        CNNs = nn.ModuleList()
        BNs = nn.ModuleList()
        for i in range(len(kernels_size)):
            kernel_dim = kernels_dim[i]
            kernel_size = kernels_size[i]
            kernel_stride = kernels_stride[i]
            kernel_pad = kernels_pad[i]
            if i==0:
                CNNs.append(nn.Conv3d(channels,kernel_dim,kernel_size,kernel_stride,kernel_pad))
                BNs.append(nn.BatchNorm3d(kernel_dim))
            else:
                CNNs.append(nn.Conv3d(kernels_dim[i-1],kernel_dim,kernel_size,kernel_stride,kernel_pad))
                BNs.append(nn.BatchNorm3d(kernel_dim))
        return CNNs,BNs  
    def receptive_field_cal(self):
        kernels_size = self.kernels_size
        kernels_stride = self.kernels_stride 
        receptive_strides = [1]
        receptive_fileds = [1]
        for i,kernel_size in enumerate(kernels_size):
            kernel_stride = kernels_stride[i]
            receptive_stride = receptive_strides[i]*kernel_stride
            receptive_strides.append(kernel_stride)
            receptive_filed = receptive_fileds[i] + \
                (kernel_size - 1)*receptive_strides[i]
            receptive_fileds.append(receptive_filed)
        return receptive_fileds,receptive_strides     
class ResBlock(nn.Module):
    def __init__(self,channels = 4,
            kernels_dim = [30,30],
            kernels_size = [3,3],
            kernels_stride = [1,1],
            kernels_pad = [1,1]
            ):
        super(ResBlock,self).__init__()
        self.channels = channels 
        self.kernels_dim = kernels_dim
        self.kernels_size = kernels_size
        self.kernels_stride = kernels_stride
        self.kernels_pad = kernels_pad 
        self.receptive_fields,self.receptive_strides = self.receptive_field_cal()
        

        self.pReLU = nn.PReLU()
        self.CNNs,self.BNs,self.CNN_res,self.BN_res = self.network_build(channels,kernels_size,
                kernels_dim,kernels_stride,kernels_pad)
    def forward(self,img):
        x = img
        for CNN,BN in zip(self.CNNs[:-1],self.BNs[:-1]):
            x = CNN(x)
            x = BN(x)
            x = self.pReLU(x)
        x = self.CNNs[-1](x)
        x = self.BNs[-1](x)
        
        x2 = self.CNN_res(img)
        x2 = self.BN_res(x2)
        x += x2
        x = self.pReLU(x)
        return x 


    def receptive_field_cal(self):
        kernels_size = self.kernels_size
        kernels_stride = self.kernels_stride 
        receptive_strides = [1]
        receptive_fields = [1]
        #print(kernels_size)
        #print(kernels_stride)
        for i,kernel_size in enumerate(kernels_size):
            kernel_stride = kernels_stride[i]

            receptive_stride = receptive_strides[i]*kernel_stride
            receptive_strides.append(kernel_stride)

            receptive_field = receptive_fields[i] + \
                (kernel_size - 1)*receptive_strides[i]
            
            receptive_fields.append(receptive_field)
        return receptive_fields,receptive_strides     

    def network_build(self,channels,kernels_size,kernels_dim,kernels_stride,kernels_pad):
        stride = self.receptive_strides[-1]
        CNN_res = nn.Conv3d(channels,kernels_dim[-1],1,stride)
        BN_res = nn.BatchNorm3d(kernels_dim[-1])
        CNNs = nn.ModuleList()
        BNs = nn.ModuleList()
        for i in range(len(kernels_size)):
            kernel_dim = kernels_dim[i]
            kernel_size = kernels_size[i]
            kernel_pad = kernels_pad[i]
            kernel_stride = kernels_stride[i]
            if i==0:
                CNNs.append(nn.Conv3d(channels,kernel_dim,kernel_size,kernel_stride,kernel_pad))
                BNs.append(nn.BatchNorm3d(kernel_dim))
            else:
                CNNs.append(nn.Conv3d(kernels_dim[i-1],kernel_dim,kernel_size,kernel_stride,kernel_pad))
                BNs.append(nn.BatchNorm3d(kernel_dim))
        return CNNs,BNs,CNN_res,BN_res 

class SubFCN(nn.Module):
    def __init__(self,subFactor,num_classes = 0,channels = 4,
            kernels_dim= [30,30,40,40,40,40,50,50],
            kernels_size = [3,3,3,3,3,3,3,3],
            kernels_stride = [1,1,1,1,1,1,1,1],
            kernels_pad = [1,1,1,1,1,1,1,1],
            layers_connect = [4,6,8]):
        super(SubFCN,self).__init__()
         
        self.FCN = BasicFCN(num_classes,channels,kernels_dim,
                kernels_size,kernels_stride,kernels_pad,layers_connect)
        
        self.downsample = nn.AvgPool3d(subFactor)

        pad = sum(self.FCN.receptive_pads)

        pad = (subFactor-1)*pad 
        self.upsample = nn.ConvTranspose3d(kernels_dim[-1],
                kernels_dim[-1],subFactor,subFactor,pad)
    def forward(self,img):
        sub_img = self.downsample(img)
        output = self.FCN(sub_img)
        up_output = self.upsample(output)
        return up_output 

class DeepMedic(nn.Module):
    def __init__(self,subFactor,num_classes = 0,channels = 4,
            kernels_dim= [30,30,40,40,40,40,50,50],
            kernels_size = [3,3,3,3,3,3,3,3],
            kernels_stride = [1,1,1,1,1,1,1,1],
            kernels_pad = [1,1,1,1,1,1,1,1],
            layers_connect = [4,6,8],
            FCs_dim = [150,150]):
        super(DeepMedic,self).__init__()
        self.FCN = BasicFCN(num_classes,channels,kernels_dim,
                kernels_size,kernels_stride,kernels_pad,layers_connect)
        self.subFCN = SubFCN(subFactor,num_classes,channels,
                kernels_dim,kernels_size,kernels_stride,kernels_pad,layers_connect)
        self.FCs = nn.ModuleList()
        self.BNs = nn.ModuleList()
        for i,FC_dim in enumerate(FCs_dim):
            if i==0:
                self.FCs.append(nn.Linear(2*kernels_dim[-1],FC_dim))
                self.BNs.append(nn.BatchNorm3d(FC_dim))
            else:
                self.FCs.append(nn.Linear(FCs_dim[i-1],FC_dim))
                self.BNs.append(nn.BatchNorm3d(FC_dim))

        
        self.receptive_fields,self.receptive_strides = self.receptive_field_cal()

        #receptive_fields = self.receptive_fields
        #receptive_strides = self.receptive_strides 
        
        self.classifier = nn.Linear(FCs_dim[-1],num_classes)
        #self.up = nn.ConvTranspose3d(num_classes,num_classes,
        #        receptive_fields[-1],receptive_strides[-1])
        
        #self.BN_1 = nn.BatchNorm3d()
        
        self.pReLU = nn.PReLU()
    def receptive_field_cal(self):
        return self.FCN.receptive_field_cal()
    def forward(self,img,img2):
        
        feature_map = self.FCN(img)
        #print(feature_map.size())
        feature_map2 = self.subFCN(img2)
        #print(feature_map2.size())
        feature_map = torch.cat([feature_map,feature_map2],dim = 1)
        #print(feature_map.size())
        feature_map = feature_map.permute(0,2,3,4,1)
        #feature_map = torch.transpose(torch.unsqueeze(feature_map,5),1,5).squeeze()
        #print(feature_map.size())
        for FC,BN in zip(self.FCs,self.BNs):
            feature_map = FC(feature_map)
            feature_map = feature_map.permute(0,4,1,2,3)
            feature_map = BN(feature_map.contiguous())
            feature_map = feature_map.permute(0,2,3,4,1)
            feature_map = self.pReLU(feature_map)
        #    print(feature_map.size())
        feature_map = self.classifier(feature_map)
        feature_map = feature_map.permute(0,4,1,2,3)
        #print(feature_map.size())
        #feature_map = self.up(feature_map)

        return feature_map 

class BaseDeepMedic(nn.Module):
    def __init__(self,subFactor,num_classes = 0,channels = 4,
            kernels_size = [3,3,3,3,3,3,3,3],
            kernels_stride = [1,1,1,1,1,1,1,1],
            kernels_dim= [30,30,40,40,40,40,50,50],
            FCs_dim = [150,150]):
        super(BaseDeepMedic,self).__init__()
        self.FCN = BasicFCN(num_classes,channels,kernels_size,
                kernels_stride,kernels_dim)
        self.subFCN = SubFCN(subFactor,num_classes,channels,
                kernels_size,kernels_stride,kernels_dim)
        self.FCs = nn.ModuleList()
        for i,FC_dim in enumerate(FCs_dim):
            if i==0:
                self.FCs.append(nn.Linear(kernels_dim[-1],FC_dim))
            
            else:
                self.FCs.append(nn.Linear(FCs_dim[i-1],FC_dim))
        self.FCs.append(nn.Linear(FCs_dim[-1],num_classes))
        
        self.receptive_fields,self.receptive_strides = self.receptive_field_cal()

        receptive_fields = self.receptive_fields
        receptive_strides = self.receptive_strides 

        self.up = nn.ConvTranspose3d(num_classes,num_classes,
                receptive_fields[-1],receptive_strides[-1])
        
        self.pReLU = nn.PReLU()
    def receptive_field_cal(self):
        return self.FCN.receptive_field_cal()
    def forward(self,img,img2):
        
        feature_map = self.FCN(img)
        feature_map2 = self.subFCN(img2)
        
        feature_map = feature_map.permute(0,2,3,4,1)
        for FC in self.FCs:
            feature_map = FC(feature_map)
    
            feature_map = self.pReLU(feature_map)
        #    print(feature_map.size())
        feature_map = feature_map.permute(0,4,1,2,3)
        #print(feature_map.size())
        feature_map = self.up(feature_map)

        return feature_map 

import torch
import torch.nn as nn
#from .clstm import ConvLSTMCell
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
import sys

def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            #print(v)
            layers += [nn.MaxPool3d(2,2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.LeakyReLU(1e-5)]
            else:
                layers += [conv3d, nn.LeakyReLU(1e-5)]
            in_channels = v
    return nn.Sequential(*layers)

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self ,input_size, hidden_size):
        super(ConvLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv3d(input_size + hidden_size, 4 * hidden_size, 3,1,1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]

        return state


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
class ConvPool(nn.Module):
    def __init__(self,in_dims = 32,out_dims = 32):
        super(ConvPool,self).__init__()
        self.pool = nn.Sequential(nn.Conv3d(in_dims,out_dims,3,2,1),\
                nn.BatchNorm3d(out_dims),nn.LeakyReLU(1e-5))
    def forward(self,input_):
        return self.pool(input_)
class RSIS(nn.Module):
    def __init__(self):
        super(RSIS,self).__init__()
        #self.feature = make_layers([1,32,"M",32,32,"M",32,32])
        self.feature = nn.Sequential(make_layers([1,32,32]),ConvPool(32,64),\
                VoxRes([64,64,64]),VoxRes([64,64,64]),\
                ConvPool(64,64),VoxRes([64,64,64]),VoxRes([64,64,64]))

        self.decoder = ConvLSTMCell(64,64)
        
        #self.conv_out = nn.Sequential(make_layers([64,64]),nn.Conv3d(64,2,1,1,0))
        self.conv_out = nn.Conv3d(64,2,1,1,0)
        self.upsample = nn.ConvTranspose3d(2,2,4,4,0)
        self.fc_pool = nn.MaxPool3d(20,20)
        self.fc_stop = nn.Linear(64,1)
        self.reset_params()
    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                #m.weight.data.normal(0,0.01)
                nn.init.normal_(m.weight,0,0.01)
                m.bias.data.zero_()
    def forward(self,input_,prev_state):
        feature = self.feature(input_)
        state = self.decoder(feature,prev_state)
        hidden = state[0]

        output = self.conv_out(hidden)
        output = self.upsample(output)
        
        stop_feature = self.fc_pool(hidden)
        stop_prob = self.fc_stop(stop_feature.view(hidden.size(0),-1))
        return output,stop_prob,state
if __name__ == "__main__":
    device = torch.device("cuda")
    img = torch.rand(1,1,80,80,80).to(device)
    model = RSIS()
    model = model.to(device)
    outputs = model(img,None)
    print(outputs[0].size(),outputs[1].size())

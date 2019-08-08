import torch
import torch.nn as nn

class FloodFillingNetwork_(nn.Module):
    """ Flood filling network model.

    """
    def __init__(self, in_planes=2, module_nums=8):
        super(FloodFillingNetwork_, self).__init__()

        self.module_nums = module_nums
        self.in_planes = in_planes
        
        self.relu = nn.ReLU(inplace=True)
        self.Convhead = nn.Conv3d(self.in_planes, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.ConvModule = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.Conv1 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0, bias=True) 
        self.Conv2 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True) 

    def forward(self, x):
        conv = self.Convhead(x)

        for _ in range(self.module_nums):
            c = self.ConvModule(conv)
            c = self.relu(c)
            
            c = self.ConvModule(c)

            conv = self.relu(c + conv) 

        conv = self.Conv1(conv)
        # conv = self.relu(conv)
        # conv = self.Conv2(conv)

        return conv

class FloodFillingNetwork(nn.Module):
    """ Flood filling network model.

    """
    def __init__(self, in_planes=2, module_nums=16):
        super(FloodFillingNetwork, self).__init__()

        self.module_nums = module_nums
        self.in_planes = in_planes
        
        self.relu = nn.ReLU(inplace=True)
        self.Conv1 = nn.Conv3d(self.in_planes, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.Conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.ConvModule = [ResBlock()] * self.module_nums
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.res6 = ResBlock()
        self.res7 = ResBlock()
        self.res8 = ResBlock()

        self.ConvHead = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()
        # self.activation = nn.LogSoftmax()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.relu(out)
        out = self.Conv2(out)
        out = self.relu(out)

        # for r in self.ConvModule:
        #     out = r(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)

        out = self.ConvHead(out)

        # out = self.activation(out)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ResBlock(nn.Module):
    def __init__(self, in_planes=32, planes=32, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.relu(out + residual)
        return out
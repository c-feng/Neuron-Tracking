import torch
import torch.nn as nn
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))

from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
from lib.config_one_data import cfg

class NeuralPointNet(nn.Module):
    def __init__(self, input_channels=0, use_xyz=True, mode="TRAIN"):
        super().__init__()
        
        # 特征提取
        # self.backbone_net = pointnet2_msg.get_model()
        npoints = cfg.NET.SA_NPOINTS
        radiis = cfg.NET.RADIIS
        nsamples = cfg.NET.NSAMPLES
        mlps = cfg.NET.MLPS

        # c_in = input_channels
        # self.SA_module1 = PointnetSAModuleMSG(npoint=npoints[0], radii=radiis[0], nsamples=nsamples[0],
        #                         mlps=[[c_in, *mlps[0][0]], [c_in, *mlps[0][1]], [c_in, *mlps[0][2]]], use_xyz=use_xyz, bn=True)
        # c_out_1 = sum(mlps[0][:][-1])

        # c_in = c_out_1
        # self.SA_module2 = PointnetSAModuleMSG(npoint=npoints[1], radii=radiis[1], nsamples=nsamples[1],
        #                         mlps=[[c_in, *mlps[1][0]], [c_in, *mlps[1][1]], [c_in, *mlps[1][2]]], use_xyz=use_xyz, bn=True)
        # c_out_2 = sum(mlps[1][:][-1])
        
        # c_in = c_out_2
        # self.SA_module3 = PointnetSAModuleMSG(npoint=npoints[2], radii=radiis[2], nsamples=nsamples[2],
        #                         mlps=[[c_in, *mlps[2][0]], [c_in, *mlps[2][1]], [c_in, *mlps[2][2]]], use_xyz=use_xyz, bn=True)
        # c_out_3 = sum(mlps[2][:][-1])

        # c_in = c_out_3
        # self.SA_module4 = PointnetSAModuleMSG(npoint=npoints[3], radii=radiis[3], nsamples=nsamples[3],
        #                         mlps=[[c_in, *mlps[3][0]], [c_in, *mlps[3][1]], [c_in, *mlps[3][2]]], use_xyz=use_xyz, bn=True)
        # c_out_4 = sum(mlps[3][:][-1])

        c_in = input_channels
        c_out = []
        sa_module = []
        for i in range(len(npoints)):
            c_o = 0
            for m in mlps[i]:
                m[:] = [c_in, *m]
                c_o += m[-1]
            c_out.append(c_o)
            sa_module.append(
                PointnetSAModuleMSG(npoint=npoints[i], radii=radiis[i], nsamples=nsamples[i],
                                    mlps=mlps[i], use_xyz=use_xyz, bn=True)
            )
            c_in = c_out[-1]
        
        self.SA_module1 = sa_module[0]
        self.SA_module2 = sa_module[1]
        self.SA_module3 = sa_module[2]
        self.SA_module4 = sa_module[3]
        c_out_1, c_out_2, c_out_3, c_out_4 = c_out
    
        self.FP_module1 = PointnetFPModule(mlp=[c_out_4+c_out_3, 512, 512], bn=True)
        self.FP_module2 = PointnetFPModule(mlp=[512+c_out_2, 512, 512], bn=True)
        self.FP_module3 = PointnetFPModule(mlp=[512+c_out_1, 256, 256], bn=True)
        self.FP_module4 = PointnetFPModule(mlp=[256+input_channels, 128, 128], bn=True)

        self.ins_fc1 = pt_utils.Conv1d(128, 64, kernel_size=1, bn=True)
        # self.ins_dp = nn.Dropout(0.1)
        self.ins_fc2 = pt_utils.Conv1d(64, 8, kernel_size=1, bn=False, activation=None)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features

    def forward(self, input_data: torch.cuda.FloatTensor):
        
        xyz, features = self._break_up_pc(input_data)

        l_xyz, l_features = [xyz], [features]
        for i in range(4):
            li_xyz, li_features = eval("self.SA_module"+str(i+1))(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        for i in range(-1, -(4 + 1), -1):
            l_features[i-1] = eval("self.FP_module"+str(-i))(l_xyz[i-1], l_xyz[i],
                                                              l_features[i-1], l_features[i])
        
        ins = self.ins_fc1(l_features[0])
        # ins = self.ins_dp(ins)
        ins = self.ins_fc2(ins)

        return ins

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import pdb
    import time
    import torch.optim as optim
    from utils.loss_utils import discriminative_loss

    net = NeuralPointNet(0)
    net.cuda()
    net.train()
    
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    with torch.autograd.set_detect_anomaly(True):
        for i in range(1000000):
            inputs = torch.rand(2, 15000, 3).cuda()
            labels = torch.randint(1, 6, [2, 15000]).float().cuda()

            out = net(inputs)
            out = out.permute(0, 2, 1)
            # print(net)
            print(out.shape)

            optimizer.zero_grad()
            loss, lv, ld, lr = discriminative_loss(out, labels)
            loss.backward()
            optimizer.step()
            print("disc loss: ", loss.data[0])
            print("loss variance: ", lv)
            print("loss distance: ", ld)
            print("loss regularization: ", lr)
            

            time.sleep(8)



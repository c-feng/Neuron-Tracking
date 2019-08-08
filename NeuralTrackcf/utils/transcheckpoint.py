import os.path as osp
from collections import OrderedDict
import torch

def transcheckpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location='cpu')
        cp = dict()
        for k, v in checkpoint.items():
            if k == "state_dict":
                cp[k] = changeDict_keys(v)
            else:
                cp[k] = v

        print("=> Loaded checkpoint '{}'".format(fpath))
        return cp
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def changeDict_keys(dicta):
    dic = OrderedDict()
    for k, v in dicta.items():
        key = "module." + k
        dic[key] = v
    
    return dic

def trans_Distributed2Parallel(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location='cpu')
        cp = dict()
        for k, v in checkpoint.items():
            if k == "state_dict":
                cp[k] = changeDict_keys_v1(v)
            else:
                cp[k] = v

        print("=> Loaded checkpoint '{}'".format(fpath))
        return cp
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def changeDict_keys_v1(dicta):
    dic = OrderedDict()
    for k, v in dicta.items():
        key = '.'.join(k.split('.')[1:])
        dic[key] = v

    return dic


if __name__ == "__main__":
    fpath = r"C:\Users\Administrator\Desktop\model_best.pth.tar"
    origin_cp = torch.load(fpath, map_location='cpu')
    checkpoint = transcheckpoint(fpath)
    print(origin_cp.keys())
    print(checkpoint.keys())
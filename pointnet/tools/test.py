import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import logging
import numpy as np

import _init_paths
# from lib.net.neural_ins import NeuralPointNet
from lib.net.neural_ins_syn import NeuralPointNet
from lib.datasets.point_dataset import PointDataset
from lib.config import cfg
from utils.clustering import cluster
from utils.show3d_balls import showpoints, ncolors
from tqdm import tqdm

parser = argparse.ArgumentParser(description="arg parser")
# parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', type=str, default=None, required=True, help='whether to use multiple gpu')
parser.add_argument('--bandwidth', type=float, default=1., help='Bandwidth for meanshift clustering [default: 1.]')
parser.add_argument('--model_path', type=str, default="./logs/log_syn/ckpt/checkpoint_epoch_1000.pth", help='whether to train with evaluation')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

args = parser.parse_args()

if args.mgpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus

def create_dataloader():
    # DATA_PATH = "/home/fcheng/Neuron/pointnet/data/real_data/"
    DATA_PATH = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/"

    test_set = PointDataset(data_root=DATA_PATH, mode="train", augment=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                                 num_workers=args.workers, shuffle=True, collate_fn=test_set.collate_batch)    
    return test_loader, test_set


def test():
    
    output_dir = "/home/fcheng/Neuron/pointnet/eval_logs/syn/syn_FPS_32768_predLabel/"
    os.makedirs(output_dir, exist_ok=True)
    # create dataloader & network & optimizer
    _, test_set = create_dataloader()
    model = NeuralPointNet(input_channels=0)

    if args.mgpus is not None and len(args.mgpus) > 2:
        model = nn.DataParallel(model)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state'])

    for i, data in enumerate(tqdm(test_set)):
        point_sets, ins = data
        name = test_set.dataNames[i]
        point_sets = point_sets[None]

        point_sets = torch.from_numpy(point_sets).cuda().float()
        # ins = torch.from_numpy(ins).cuda().float()

        embeded_feature = model(point_sets)
        
        embeded_feature = embeded_feature.permute(0, 2, 1)[0]
        embeded_feature = embeded_feature.cpu().detach().numpy()

        num_clusters, labels, cluster_centers = cluster(embeded_feature)

        colors = ncolors(num_clusters)

        print(name, "\n", np.unique(ins))
        print(num_clusters, np.unique(labels), "\n")
        # showpoints(point_sets.data.cpu().numpy()[0], c_gt=colors[labels], normalizecolor=False, ballradius=2)
        np.save(os.path.join(output_dir, name+".npy"), labels)

def test_showpoints():
    file_paths = "/home/fcheng/Neuron/pointnet/eval_logs/syn/syn_FPS_32768_predLabel/"

    DATA_PATH = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/"
    data_set = PointDataset(data_root=DATA_PATH, mode="train", augment=True)
    # for i in data_set.__len__():
    i = 0
    name = data_set.dataNames[i]
    point_sets, _ = data_set[i]

    labels = np.load(os.path.join(file_paths, name+".npy"))
    uni_labels = np.unique(labels)
    
    colors = ncolors(len(uni_labels))
    showpoints(point_sets, c_gt=colors[labels], normalizecolor=False, background=(255,255,255), ballradius=2)



if __name__ == "__main__":
    # test()
    test_showpoints()







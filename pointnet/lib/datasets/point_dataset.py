import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH, "../../"))

import lib.datasets.pc_augment as pc_augment


class PointDataset(Dataset):
    def __init__(self, data_root, mode="train", transform=None, augment=True, tag="Neural"):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.augment = augment

        self.cache = {}
        self.cache_size = 20000

        self.dataNames = self.getDataname()

    def getDataname(self):
        names = os.listdir(os.path.join(self.data_root, self.mode))
        names.sort()
        names = [name.split(".")[0] for name in names]

        return names

    def _augment_data(self, data):
        # rotated_data = provider.rotate_point_cloud(data)
        # rotated_data, _ = pc_augment.rotate_point_cloud(data)
        rotated_data, _ = pc_augment.rotate_point_cloud_z(data)
        # rotated_data, _ = pc_augment.rotate_perturbation_point_cloud(rotated_data)
        # rotated_data = pc_augment.random_scale_point_cloud(rotated_data)
        # rotated_data = pc_augment.shift_point_cloud(rotated_data)
        rotated_data = pc_augment.jitter_point_cloud(rotated_data)
        rotated_data, shuffle_idx = pc_augment.shuffle_points(rotated_data)
        # shuffle_idx = None
        return rotated_data, shuffle_idx

    def readData(self, idx):
        if idx in self.cache:
            point_set, ins = self.cache[idx]
        else:
            name = self.dataNames[idx]
            path = os.path.join(self.data_root, self.mode, name+".npy")
            pc_label = np.load(path)
            point_set = pc_label[:, :3]
            ins = pc_label[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (point_set, ins)
        
        if self.augment:
            point_set, shuffle_idx = self._augment_data(point_set)
            if shuffle_idx is not None:
                ins = ins[shuffle_idx]

        return point_set, ins

    def __len__(self):
        return len(self.dataNames)
    
    def __getitem__(self, idx):
        point_set, ins = self.readData(idx)
        # name = self.dataNames[idx]

        if self.transform is not None:
            point_set, ins = self.transform(point_set, ins)
        
        return point_set, ins

    def collate_batch(self, batch):
        """ batch is a list, length is batch_size
            point_batch: B x N x 3
            ins_batch: B x N
        """
        point_batch = []
        ins_batch = []
        for b in batch:
            point_batch.append(b[0])
            ins_batch.append(b[1])
        # point_batch, ins_batch = zip(*batch)
        # point_batch = torch.stack([torch.from_numpy(b) for b in batch])
        return np.stack(point_batch, 0), np.stack(ins_batch, 0)

if __name__ == "__main__":
    
    from utils.show3d_balls import showpoints, ncolors
    import argparse

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--idx", type=int, default=0, required=True, help="batch size for training")  # 68
    args = parser.parse_args()

    DATA_PATH = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/"
    train_set = PointDataset(data_root=DATA_PATH, mode="train", augment=True)

    idx = args.idx
    point_sets, ins = train_set[idx]
    name = train_set.dataNames[idx]

    uni_l = np.unique(ins)
    ordered_labels = np.zeros_like(ins)
    for cnt, l in enumerate(uni_l):
        ordered_labels[ins==l] = cnt
    
    colors = ncolors(len(uni_l))
    
    print(name)
    print(np.unique(ordered_labels))
    showpoints(point_sets, c_gt=colors[ordered_labels], background=(255,255,255), normalizecolor=False, ballradius=2)

    




import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.external import tifffile
import json
import numpy as np
from matplotlib import pyplot as plt

MEAN = 153.98155410163966 
STD = 47.97406135952655


def read_json(fpath):
    dir = os.path.dirname(fpath)
    with open(fpath, 'r') as f:
        return json.load(f)


class NeuralData(Dataset):
    def __init__(self, list_file, data_root, transform=None, dataset_name='Neural'):
        self.list_id = self.get_ids(list_file)
        self.tif_paths = read_json(list_file)['tif']
        self.gt_paths = read_json(list_file)['gt']
        self.ins_paths = read_json(list_file)['ins']
        self.data_root = data_root
        self.name = dataset_name
        self.transform = transform

    def __len__(self):
        return len(self.list_id)
    
    def __getitem__(self, idx):

        img = tifffile.imread(os.path.join(self.data_root, self.tif_paths[idx]))
        img = (img - MEAN) / STD
        img = img[:300, :300, :300]
        img = np.expand_dims(img, axis=0)

        gt = tifffile.imread(os.path.join(self.data_root, self.gt_paths[idx]))
        gt = gt[:300, :300, :300]
        gt = np.expand_dims(gt, axis=0)

        ins = tifffile.imread(os.path.join(self.data_root, self.ins_paths[idx]))
        ins = ins[:300, :300, :300]

        return torch.from_numpy(gt.astype(np.float)), torch.from_numpy(ins.astype(np.float)), torch.from_numpy(img.astype(np.float))

    def get_ids(self, file_path):
        dict_paths = read_json(file_path)
        tif_paths = dict_paths['tif']
        tif_ids = [os.path.split(i)[1].split('.')[0] for i in tif_paths]

        return tif_ids

    def get_img_name(self, idx):
        return self.list_id[idx]

    def pull_item(self, idx):
        img = tifffile.imread(os.path.join(self.data_root, self.tif_paths[idx]))
        gt_box = self.generate_gtbox(os.path.join(self.data_root, self.json_paths[idx]), label=0)
        temp = 0
        while(not gt_box):
            if temp == (len(self.list_id) - 1):
                idx = 0
            else:
                temp = idx
                idx = min(idx+1, len(self.list_id)-1)
            img = tifffile.imread(os.path.join(self.data_root, self.tif_paths[idx]))
            gt_box = self.generate_gtbox(os.path.join(self.data_root, self.json_paths[idx]), label=0)

        img0 = np.sum(img, axis = 0) > 0
        img1 = np.sum(img, axis = 1) > 0
        img2 = np.sum(img, axis = 2) > 0
        
        img0 = img0.astype(np.int)
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        img0[img0>0] = 1
        img1[img1>0] = 1
        img2[img2>0] = 1
        img_ = np.stack([img0, img1, img2], axis=0)
        height, width, length = img.shape

        img = (img>0).astype(np.int)
        img = np.expand_dims(img, axis=0)

        #gt_box = np.array(gt_box)
        return torch.from_numpy(img), gt_box, height, width, length, torch.from_numpy(img_)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    gts = []
    targets = []
    imgs = []
    for sample in batch:
        gts.append(sample[0])
        targets.append(sample[1])
        imgs.append(sample[2])
    return torch.stack(gts, 0), torch.stack(targets, 0), torch.stack(imgs, 0)

def func1():
    Neural_ROOT = "/home/jjx/Biology/DirectField/data_modified/"
    List_file = "./test.json"
    dataset = NeuralData(list_file=List_file,
                        data_root=Neural_ROOT)
    dataloader = DataLoader(dataset, 4,
                                  num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    num = 5
    start_idx = 50
    for i, (img, gt) in enumerate(dataloader):
        # img, annotation, _, _, img_ = dataset.pull_item(start_idx+i)
        if i > 5:
            break
        # img, gt = img[0], img[1]
        print(type(img), type(gt))
        print(img.shape, gt.shape)
        print(np.unique(gt))
    
if __name__ == "__main__":
    func1()

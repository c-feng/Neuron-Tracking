import os
import numpy as np
from skimage.external import tifffile

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

class NeuralDataset(Dataset):
    def __init__(self, data_root, mode="train", transform=None,
                 sample_shape=[120, 120, 120], sample_num=9, tag="Neural"):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.name = tag
        self.datashape = [301, 301, 301]
        self.sample_shape = sample_shape
        self.sample_num = sample_num

        self.datanames = self.getDataname(self.data_root)

    def resetSample(self):
        self.sampleInfo = self.sampleCropdata(self.datanames, self.sample_shape, self.sample_num)
    
    def getDataname(self, data_root):
        names = os.listdir(os.path.join(data_root, "tiffs"))
        names.sort()
        names = [name.split('.')[0] for name in names]
        num = len(names)
        if self.mode == "train":
            return names[:int(num*0.8)]
        elif self.mode == "eval":
            return names[int(num*0.8):]
    
    def sampleCropdata(self, datanames, sam_shape, num):
        shape = np.array(self.datashape)
        sam_shape = np.array(sam_shape)

        gap = shape - sam_shape + 1
        infos = []
        for name in datanames:
            x = np.random.choice(np.arange(gap[0]), size=num, replace=False)
            y = np.random.choice(np.arange(gap[1]), size=num, replace=False)
            z = np.random.choice(np.arange(gap[2]), size=num, replace=False)
            
            xyz = np.stack([x,y,z], axis=1).tolist()
            info = [[name]+i for i in xyz]
            infos += info

        return infos

    def readData(self, idx):
        info = self.sampleInfo[idx]
        name = info[0]
        coords = info[1:]

        img_path = os.path.join(self.data_root, "tiffs", name+".tif")
        # ins_path = os.path.join(self.data_root, "ins_modified", name+".tif")
        ins_path = os.path.join(self.data_root, "ins", name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        whd = self.sample_shape
        img = img[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        ins = ins[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]

        return img, ins, info

    def __len__(self):
        return len(self.sampleInfo)
    
    def __getitem__(self, idx):
        img, ins, info = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins, info



if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    data_root = r"H:\dataset\Neural\data_modified"
    data_root = "/home/jjx/Biology/data/data_modified/"
    data_root = "/home/jjx/Biology/data/data_synthesize/"
    
    dataset = NeuralDataset(data_root, sample_shape=[300, 300, 300], sample_num=1)
    dataset.resetSample()
    print("reset completed")

    point_sum = []
    nums = dataset.__len__()
    for i in tqdm(range(nums)):
        _, b, _ = dataset.readData(i)
        print("unique: ", np.unique(b))
        print("shape: ", b.shape, np.sum(b>0))
        print("\n")
        point_sum.append(np.sum(b>0))


    print("Average points in a [300, 300, 300] is {}".format(np.mean(point_sum)))
    print("Min is {}".format(np.min(point_sum)))
    print("Max is {}".format(np.max(point_sum)))
    np.save("point_num_syn.npy", np.array(point_sum))
    plt.hist(point_sum)
    plt.show()

    print("*****")

    #####################################################
    # import matplotlib.pyplot as plt
    # point_sum = np.load("point_num.npy")
    # plt.hist(point_sum, bins=100, density=1)
    # plt.show()

# point取 15000
# Average points in a [300, 300, 300] is 12723.71052631579
# Min is 4858
# Max is 33390

    #####################################################

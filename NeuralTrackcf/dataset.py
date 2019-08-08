import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from skimage.external import tifffile
import numpy as np
import pdb

INS_PATH = "/media/jjx/Biology/data/data_synthesize/ins/"


class DatasetSyn(Dataset):
    def __init__(self, data_root, mode="train", transform=None, sample_shape=[96, 96, 96], sample_num=9, dataset_name="Neural"):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.name = dataset_name
        self.datashape = [300, 300, 300]
        self.sample_shape = sample_shape
        self.sample_num = sample_num

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)
    
    def reset_Sample(self):
        self.sampleInfo = self.samplecropdata(self.datanames, self.sample_shape, self.sample_num)

    def getDataName(self, data_root):
        names = os.listdir(data_root)
        names.sort()
        num = len(names)
        if self.mode == "train":
            return names[:int(num*0.8)]
        elif self.mode == "eval":
            return names[int(num*0.8):]

    def samplecropdata(self, datanames, sam_shape=[96, 96, 96], num=3):
        shape = np.array(self.datashape)
        sam_shape = np.array(sam_shape)
        # margin = (shape - sam_shape) // 2
        # interval = [margin - 1, shape - margin]
        gap = shape - sam_shape + 1
        infos = []
        for name in datanames:
            # x = np.random.randint(low=interval[0, 0], high=interval[1, 0], size=num)
            # y = np.random.randint(low=interval[0, 1], high=interval[1, 1], size=num)
            # z = np.random.randint(low=interval[0, 2], high=interval[1, 2], size=num)
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

        img_path = os.path.join(self.data_root, name, "noises", name+'.tif')
        ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 4.

        whd = self.sample_shape
        img = img[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        ins = ins[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]] 
        assert img.shape == tuple(self.sample_shape), pdb.set_trace()
                # "The img's shape is {} are not equal to expected {}".format(img.shape, self.sample_shape)
        assert ins.shape == tuple(self.sample_shape), \
                "The ins's shape is {} are not equal to expected {}".format(ins.shape, self.sample_shape)
        return img, ins, info

    def __len__(self):
        return len(self.sampleInfo)
    
    def __getitem__(self, idx):
        img, ins, info = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins, info

class DatasetTrain(Dataset):
    def __init__(self, data_root, transform=None, sample_shape=[96, 96, 96], sample_num=9, dataset_name="Neural"):
        self.data_root = data_root
        self.transform = transform
        self.name = dataset_name
        self.datashape = [300, 300, 300]
        self.sample_shape = sample_shape
        self.sample_num = sample_num

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)
    
    def reset_Sample(self):
        self.sampleInfo = self.samplecropdata(self.datanames, self.sample_shape, self.sample_num)

    def getDataName(self, data_root):
        names = os.listdir(data_root)
        names.sort()
        num = len(names)
        return names[:int(num*0.8)]

    def samplecropdata(self, datanames, sam_shape=[96, 96, 96], num=3):
        shape = np.array(self.datashape)
        sam_shape = np.array(sam_shape)
        # margin = (shape - sam_shape) // 2
        # interval = [margin - 1, shape - margin]
        gap = shape - sam_shape + 1
        infos = []
        for name in datanames:
            # x = np.random.randint(low=interval[0, 0], high=interval[1, 0], size=num)
            # y = np.random.randint(low=interval[0, 1], high=interval[1, 1], size=num)
            # z = np.random.randint(low=interval[0, 2], high=interval[1, 2], size=num)
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

        img_path = os.path.join(self.data_root, name, "noises", name+'.tif')
        # ins_path = os.path.join(self.data_root, name, "gts", name+".tif")
        ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)

        whd = self.sample_shape
        img = img[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        ins = ins[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]] 
        assert img.shape == tuple(self.sample_shape), \
                "The img's shape is {} are not equal to expected {}".format(img.shape, self.sample_shape)
        assert ins.shape == tuple(self.sample_shape), \
                "The ins's shape is {} are not equal to expected {}".format(ins.shape, self.sample_shape)
        return img, ins

    def __len__(self):
        return len(self.sampleInfo)
    
    def __getitem__(self, idx):
        img, ins = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins

class DatasetEvaluate(Dataset):
    def __init__(self, data_root, transform=None, sample_shape=[96, 96, 96], sample_num=4, dataset_name="Neural"):
        self.data_root = data_root
        self.transform = transform
        self.name = dataset_name
        self.datashape = [300, 300, 300]
        self.sample_shape = sample_shape
        self.sample_num = sample_num

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)

    def reset_Sample(self):
        self.sampleInfo = self.samplecropdata(self.datanames, self.sample_shape, self.sample_num)

    def getDataName(self, data_root):
        names = os.listdir(data_root)
        names.sort()
        num = len(names)
        return names[int(num*0.8):]

    def samplecropdata(self, datanames, sam_shape=[96, 96, 96], num=3):
        shape = np.array(self.datashape)
        sam_shape = np.array(sam_shape)
        # margin = (shape - sam_shape) // 2
        # interval = [margin - 1, shape - margin]
        gap = shape - sam_shape + 1
        infos = []
        for name in datanames:
            # x = np.random.randint(low=interval[0, 0], high=interval[1, 0], size=num)
            # y = np.random.randint(low=interval[0, 1], high=interval[1, 1], size=num)
            # z = np.random.randint(low=interval[0, 2], high=interval[1, 2], size=num)
            # 左上角顶点
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

        img_path = os.path.join(self.data_root, name, "noises", name+'.tif')
        # ins_path = os.path.join(self.data_root, name, "gts", name+".tif")
        ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)

        whd = self.sample_shape
        img = img[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        ins = ins[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]] 
        assert img.shape == tuple(self.sample_shape), \
                "The img's shape is {} are not equal to expected {}".format(img.shape, self.sample_shape)
        assert ins.shape == tuple(self.sample_shape), \
                "The ins's shape is {} are not equal to expected {}".format(ins.shape, self.sample_shape)
        return img, ins

    def __len__(self):
        return len(self.sampleInfo)
    
    def __getitem__(self, idx):
        img, ins = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins


class DatasetReal(Dataset):
    def __init__(self, data_root, mode="train", transform=None, sample_shape=[96, 96, 96], sample_num=9, dataset_name="Neural"):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.name = dataset_name
        self.datashape = [301, 301, 301]
        self.sample_shape = sample_shape
        self.sample_num = sample_num

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)
    
    def reset_Sample(self):
        self.sampleInfo = self.samplecropdata(self.datanames, self.sample_shape, self.sample_num)

    def getDataName(self, data_root):
        names = os.listdir(os.path.join(data_root, "tiffs"))
        names.sort()
        names = [name.split('.')[0] for name in names]
        num = len(names)
        if self.mode == "train":
            return names[:int(num*0.8)]
        elif self.mode == "eval":
            return names[int(num*0.8):]

    def samplecropdata(self, datanames, sam_shape=[96, 96, 96], num=3):
        shape = np.array(self.datashape)
        sam_shape = np.array(sam_shape)
        # margin = (shape - sam_shape) // 2
        # interval = [margin - 1, shape - margin]
        gap = shape - sam_shape + 1
        infos = []
        for name in datanames:
            # x = np.random.randint(low=interval[0, 0], high=interval[1, 0], size=num)
            # y = np.random.randint(low=interval[0, 1], high=interval[1, 1], size=num)
            # z = np.random.randint(low=interval[0, 2], high=interval[1, 2], size=num)
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

        img_path = os.path.join(self.data_root, "tiffs", name+'.tif')
        ins_path = os.path.join(self.data_root, "ins_modified", name+".tif")
        # ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 4.

        whd = self.sample_shape
        img = img[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        ins = ins[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]] 
        if img.shape != tuple(self.sample_shape):
            pdb.set_trace()
        # assert img.shape == tuple(self.sample_shape), pdb.set_trace()
        #         # "The img's shape is {} are not equal to expected {}".format(img.shape, self.sample_shape)
        # assert ins.shape == tuple(self.sample_shape), \
        #         "The ins's shape is {} are not equal to expected {}".format(ins.shape, self.sample_shape)
        return img, ins, info

    def __len__(self):
        return len(self.sampleInfo)
    
    def __getitem__(self, idx):
        img, ins, info = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins, info


class DatasetTest(Dataset):
    def __init__(self, data_root, transform=None, dataset_name="Neural"):
        self.data_root = data_root
        self.transform = transform
        self.name = dataset_name

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)

    def getDataName(self, data_root):
        names = os.listdir(data_root)
        names.sort()
        num = len(names)
        return names[int(num*0.8):]

    def readData(self, idx):
        name = self.datanames[idx]
        
        img_path = os.path.join(self.data_root, name, "noises", name+'.tif')
        # ins_path = os.path.join(self.data_root, name, "gts", name+".tif")
        ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        ins = tifffile.imread(ins_path)

        return img, ins

    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):
        img, ins = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins

class DatasetRealTest(Dataset):
    def __init__(self, data_root, transform=None, dataset_name="Neural"):
        self.data_root = data_root
        self.transform = transform
        self.name = dataset_name

        self.datanames = self.getDataName(self.data_root)
        # self.sampleInfo = self.samplecropdata(self.datanames, sample_shape, sample_num)

    def getDataName(self, data_root):
        names = os.listdir(os.path.join(data_root, "tiffs"))
        names.sort()
        names = [name.split('.')[0] for name in names]
        num = len(names)
        return names[int(num*0.8):]

    def readData(self, idx):
        name = self.datanames[idx]
        
        img_path = os.path.join(self.data_root, "tiffs", name+'.tif')
        ins_path = os.path.join(self.data_root, "ins_modified", name+".tif")
        # ins_path = os.path.join(self.data_root, name, "gts", name+".tif")
        # ins_path = os.path.join(INS_PATH, name+".tif")

        img = tifffile.imread(img_path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 4.
        ins = tifffile.imread(ins_path)

        return img, ins

    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):
        img, ins = self.readData(idx)

        if self.transform is not None:
            img, ins = self.transform(img, ins)
        
        return img[None].float(), ins


#### **************** ####
class Compose():
    """ Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *args):
        for i, t in enumerate(self.transforms):
            img, gts = args
            args = t(*args)
        return args

class ToTensor(object):

    def __call__(self, img, gts=None):
        img = torch.from_numpy(img.astype(float))
        if gts is None:
            return img,
        else:
            gts = torch.from_numpy(gts.astype(int))
            gts = gts.long()
            return img, gts


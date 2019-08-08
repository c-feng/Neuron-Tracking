import numpy as np
import os.path as osp
import torch

from glob import glob
from skimage import io 
from skimage.external import tifffile
from skimage.morphology import  skeletonize_3d,dilation,ball

from .serialization import read_json

def imgs_read(fnames):
    #print(fnames)
    #assert isinstance(fnames,list)
    imgs = []
    for fname in fnames:
        img = tifffile.imread(fname)
        img = img.astype(float)
        imgs.append(img[None])
    imgs = np.concatenate(imgs,axis = 0)
    return np.squeeze(imgs) 

def data_prepare(data_dir):
    assert osp.exists(data_dir)
    imgs_dir = osp.join(data_dir, "tiffs")
    segs_dir = osp.join(data_dir, "segs")
    assert osp.exists(imgs_dir)
    assert osp.exists(segs_dir)

    ends_dir = osp.join(data_dir, "ends")

    insts_dir = osp.join(data_dir, "ins")
    juncs_dir = osp.join(data_dir, "juncs")
    centerlines_dir = osp.join(data_dir, "centerlines")

    imgs_p = glob("{}/*".format(imgs_dir))
    imgs_p.sort()

    segs_p = []
    juncs_p = []

    ends_p = []
    
    insts_p = []
    juncs_p = []
    centerlines_p = []
    
    for img_p in imgs_p: 
        fname = osp.basename(img_p)
        seg_p = osp.join(segs_dir, fname)
        assert osp.exists(seg_p)
        segs_p.append(seg_p)

        if osp.exists(ends_dir):
            end_p = osp.join(ends_dir, fname)
            assert osp.exists(end_p)
            ends_p.append(end_p)
            if osp.exists(insts_dir) and osp.exists(juncs_dir) and osp.exists(centerlines_dir):
                inst_p = osp.join(insts_dir, fname)
                junc_p = osp.join(juncs_dir, fname)
                centerline_p = osp.join(centerlines_dir, fname) 
                assert osp.exists(inst_p) and osp.exists(junc_p) and osp.exists(centerline_p)
                insts_p.append(inst_p)
                juncs_p.append(junc_p)
                centerlines_p.append(centerline_p)
    
    if len(ends_p) > 0:
        if len(insts_p) > 0 and len(juncs_p) > 0 and len(centerlines_p) > 0:
            imgs_info = list(zip(imgs_p, segs_p, ends_p, insts_p, juncs_p, centerlines_p))
        else:
            imgs_info = list(zip(imgs_p, segs_p, ends_p))
    else:
        imgs_info = list(zip(imgs_p, segs_p))
    return imgs_info

class Dataset(object):
    def __init__(self,dataset,transform = None,gts_num = 1):
        super(Dataset,self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.gts_num = gts_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,ind):
        fnames = self.dataset[ind]
        imgs = imgs_read(fnames)  
        img = imgs[0]
        gts = imgs[1:self.gts_num + 1]
        if self.transform is not None:
            img, gts = self.transform(img, gts)
        return img[None].float(), gts

class LmdbDataset(object):
    def __init__(self,dataset,lmdb_dataset,transform = None,gts_num = 1):
        super(LmdbDataset,self).__init__()
        self.dataset = dataset
        
        self.lmdb_dataset = lmdb_dataset
        self.imgs_data = lmdb_dataset.open_db("imgs")
        self.imgs_size_data = lmdb_dataset.open_db("imgs_size")
        self.txn = env.begin() 

        self.transform = transform
        self.gts_num = gts_num

    def __len__(self):
        return len(self.dataset)
    
    def lmdb_to_img(self,fnames):
        fname = osp.splitext(osp.basename(fnames))[0] 
        imgs_bytes = self.txn.get(fname.encode(), db = self.imgs_data)
        imgs_size_bytes = self.txn.get(fname.encode(), db = self.imgs_size_data) 

        imgs_size = np.frombuffer(imgs_size_bytes)
        imgs_flatten = np.frombuffer(imgs_bytes)
        imgs = imgs_flatten.reshape(imgs_size)
        return imgs

    def __getitem__(self,ind):
        fnames = self.dataset[ind]
        #imgs = imgs_read(fnames)  
        imgs = self.lmdb_to_img(fnames)
        img = imgs[0]
        gts = imgs[1:self.gts_num + 1]
        if self.transform is not None:
            img, gts = self.transform(img, gts)
        return img[None].float(), gts

class FFNDataset(object):
    def __init__(self,dataset,transform = None):
        super(FFNDataset,self).__init__()
        self.dataset = dataset
        self.transform = transform
        #self.gts_num = gts_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,ind):
        fnames = self.dataset[ind]
        img_fname = fnames[0]
        #seg_fname = fnames[1]
        ins_fname = fnames[3]
        junc_fname = fnames[4]

        imgs = imgs_read([img_fname, ins_fname, junc_fname])
        

        img = imgs[0]
        gts = imgs[1:]
        gts = gts.astype(int)
        if self.transform is not None:
            img, gts = self.transform(img, gts)
        return img[None].float(), gts

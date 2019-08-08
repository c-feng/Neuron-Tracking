import numpy as np
import os.path as osp 
import sys
import argparse

from glob import glob
from multiprocessing import Pool
from skimage.external import tifffile
from sklearn.cluster import MeanShift

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.utils.data import DataLoader

from neuralTrack.utils.osutils import mkdir_if_missing
from neuralTrack.utils.patches_utils import patches_split,patches_merge

parser = argparse.ArgumentParser(description="Split tiffs into patches")

working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                                 default=osp.join(working_dir, 'logs'))

args = parser.parse_args()

mkdir_if_missing(args.logs_dir)

def single_run(i, img, seg, end, ins, junc, centerline, thres = 2):
    
    img = img.numpy()
    seg = seg.numpy()

    end = end.numpy()

    ins = ins.numpy()
    junc = junc.numpy()
    centerline = centerline.numpy()

    edge_mask = np.zeros_like(end, dtype = np.bool)
    edge_mask[:thres] = 1
    edge_mask[-thres:] = 1

    edge_mask[:,:thres] = 1
    edge_mask[:,-thres:] = 1
    
    edge_mask[:,:,:thres] = 1
    edge_mask[:,:,-thres:] = 1

    end[np.logical_and(edge_mask, seg > 0)] = 1

    if np.sum(seg) < 1000:
        return 

    fname_tiff_ = osp.join(args.logs_dir,\
                "tiffs/{}_{}.tif".format(fname,i))
    fname_seg_ = osp.join(args.logs_dir,\
                "segs/{}_{}.tif".format(fname,i))

    fname_end_ = osp.join(args.logs_dir,\
                "ends/{}_{}.tif".format(fname,i))

    fname_ins_ = osp.join(args.logs_dir,\
                "ins/{}_{}.tif".format(fname,i))
    fname_junc_ = osp.join(args.logs_dir,\
                "juncs/{}_{}.tif".format(fname,i))
    fname_centerline_ = osp.join(args.logs_dir,\
                "centerlines/{}_{}.tif".format(fname,i))

    tifffile.imsave(fname_tiff_, img.astype(np.uint16))
    tifffile.imsave(fname_seg_, seg.astype(np.uint8))

    tifffile.imsave(fname_end_, end.astype(np.uint8))

    tifffile.imsave(fname_ins_, ins.astype(np.uint16))
    tifffile.imsave(fname_junc_, junc.astype(np.uint8))
    tifffile.imsave(fname_centerline_, centerline.astype(np.uint16))

def imgs_read(imgs_p):
    imgs = []
    for img_p in imgs_p:
        img_array = tifffile.imread(img_p)
        img = torch.from_numpy(img_array.astype(int))
        imgs.append(img)
    return imgs


#tiffs_info = glob("/home/jjx/Biology/data_modified/tiffs/*")
tiffs_info = glob("/home/jjx/Biology/data_synthesize/tiffs/*")
tiffs_info.sort()

#segs_info = glob("/home/jjx/Biology/data_modified_junc/segs/*")
segs_info = glob("/home/jjx/Biology/data_synthesize/gts/*")
segs_info.sort()

#ends_info = glob("/home/jjx/Biology/data_modified/ends/*")
ends_info = glob("/home/jjx/Biology/data_synthesize/ends/*")
ends_info.sort()

#ins_info = glob("/home/jjx/Biology/data_modified_junc/ins_filter/*")
ins_info = glob("/home/jjx/Biology/data_synthesize/ins/*")
ins_info.sort()

#juncs_info = glob("/home/jjx/Biology/data_modified_junc/juncs/*")
juncs_info = glob("/home/jjx/Biology/data_synthesize/juncs/*")
juncs_info.sort()

#centerlines_info = glob("/home/jjx/Biology/data_modified/ins_modified/*")
centerlines_info = glob("/home/jjx/Biology/data_synthesize/centerlines/*")
centerlines_info.sort()

imgs_info = list(zip(tiffs_info, segs_info, ends_info, ins_info, juncs_info, centerlines_info))

#colors = np.array([0,500],dtype = np.uint16)
#colors = np.array([0,1],dtype = np.int8)

#patch_size = [168, 168, 168]
#patch_stride = [151,151,151]
patch_size = [64,64,64]
patch_stride = [64,64,64]
def consistency_check(fnames_p):
    fnames = [ osp.basename(x) for x in fnames_p ]
    return fnames.count(fnames[0]) == len(fnames)


for i,fnames in enumerate(imgs_info[-2:]):
    fname_dir = osp.join(args.logs_dir,osp.splitext(osp.basename(fnames[0]))[0])
    fname = osp.splitext(osp.basename(fnames[0]))[0]
    print(fname)
    assert consistency_check(fnames)
    img, seg, end, ins, junc, centerline = imgs_read(fnames) 
    
    img_patches,_ = patches_split(img, patch_size, patch_stride)
    seg_patches,_ = patches_split(seg, patch_size, patch_stride)
    
    end_patches,_ = patches_split(end, patch_size, patch_stride)

    ins_patches,_ = patches_split(ins, patch_size, patch_stride)
    junc_patches,_ = patches_split(junc, patch_size, patch_stride)
    centerline_patches,_ = patches_split(centerline, patch_size, patch_stride)


    mkdir_if_missing(osp.join(args.logs_dir, "tiffs"))
    mkdir_if_missing(osp.join(args.logs_dir, "segs"))

    mkdir_if_missing(osp.join(args.logs_dir, "ends"))
    
    mkdir_if_missing(osp.join(args.logs_dir, "ins"))
    mkdir_if_missing(osp.join(args.logs_dir, "juncs"))
    mkdir_if_missing(osp.join(args.logs_dir, "centerlines"))

    i = 0
    with Pool(1) as pool:
        _ = pool.starmap(single_run,\
                zip(range(len(img_patches)), img_patches, seg_patches,\
        end_patches, \
        ins_patches, junc_patches, centerline_patches))

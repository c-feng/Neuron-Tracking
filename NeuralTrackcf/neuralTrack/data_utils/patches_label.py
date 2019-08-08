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
from neuralTrack.utils.dataset import imgs_read

parser = argparse.ArgumentParser(description="Split tiffs into patches")

working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                                 default=osp.join(working_dir, 'logs'))

args = parser.parse_args()

mkdir_if_missing(args.logs_dir)

def single_run(i, img, seg, end, ins, junc, centerline, thres = 2):
    
    img = img.numpy()
    #seg = seg.numpy()

    end = end.numpy()

    ins = ins.numpy()
    junc = junc.numpy()
    centerline = centerline.numpy()

    seg = (ins > 0).astype(int)

    #edge_mask = np.zeros_like(end, dtype = np.bool)
    #edge_mask[:thres] = 1
    #edge_mask[-thres:] = 1

    #edge_mask[:,:thres] = 1
    #edge_mask[:,-thres:] = 1
    
    #edge_mask[:,:,:thres] = 1
    #edge_mask[:,:,-thres:] = 1

    #end[np.logical_and(edge_mask, seg > 0)] = 1

    if np.sum(seg) == 0:
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

    tifffile.imsave(fname_tiff_, img.astype(np.float32))
    tifffile.imsave(fname_seg_, seg.astype(np.uint8))

    tifffile.imsave(fname_end_, end.astype(np.uint8))

    tifffile.imsave(fname_ins_, ins.astype(np.uint16))
    tifffile.imsave(fname_junc_, junc.astype(np.uint8))
    tifffile.imsave(fname_centerline_, centerline.astype(np.uint16))


#tiffs_info = glob("/home/jjx/Biology/data_modified/tiffs/*")
tiffs_info = glob("/home/jjx/Biology/data_synthesize_100/noises/*")
#tiffs_info = glob("/home/jjx/Biology/data_synthesize/noises/*")
tiffs_info.sort()

#segs_info = glob("/home/jjx/Biology/data_modified_junc/segs/*")
segs_info = glob("/home/jjx/Biology/data_synthesize_junc_100/segs/*")
segs_info.sort()

#ends_info = glob("/home/jjx/Biology/data_modified/ends/*")
ends_info = glob("/home/jjx/Biology/data_synthesize_100/ends/*")
ends_info.sort()

#ins_info = glob("/home/jjx/Biology/data_modified_junc/ins_filter/*")
ins_info = glob("/home/jjx/Biology/data_synthesize_junc_100/ins_filter/*")
ins_info.sort()

#juncs_info = glob("/home/jjx/Biology/data_modified_junc/juncs/*")
juncs_info = glob("/home/jjx/Biology/data_synthesize_junc_100/juncs/*")
juncs_info.sort()

#centerlines_info = glob("/home/jjx/Biology/data_modified/ins_modified/*")
centerlines_info = glob("/home/jjx/Biology/data_synthesize_junc_100/ins_filter/*")
centerlines_info.sort()

imgs_info = list(zip(tiffs_info, segs_info, ends_info, ins_info, juncs_info, centerlines_info))
#print(imgs_info)

#colors = np.array([0,500],dtype = np.uint16)
#colors = np.array([0,1],dtype = np.int8)

patch_size = [96, 96, 96]
patch_stride = [90, 90, 90]
#patch_size = [120,120,120]
#patch_stride = [101,101,101]
def consistency_check(fnames_p):
    fnames = [ osp.basename(x) for x in fnames_p ]
    return fnames.count(fnames[0]) == len(fnames)


for i,fnames in enumerate(imgs_info[-20:]):
    fname_dir = osp.join(args.logs_dir,osp.splitext(osp.basename(fnames[0]))[0])
    fname = osp.splitext(osp.basename(fnames[0]))[0]
    assert consistency_check(fnames)
    imgs_array = imgs_read(fnames)
    imgs = torch.from_numpy(imgs_array)

    img = imgs[0]
    gts = imgs[1:].long()
    seg, end, ins, junc, centerline = gts 
    print(fname, torch.sum(img))

    
    img_patches, _, _ = patches_split(img, patch_size, patch_stride)
    seg_patches, _, _ = patches_split(seg, patch_size, patch_stride)
    
    end_patches, _, _ = patches_split(end, patch_size, patch_stride)

    ins_patches, _, _ = patches_split(ins, patch_size, patch_stride)
    junc_patches, _, _ = patches_split(junc, patch_size, patch_stride)
    centerline_patches,_, _ = patches_split(centerline, patch_size, patch_stride)


    mkdir_if_missing(osp.join(args.logs_dir, "tiffs"))
    mkdir_if_missing(osp.join(args.logs_dir, "segs"))

    mkdir_if_missing(osp.join(args.logs_dir, "ends"))
    
    mkdir_if_missing(osp.join(args.logs_dir, "ins"))
    mkdir_if_missing(osp.join(args.logs_dir, "juncs"))
    mkdir_if_missing(osp.join(args.logs_dir, "centerlines"))

    i = 0
    with Pool(8) as pool:
        _ = pool.starmap(single_run,\
                zip(range(len(img_patches)), img_patches, seg_patches,\
        end_patches, \
        ins_patches, junc_patches, centerline_patches))

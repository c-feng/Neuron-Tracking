import numpy as np
import os.path as osp 
import sys
import argparse
from glob import glob

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.external import tifffile
from scipy.ndimage.interpolation import zoom
from skimage.util import pad
from skimage.morphology import skeletonize_3d,dilation,cube,ball

from neuralTrack.utils.serialization import read_json
from neuralTrack.utils import transform  
from neuralTrack.utils.osutils import mkdir_if_missing
from neuralTrack.utils.patches_utils import patches_split,patches_merge


parser = argparse.ArgumentParser(description="Softmax loss classification")

working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                                 default=osp.join(working_dir, 'logs'))

args = parser.parse_args()


t = []

imgs_info_centers = glob("/home/jjx/Biology/DirectField/data_300_modified_junc2/centers/*")
imgs_info_centers.sort()

imgs_info_juncs = glob("/home/jjx/Biology/DirectField/data_300_modified_junc2/juncs/*")
imgs_info_juncs.sort()

imgs_info_ins = glob("/home/jjx/Biology/DirectField/data_300_modified_junc2/ins/*")
imgs_info_ins.sort()

#imgs_info_tiffs = glob("/home/jjx/Biology/DirectField/data_300_modified_VOX/visual_results/*/*_pred.tif")
imgs_info_gts = glob("/home/jjx/Biology/DirectField/data_300_modified_junc2/gts/*")
imgs_info_gts.sort()

imgs_info_tiffs = glob("/home/jjx/Biology/DirectField/data_300_modified/tiffs/*")
imgs_info_tiffs.sort()

imgs_info = list(zip(imgs_info_gts,imgs_info_ins,imgs_info_juncs,imgs_info_centers,imgs_info_tiffs))[:2]

mkdir_if_missing(args.logs_dir)


def range_mask_generate(coord,shape,thres = 8):
    x,y,z = coord
    x_r = x + thres if x + thres <= shape[0] else shape[0]
    x_l = x - thres if x - thres >=0 else 0

    y_r = y + thres if y + thres <= shape[1] else shape[1]
    y_l = y - thres if y - thres >=0 else 0

    z_r = z + thres if z + thres <= shape[2] else shape[2]
    z_l = z - thres if z - thres >=0 else 0
    
    mask = np.zeros(shape,dtype = bool)
    mask[x_l:x_r,y_l:y_r,z_l:z_r] = 1
    return mask

def center_crop(img,img_gt,ins_gt,tiff,center_coord,patch_size,mode = "constant",flag = "True"):


    x,y,z = center_coord 
    x_l = x - patch_size[0]//2 
    x_r = x + patch_size[0] - patch_size[0]//2

    y_l = y - patch_size[1]//2
    y_r = y + patch_size[1] - patch_size[1]//2

    z_l = z - patch_size[2]//2 
    z_r = z + patch_size[2] - patch_size[2]//2
        
    pad_x_l = 0 if x_l >= 0 else -x_l
    pad_x_r = 0 if x_r <= img_gt.shape[0] else x_r - img_gt.shape[0]
    pad_x = [pad_x_l,pad_x_r]

    pad_y_l = 0 if y_l >= 0 else -y_l
    pad_y_r = 0 if y_r <= img_gt.shape[1] else y_r - img_gt.shape[1]
    pad_y = [pad_y_l,pad_y_r]

    pad_z_l = 0 if z_l >= 0 else -z_l
    pad_z_r = 0 if z_r <= img_gt.shape[2] else z_r - img_gt.shape[2]
    pad_z = [pad_z_l,pad_z_r]

    if x_l <0:
        x_r -= x_l 
        x_l -= x_l 

    if y_l <0:
        y_r -= y_l 
        y_l -= y_l

    if z_l <0:
        z_r -= z_l 
        z_l -= z_l

        
    img_pad = pad(img,[pad_x,pad_y,pad_z],mode = mode)
    img_c = img_pad[x_l:x_r,y_l:y_r,z_l:z_r]

    img_gt_pad = pad(img_gt,[pad_x,pad_y,pad_z],mode = mode)
    img_gt_c = img_gt_pad[x_l:x_r,y_l:y_r,z_l:z_r]
    
    ins_gt_pad = pad(ins_gt,[pad_x,pad_y,pad_z],mode = mode)
    ins_c = ins_gt_pad[x_l:x_r,y_l:y_r,z_l:z_r]

    tiff_gt_pad = pad(tiff,[pad_x,pad_y,pad_z],mode = mode)
    tiff_c = tiff_gt_pad[x_l:x_r,y_l:y_r,z_l:z_r]

    if flag:
        labels = np.unique(ins_c)[1:]
        for label_ in labels:
            mask = ins_c == label_
            if np.sum(mask) < 100:
                mask = np.logical_and(mask,img_gt_c)
                inds_sel = np.arange(img_gt_c.size)[mask.flatten()]
                if not inds_sel.any():continue
                print(inds_sel)
                xs,ys,zs = np.unravel_index(inds_sel,mask.shape)
                
                x = int(xs.mean())
                y = int(ys.mean())
                z = int(zs.mean())
                mask_r = range_mask_generate([x,y,z],mask.shape,5)
                img_gt_c[mask_r] = 0

    return img_c,img_gt_c,ins_c,tiff_c

colors = np.array([0,500],dtype = np.uint16)
patch_size = [120,120,120]
patch_stride = [60,60,60]

for i,(img_p,ins_p,gt_p,center_p,tiff_p) in enumerate(imgs_info):
    fnames = imgs_info[i]
    print(fnames)
    img = tifffile.imread(img_p)
    img_gt = tifffile.imread(gt_p)
    ins = tifffile.imread(ins_p)
    center = tifffile.imread(center_p)
    tiff = tifffile.imread(tiff_p)

    print(np.sum(center))

    img = (img > 0).astype(int)
    img_gt = (img_gt > 0).astype(int)
    
    img_gt = np.logical_and(img_gt,img).astype(int)

    fname_dir = osp.join(args.logs_dir,osp.splitext(osp.basename(fnames[0]))[0])
    fname = osp.splitext(osp.basename(fnames[0]))[0]

    mkdir_if_missing(osp.join(args.logs_dir,fname)) 
    mkdir_if_missing(osp.join(args.logs_dir,"{}/juncs".format(fname)))
    mkdir_if_missing(osp.join(args.logs_dir,"{}/gts".format(fname)))
    mkdir_if_missing(osp.join(args.logs_dir,"{}/tiffs".format(fname)))
    mkdir_if_missing(osp.join(args.logs_dir,"{}/ins".format(fname)))
    
    
    mask = center > 0
    inds_sel = np.arange(center.size)[mask.flatten()] 
    xs,ys,zs = np.unravel_index(inds_sel,center.shape)
    i = 0 
    for x,y,z in zip(xs,ys,zs):
        print(x,y,z)
        i += 1
        img_,img_gt_,ins_,tiff_ = center_crop(img,img_gt,ins,tiff,[x,y,z],patch_size)
        
        if np.sum(img_) ==0:continue

        fname_gt_ = osp.join(args.logs_dir,\
                "{}/gts/{}.tif".format(fname,i))
        
        fname_junc_ = osp.join(args.logs_dir,\
                "{}/juncs/{}.tif".format(fname,i))

        fname_tiff_ = osp.join(args.logs_dir,\
                "{}/tiffs/{}.tif".format(fname,i))
        
        fname_ins_ = osp.join(args.logs_dir,\
                "{}/ins/{}.tif".format(fname,i))
         

        tifffile.imsave(fname_gt_,colors[img_])
        tifffile.imsave(fname_junc_,colors[img_gt_])
        tifffile.imsave(fname_tiff_,tiff_)
        tifffile.imsave(fname_ins_,ins_)


    fname_tiff = osp.join(args.logs_dir,"{}/{}.tif".format(fname,fname))
    fname_label = osp.join(args.logs_dir,"{}/{}_label.tif".format(fname,fname))
    
    tifffile.imsave(fname_tiff,colors[img])
    tifffile.imsave(fname_label,colors[img_gt])
    #sys.exit(0)



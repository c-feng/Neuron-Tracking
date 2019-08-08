import json
import os
import sys

import os.path as osp
import numpy as np
import scipy

from skimage import io 
from skimage.external import tifffile
from skimage.transform import resize
from skimage.morphology import dilation,ball
from glob import glob 
from multiprocessing import Pool
from functools import partial
from treelib import Node, Tree


from ..utils.serialization import  read_json,write_json
from ..utils.osutils import mkdir_if_missing
from ..utils.direct_field import distance_transform,diret_field_cal
from ..utils.coords_utils import endpoint_cal, connect_2_points, coords_filted_get
from .ins_modified import name_to_int, analyze_single_swc

def fill_2_nodes(node_0, node_1, shape):
    p0 = node_0.data
    p1 = node_1.data
    mask = connect_2_points(p0, p1, shape)
    return mask > 0

def fill_nodes(a_tree, a_uid, mask):
    queue = [a_tree[a_uid]]
    while(len(queue) > 0):
        node_0 = queue.pop()
        uid_0 = node_0.identifier
        p0 = node_0.data

        for node_1 in a_tree.children(uid_0):
            uid_1 = node_1.identifier
            p1 = node_1.data

            mask_ = fill_2_nodes(node_0, node_1, mask.shape)
            mask[mask_] = 1
            queue.append(node_1)
    return mask 


def gt_fill_single(ind, tiff_path, swcs_path, shape_path = None,\
        target_dir = None, unit = None, sep = None, flag_visual = False):

    assert target_dir is not None , "No targetdir is specified"

    fname = osp.basename(tiff_path)

    tiffs_dir = osp.join(target_dir,"tiffs")
    gts_dir = osp.join(target_dir,"gts")
    ins_dir = osp.join(target_dir,"ins")

    tiff = tifffile.imread(tiff_path)
    print("{} {}".format(fname, tiff.shape))
    target_size = tiff.shape

    img_mean = np.mean(tiff.flatten())
    img_std = np.std(tiff.flatten())
    
    gt = np.zeros_like(tiff, dtype = np.uint8)
    gt_ins = np.zeros_like(tiff, dtype = np.uint16)
    
    #swcs_path.sort()
    swcs_p_length = [[x, len(open(x).readlines()) ] for x in swcs_path]
    swcs_p_length.sort(key = lambda x: x[-1], reverse = True)
    i = 0
    for swc_path, length in swcs_p_length:
        fname_swc = osp.basename(swc_path)
        ins_label = name_to_int(fname_swc) + 1 
        print("{} is processing".format(swc_path))
        end_coords, branch_coords, coords, ftree = endpoint_cal(swc_path, unit, sep = sep)

        #coords += 3
        
        
        root_uid = ftree.root
        mask = np.zeros_like(tiff)
        mask = fill_nodes(ftree, root_uid, mask)
        mask = mask > 0
        print("{} {} {}".format(swc_path, ins_label, np.sum(mask)))

        gt[mask] = 1
        gt_ins[mask] = ins_label

    if shape_path is not None: 
        coords, labels, ids, pars = coords_filted_get(shape_path, unit, sep, target_size)
        gt = np.zeros_like(tiff)
        gt[coords[:,0],coords[:,1],coords[:,2]] = 1
        gt = dilation(gt,ball(1))
    else:
        #gt = dilation(gt,ball(2))
        pass

    tiff_path = osp.join(tiffs_dir,fname)
    tifffile.imsave(tiff_path,tiff)

    gt_path = osp.join(gts_dir,fname)
    tifffile.imsave(gt_path,gt.astype(np.uint8))

    ins_path = osp.join(ins_dir,fname)
    tifffile.imsave(ins_path,gt_ins)

    if flag_visual:
        colors = np.array([0,500],dtype = np.uint16)
        gts_visual_dir = osp.join(target_dir,"gts_visual")
        gts_visual_xyz_dir = osp.join(target_dir,"gts_visual_xyz")
        
        gt_visual_path = osp.join(gts_visual_dir,fname)
        tifffile.imsave(gt_visual_path,colors[gt])

        fname_ = osp.splitext(fname)[0]
        ins_xyz = []
        for i,d in  enumerate(["x","y","z"]):
            ins_mask = (np.sum(gt_ins,axis = i)>0).astype(np.uint16)
            ins_xyz.append(colors[ins_mask])
        ins_xyz = np.hstack(ins_xyz)

        fpath = osp.join(gts_visual_xyz_dir,"{}_xyz.jpg".format(fname_))
        scipy.misc.imsave(fpath,ins_xyz)

    imgs_list = [tiff_path,ins_path,gt_path]
    seg_len = len(np.unique(gt_ins)) - 1
    return imgs_list, seg_len, img_mean, img_std

def gt_fill(tiff_paths, swcs_paths, shape_paths = None, target_dir = None,\
        unit = [0.2,0.2,1], sep = ",", flag_visual = False):

    tiffs_dir = osp.join(target_dir,"tiffs")
    mkdir_if_missing(tiffs_dir)

    gts_dir = osp.join(target_dir,"gts")
    mkdir_if_missing(gts_dir)

    ins_dir = osp.join(target_dir,"ins")
    mkdir_if_missing(ins_dir)
    
    if flag_visual:
        gts_visual_dir = osp.join(target_dir,"gts_visual")
        mkdir_if_missing(gts_visual_dir)
     
        gts_visual_xyz_dir = osp.join(target_dir,"gts_visual_xyz")
        mkdir_if_missing(gts_visual_xyz_dir)

    cpu_nums = os.cpu_count()//2 
    #cpu_nums = 1
    
    with Pool(cpu_nums) as pool:
        if shape_paths is None:
            infos = pool.starmap(partial(gt_fill_single,\
                target_dir = target_dir,\
                flag_visual = flag_visual,\
                shape_path = None,
                unit = unit,
                sep = sep)\
                ,zip(range(len(tiff_paths)), tiff_paths, swcs_paths))
        else:
            infos = pool.starmap(partial(gt_fill_single,\
                target_dir = target_dir,\
                flag_visual = flag_visual,
                unit = unit,
                sep = sep)\
                ,zip(range(len(tiff_paths)),tiff_paths,swcs_paths,shape_paths))

    return infos

if __name__ == "__main__":

    #target_dir = "/home/jjx/Biology/DirectField/data_test"
    target_dir = "/home/jjx/Biology/data_synthesize_100"

    tiff_paths = []
    swcs_paths = []
    shape_paths = []

    for data_dir in ["/home/jjx/Biology/Synthesize_Selected_Dataset2"]:

        tiff_path = [ x for x in glob(data_dir+"/tiffs/*.tif") \
                if not "gts" in osp.dirname(x)]
        swcs_path = [glob( osp.join(data_dir, "swcs/{}*.swc".format(\
                osp.splitext(osp.basename(x))[0]))) for x in tiff_path] 
        
        tiff_paths += tiff_path
        swcs_paths += swcs_path

    print(len(tiff_paths))
    
    gts_dir = osp.join(target_dir,"gts")
    tiffs_dir = osp.join(target_dir,"tiffs")
    ins_dir = osp.join(target_dir,"ins")
    gts_visual_dir = osp.join(target_dir,"gts_visual")

    
    imgs_info = []
    lengths_info = []
    imgs_mean = []
    imgs_std = []
    
    #unit = [0.2,0.2,1]
    unit = [1, 1, 1]

    infos = gt_fill(tiff_paths[:100],swcs_paths[:100],None,target_dir,unit = unit,sep = " ",\
        flag_visual = False)

    for x in infos:
        imgs_list,seg_len,img_mean,img_std = x
        
        imgs_info.append(imgs_list) 
        lengths_info.append(seg_len)
        imgs_mean.append(img_mean)
        imgs_std.append(img_std)

    infos = {"imgs":imgs_info,
            "num_ins":lengths_info,
            "mean":np.mean(imgs_mean).item(),
            "std":np.mean(imgs_std).item()}

    infos_path = osp.join(target_dir,"info.json")
    write_json(infos,infos_path)


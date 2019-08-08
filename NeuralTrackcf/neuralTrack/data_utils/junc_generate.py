from skimage.morphology import dilation,ball,cube,closing,remove_small_objects 
from skimage.external import tifffile
from sklearn.cluster import MeanShift
from multiprocessing import Pool
from glob import glob

from neuralTrack.utils.serialization import load_checkpoint,save_checkpoint,read_json,write_json
from neuralTrack.utils.osutils import mkdir_if_missing
from neuralTrack.utils.logging import Logger, TFLogger
from neuralTrack.utils.mask_utils import dilated_mask_generate
from neuralTrack.finchpy.finch import finch

import os.path as osp
import numpy as np
import scipy
import sys

def juncs_label(ins,seg,thres = 2):
    labels = np.unique(ins)[1:]
    inds = np.arange(ins.size)
    ins_filter = np.zeros_like(ins)
    junc_inds = set()
    for label in labels:
        
        mask_l = ins == label 
        #mask_l_d = dilation(mask_l, ball(1))
        mask_l_d = mask_l

        ins_filter[mask_l_d > 0] = label

        print(label,np.sum(mask_l_d > 0)) 
        
        mask_r = np.logical_xor(ins > 0, mask_l)
        mask_r_d = dilation(mask_r, ball(2))
        
        mask_inters = np.logical_and(mask_r_d,mask_l_d)

        junc_inds = junc_inds.union(set(inds[(mask_inters > 0).flatten()]))
        
    junc_inds = list(junc_inds)

    if len(junc_inds) == 0:
        junc = np.zeros_like(ins)
        return junc,ins_filter 

    coords = np.array(np.unravel_index(junc_inds,ins.shape)).transpose()
    
    print(len(coords))

    #clustering = MeanShift(3).fit(coords)
    clusterings, num_clusters = finch(coords, [], 1)
    print(clusterings.shape)
    cluster_labels = clusterings[:,-1]

    labels_ = np.unique(cluster_labels)

    junc = np.zeros_like(ins)
   
    for label_ in labels_:
        mask_ = np.zeros_like(ins)

        coords_sel = coords[cluster_labels == label_]
        
        coord_lt = np.min(coords_sel, axis = 0)
        coord_rb = np.max(coords_sel, axis = 0)
        
        h,w,d = coord_rb - coord_lt
        area = h * w * d
        if area == 0  : continue
        print(area)

        x,y,z = np.mean(coords_sel,axis = 0).astype(int)
        mask = dilated_mask_generate((x,y,z),ins.shape, ball, [thres])
        junc[mask > 0] = seg[mask > 0]
    return junc, ins_filter 


def single_run(imgs_p,flag_junc = False,flag_seg = False,flag_ins = True):    
#def single_run(imgs_p,flag_junc = True,flag_seg = True,flag_ins = True):    
    colors = np.array([0,500],np.uint16)

    tiff_p,ins_p,seg_p = imgs_p
    fname = osp.basename(ins_p)

    ins = tifffile.imread(ins_p)
    print(fname, len(np.unique(ins)))

    seg = tifffile.imread(ins_p)
    seg = (ins > 0).astype(int)

    junc_generate,ins_filter = juncs_label(ins,seg)

    ins_generate_p = osp.join(target_dir,"ins_filter/{}".format(fname))

    seg_generate_p = osp.join(target_dir, "segs/{}".format(fname))
    seg_generate_visual_p = osp.join(target_dir, "segs_visual/{}".format(fname))

    junc_generate_p = osp.join(target_dir, "juncs/{}".format(fname))
    junc_generate_visual_p = osp.join(target_dir, "juncs_visual/{}".format(fname))

    if flag_ins: 
        mkdir_if_missing(osp.join(target_dir,"ins_filter"))
        tifffile.imsave(ins_generate_p, ins_filter)
    
    mkdir_if_missing(osp.join(target_dir,"juncs"))
    tifffile.imsave(junc_generate_p, junc_generate.astype(np.uint8))
    if flag_junc:
        mkdir_if_missing(osp.join(target_dir,"juncs_visual"))
        tifffile.imsave(junc_generate_visual_p, colors[junc_generate])
    
    mkdir_if_missing(osp.join(target_dir,"segs"))
    tifffile.imsave(seg_generate_p, (ins_filter > 0).astype(np.uint8)) 
    if flag_seg:
        mkdir_if_missing(osp.join(target_dir,"segs_visual"))
        tifffile.imsave(seg_generate_visual_p, colors[seg])

    return [seg_generate_p, junc_generate_p]

if __name__ == "__main__":

    target_dir = "/home/jjx/Biology/data_synthesize_junc_100"
    mkdir_if_missing(target_dir)
    sys.stdout = Logger(osp.join(target_dir, "log.txt"))

    infos_tiffs = glob("/home/jjx/Biology/data_synthesize_100/tiffs/*")
    infos_ins = glob("/home/jjx/Biology/data_synthesize_100/ins/*")
    infos_segs = glob("/home/jjx/Biology/data_synthesize_100/gts/*")
    
    infos_tiffs.sort()
    infos_ins.sort()
    infos_segs.sort()

    imgs_p = list(zip(infos_tiffs,infos_ins,infos_segs))
    imgs_info = [] 

    
    with Pool(8) as pool:
        imgs_info = pool.starmap(single_run,zip(imgs_p))

    infos = {
        "imgs":imgs_info,
        "mean":0,
        "std":1
        }

    write_json(infos,osp.join(target_dir,"info.json"))

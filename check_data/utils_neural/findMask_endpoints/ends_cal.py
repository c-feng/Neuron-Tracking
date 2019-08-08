import numpy as np
import torch
import skimage
import kimimaro
import os.path as osp

from treelib import Node, Tree
from skimage.morphology import dilation, ball, closing
from skimage.external import tifffile
from glob import glob

def skel_cal(fov_seg_array, dust_thres = 2, nums_cpu = 1, anisotropy=(200,200,1000)):
    fov_seg_array = fov_seg_array.astype(int)
    skels = kimimaro.skeletonize(
        fov_seg_array, 
        teasar_params={
            'scale': 4,
            'const': 500, # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100, # physical units
            'soma_acceptance_threshold': 3500, # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300, # physical units
            'max_paths': None, # default None
                },
            dust_threshold=dust_thres,
            anisotropy=anisotropy, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=False, # default False
            parallel=nums_cpu, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )
    print(skels.keys())
    skel = skels[1]

    fov_ins_skel_array = np.zeros_like(fov_seg_array)
    coords = (skel.vertices / np.array(anisotropy)).astype(int)
    fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return fov_ins_skel_array, skel

def ends_cal(skel, anisotropy=(200,200,1000)):
    edges = skel.edges
    coords = (skel.vertices / np.array([200, 200, 1000])).astype(int)
    
    #coords = coords
    #epdges = edges
    #ends_ind = [ for x in range(len(coords))]
    ends = []
    vecs = []
    for edge  in edges:
        l, r = edge
        if np.sum(edges == l) == 1:
            ends.append(coords[l])
            vec = coords[l] - coords[r]
            vecs.append(vec)
        elif np.sum(edges == r) == 1:
            ends.append(coords[r])
            vec = coords[r] - coords[l]
            vecs.append(vec)
        else:
            pass
    ends = np.array(ends)
    vecs = np.array(vecs)
    return ends, vecs 

fov_ins_paths = glob("./test/*")
for fov_ins_path in fov_ins_paths:
    fov_ins_array = tifffile.imread(fov_ins_path)
    
    ends_array = np.zeros_like(fov_ins_array)
    skels_array = np.zeros_like(fov_ins_array)

    labels = np.unique(fov_ins_array[fov_ins_array > 0])
    i = 0
    for label in labels:
        i += 1
        if i > 5:break
        fov_seg_array = closing(fov_ins_array == label, ball(1))
        fov_label_array = skimage.measure.label(fov_seg_array)
        fov_labels = np.unique(fov_label_array[fov_label_array > 0])

        label_sel = fov_labels[np.argmax([np.sum(fov_label_array == x) for x in fov_labels])]

        fov_seg_array = fov_label_array == label_sel
        if np.sum(fov_seg_array) < 50:
            continue
        fov_skel_array, skel = skel_cal(fov_seg_array) 
        
        skels_array[fov_skel_array > 0] = 1

        ends, vecs = ends_cal(skel)
        ends_array[ends[:, 0], ends[:, 1], ends[:, 2]] = 1

    fname = osp.splitext(osp.basename(fov_ins_path))[0]
    fpath_ends = osp.join(osp.dirname(fov_ins_path), "{}_ends.tif".format(fname))
    fpath_skels = osp.join(osp.dirname(fov_ins_path), "{}_skels.tif".format(fname))

    tifffile.imsave(fpath_ends, ends_array)
    tifffile.imsave(fpath_skels, skels_array)



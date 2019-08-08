import numpy as np
import torch
from skimage.morphology import dilation, ball

from .metrics import soft_metric, calc_bd

def fov_seed(fov_seeds_array, fov_ins_array):
    seeds_be_viewed = dilation(fov_ins_array > 0, ball(3))
    seeds_to_be_viewed = np.logical_and(fov_seeds_array > 0, seeds_be_viewed == 0) 

    seed_mask = np.zeros_like(fov_seeds_array)
    print(np.sum(seeds_to_be_viewed))
    if np.sum(seeds_to_be_viewed) < 200:
        return seed_mask

    seed_ind = np.random.choice(\
            np.arange(seeds_to_be_viewed.size)\
            [seeds_to_be_viewed.flatten()], 1)[0]

    seed_coord = np.unravel_index(seed_ind, fov_seeds_array.shape)
    seed_mask[seed_coord[0], seed_coord[1], seed_coord[2]] = 1
    seed_mask = np.logical_and(seeds_to_be_viewed, dilation(seed_mask, ball(3))).astype(int)

    return seed_mask

def batch_fov_update(fov_seg_arrays, fov_ins_arrays, thres = 0.1, labels = None):
    fov_ins_list = []
    metrics = []
    matchs = []
    if labels == None:
        labels = [None]*len(fov_seg_arrays)
    for fov_seg_array, fov_ins_array, label in zip(fov_seg_arrays, fov_ins_arrays, labels):
        fov_ins, metric, match = fov_update(fov_seg_array, fov_ins_array, mode, thres)
        fov_ins_list.append(fov_ins)
        metrics.append(metric)
        matchs.append(match)
    return np.array(fov_ins_list), np.array(metrics), np.array(matchs) 

def fov_update(fov_seg_array, fov_ins_array, thres = 0.1, label = None):
    fov_seg_mask = fov_seg_array > 0

    if label == None:
        label = np.max(fov_ins_array) + 1

    metrics_, matchs_ = calc_bd(fov_seg_array, fov_ins_array, 0)
    metric = metrics_[0]
    match = matchs_[0]


    if metric[0] > thres:
        label = match[-1] 
        fov_ins_array[fov_seg_mask] = int(label)
        return fov_ins_array, metric, match[-1] 
    else:
        fov_ins_array[fov_seg_mask] = int(label)
        return fov_ins_array, metric, None


def fov_spread(fov_ins, intersect = [1,1,1]):
    prev_infos = {}

    if np.sum(fov_ins) == 0:
        return prev_infos 

    yz_l = fov_ins[:intersect[0]]
    yz_r = fov_ins[-intersect[0]:]

    xz_l = fov_ins[:,:intersect[1]]
    xz_r = fov_ins[:,-intersect[1]:]

    xy_l = fov_ins[:,:,:intersect[2]]
    xy_r = fov_ins[:,:,-intersect[2]:]
    
    labels = np.unique(fov_ins[fov > 0]).tolist()    
    
    for label in labels:
        prev_infos[label] = []
        
        if np.sum(yz_l == label) > 0:
            prev_infos[label].append([-1,0,0]) 

        if np.sum(xz_l == label) > 0:
            prev_infos[label].append([0,-1,0])

        if np.sum(xy_l == label) > 0:
            prev_infos[label].append([0,0,-1])

        if np.sum(yz_r == label) > 0:
            prev_infos[label].append([1,0,0])
        
        if np.sum(xz_r == label) > 0:
            prev_infos[label].append([0,1,0])

        if np.sum(xy_r == label) > 0:
            prev_infos[label].append([0,0,1])

    return prev_infos            
    
def fov_intersect(idx, fov, prev_preds_list, grid_size, intersect ):
    x,y,z = np.unravel_index(idx, grid_size)
    
    directs = []
    prev_infos = []

    if x > 0:
        fov_idx = np.ravel_multi_index([x - 1, y, z], grid_size)
        side_fov = prev_preds_list[fov_idx]
        fov[:intersect[0]] = side_fov[-intersect[0]:]
        directs.append([-1, 0, 0])
        #ins_info = np.unique(side_fov[-intersect[0]:]).tolist().remove(0)
        prev_infos.append(side_fov)

    if y > 0:
        fov_idx = np.ravel_multi_index([x, y - 1, z], grid_size)
        side_fov = prev_preds_list[fov_idx]
        fov[:,:intersect[1]] = side_fov[:,-intersect[1]:]
        directs.append([0, -1, 0])
        #ins_info = np.unique(side_fov[:,-intersect[1]:]).tolist().remove(0)
        prev_infos.append(side_fov)

    if z > 0:
        fov_idx = np.ravel_multi_index([x, y, z - 1], grid_size)
        side_fov = prev_preds_list[fov_idx]
        fov[:,:,:intersect[2]] = side_fov[:,:,-intersect[2]:]
        directs.append([0, 0, -1])
        #ins_info = np.unique(side_fov[:,:,-intersect[2]:]).tolist().remove(0)
        prev_infos.append(side_fov)
        
    return fov, directs, prev_infos 
    
def fov_select(idx, fov_list, direct, grid_size):
    coord_ = np.array(np.unravel_index(idx, grid_size))
    coord_ += np.array(direct)
    idx_ = np.ravel_multi_index(coord_, grid_size)
    return fov_list[idx_]
    

    


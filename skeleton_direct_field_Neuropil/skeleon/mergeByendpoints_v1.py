import os
import numpy as np
import copy
import math
from scipy.spatial.distance import squareform, pdist
import skimage
from skimage.morphology import dilation, ball, closing
import kimimaro
import pdb

def merge_lists_v1(labels_list):
    temp = copy.deepcopy(labels_list)
    merged = []
    while len(temp) > 0:
        m = set()
        poped_idx = []
        m.update(temp[0])
        poped_idx.append(0)
        for i, label in enumerate(temp[1:], 1):
            a_l = np.array(list(m))
            b_l = np.repeat(label, len(a_l), axis=0).reshape(len(label), len(a_l))
            if np.any(a_l == b_l):
                m.update(label)
                poped_idx.append(i)
        
        merged.append(list(m))
        
        temp_ = []
        for i, t in enumerate(temp):
            if i not in poped_idx:
                temp_.append(t)
        temp = temp_

    return merged

def merge_lists(ml, itr=3):
    for _ in range(itr):
        ml = merge_lists_v1(ml)
    return ml

def mergeByendpoints_v1(label_eps, threshold=5, use_degree=True):
    """
    """
    labels_ = label_eps.keys()
    eps_ = label_eps.values()

    # trans endpoints, labels
    labels = []
    eps = []
    des = []
    for l, (ep,de) in label_eps.items():
        for e, d in zip(ep, de):
            labels.append(l)
            eps.append(e)
            des.append(d)

    dist = squareform(pdist(eps))
    degree = np.arccos(1 - squareform(pdist(des, metric='cosine'))) / math.pi * 180
    np.fill_diagonal(dist, np.inf)

    # 相同实例点的距离 设为np.inf
    for i in range(dist.shape[0]):
        dist[i, labels==labels[i]] = np.inf

    #计算最近邻的点
    min0 = np.min(dist, axis=0)
    idx0 = np.argmin(dist, axis=0)
    min1 = np.min(dist, axis=1)
    idx1 = np.argmin(dist, axis=1)

    #只保留满足阈值条件的最近邻点
    valid_min0 = np.where(min0<threshold)[0]
    valid_min1 = np.where(min1<threshold)[0]

    # valid_idx0 = idx0[np.where(min0<threshold)[0]]
    # valid_idx1 = idx1[np.where(min1<threshold)[0]]


    # 寻找满足阈值的点
    valid_idx = np.stack(np.where(dist<threshold), axis=0).T

    if use_degree:
        maybeSame = []
        for _, vm in enumerate(valid_idx):
            if True:#degree[vm[0], vm[1]] > 70:
                temp = [labels[vm[0]], labels[vm[1]]]
                temp.sort()
                maybeSame.append(temp)
    else:
        # 将最近邻点配对
        maybeSame = []
        for _, (vm0, _) in enumerate(zip(valid_min0, valid_min1)):
            # if vm0 == idx1[idx0[vm0]]:
            if degree[vm0, idx0[vm0]] > 90:
                temp = [labels[vm0], labels[idx0[vm0]]]
                temp.sort()
                maybeSame.append(temp)

    # 去除两两配对中的重复
    s = set()
    for i in maybeSame:
        s.add(tuple(i))
    s = sorted(s, key=lambda k:k[0])

    s_ = []
    for i in s:
        if i[0] != i[1]:
            s_.append(i)
    s = s_

    # 融合为一个链式关系
    merged_s = merge_lists(s)
    merged_s = [[int(i) for i in ms] for ms in merged_s]

    # # 合并label
    # merged_tif = tif_remove.copy()

    # for ms in merged_s:
    #     label = labels[ms[0]]
    #     for l in ms[1:]:
    #         merged_tif[merged_tif==labels[l]] = label
    return merged_s


def skels_cal(fov_ins_array, nums_cpu = 1, anisotropy=(200,200,1000)):
    skels = kimimaro.skeletonize(
        fov_ins_array, 
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
            dust_threshold=0,
            anisotropy=anisotropy, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=False, # default False
            parallel=nums_cpu, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )

    fov_ins_skel_array = np.zeros_like(fov_ins_array)
    fov_ins_ends_array = np.zeros_like(fov_ins_array)
    ends_dict = {}
    vecs_dict = {}
    for label in skels:
        skel = skels[label]
        ends, vecs = ends_cal(skel, anisotropy)

        coords = (skel.vertices / np.array(anisotropy)).astype(int)
        
        fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label
        fov_ins_ends_array[ends[:,0], ends[:,1], ends[:,2]] = label
        ends_dict[label] = ends
        vecs_dict[label] = vecs

    return ends_dict, vecs_dict, fov_ins_skel_array, fov_ins_ends_array

def ends_cal(skel, anisotropy=(200,200,1000)):
    edges = skel.edges
    coords = (skel.vertices / np.array([200, 200, 1000])).astype(int)
    
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

def skel_ends(tif):

    labels_eps = {}

    labels = np.unique(tif)[1:]

    ends_dict, vecs_dict, skels_array, ends_array = skels_cal(tif)
    # ins_skel_array, skel = skels_cal(ins_i)

    # ends, vecs = ends_cal(skel)
    # labels_eps[label] = [ends.tolist(), vecs.tolist()]
    for label in labels:
        labels_eps[label] = [ends_dict[label].tolist(), vecs_dict[label].tolist()]

    return labels_eps, ends_array, skels_array
        

def merged_a_tif(tif_path, out_path=None, imsaved=True):
    root, name = os.path.split(tif_path)
    name = name.split('.')[0]
    if out_path==None:
        out_path = root
    print("Processing:", name)
    tif = tifffile.imread(tif_path)
    ends_dict, ends_array, skel_array = skel_ends(tif)
    merged_s = mergeByendpoints_v1(ends_dict)

    merged_s = [sorted(s) for s in merged_s]

    merged_tif = np.zeros_like(tif)

    # 相同实例, 赋给相同的label
    for ms in merged_s:
        label = ms[0]
        for l in ms[1:]:
            merged_tif[tif==l] = label
    
    merged_tif[merged_tif==0] = tif[merged_tif==0]

    print(merged_s)
    print(len(np.unique(merged_tif))-1)

    if imsaved:
        tifffile.imsave(os.path.join(out_path, name+"_merged.tif"), merged_tif.astype(np.uint16))
        # tifffile.imsave(os.path.join(out_path, name+"_ends.tif"), ends_array.astype(np.uint16))
        # tifffile.imsave(os.path.join(out_path, name+"_skels.tif"), skel_array.astype(np.uint16))

    return merged_tif

if __name__ == "__main__":
    from skimage.external import tifffile

    # a = np.zeros((100, 100, 100))
    # a[50, 50, 14:67] = 1
    # a[50, 50, 70:94] = 2
    # a[50, 24:50, 69] = 3

    # a[50, 50, 14:67] = 1
    # a[50, 50, 70:94] = 2
    # a[50, 48, 35:68] = 3
    # labels = a

    file_path = "/media/jjx/Biology/logs/test_modified_multiseg_tcl_df/visual_results/5700_35350_4150_0/ins_pred_filter.tif"
    out_path = "./test"
    _ = merged_a_tif(file_path, out_path)

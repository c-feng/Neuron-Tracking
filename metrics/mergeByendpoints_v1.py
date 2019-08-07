import os
import numpy as np
import logging
import copy
import math
from scipy.spatial.distance import squareform, pdist
import skimage
from skimage.morphology import dilation, ball, closing
from skimage.external import tifffile
import kimimaro
import pdb


## *********************** ##
# 加入剪枝, 去除细小的分叉
def find_ends_angles_cross(skel, anisotropy=(200,200,1000)):
    # skel = skels[label]
    vertices = (skel.vertices / anisotropy).astype(int)
    edges = skel.edges

    # 找出端点和交叉点
    num_points = len(vertices)
    end_points = []
    cross_points = []
    angles = []
    for i in range(num_points):
        cnt = 0
        for edge in edges:
            if i in edge:
                cnt += 1

        if cnt == 1:
            end_points.append(vertices[i].tolist()+[i])
            idx0, idx1 = np.where(edges==i)
            idy1 = 0 if idx1 else 1
            angle = vertices[edges[idx0, idy1]] - vertices[i]
            angles.append(angle)
        if cnt > 2:
            cross_points.append(vertices[i].tolist()+[i])

    if len(cross_points) > 0:
        sel = prune_skeleton(edges, end_points, cross_points)
    else:
        return np.array(end_points)[:, :-1], np.array(angles)

    end_points = np.array(end_points)[sel, :-1]
    angles = np.array(angles)[sel]

    return end_points, angles

def prune_skeleton(edges, end_points, cross_points, min_length=20):
    assert len(cross_points) > 0, "the cross_points is empty"
    reserve_ends = []
    # for cp in cross_points:
    cross_id = np.array(cross_points)[:, -1]

    for i, ep in enumerate(end_points):
        id_ = ep[-1]
        idx0, idx1 = np.where(edges==id_)
        idy1 = 0 if idx1 else 1
        length = 1
        while edges[idx0, idy1] not in cross_id:
            id_ = edges[idx0, idy1]
            idx0s, _ = np.where(edges==id_)

            for j in idx0s:
                if j != idx0:
                    idx0 = j
                    break

            idx1 = np.where(edges[idx0]==id_)[0]
            idy1 = 0 if idx1 else 1
            length += 1
        
        if length > min_length:
            reserve_ends.append(i)

    return reserve_ends
## ******************************* ##

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
            if True:#degree[vm[0], vm[1]] > 100:
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


def skels_cal(fov_ins_array, nums_cpu=1, anisotropy=(200,200,1000)):
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
        # ends, vecs = ends_cal(skel, anisotropy)
        ends, vecs = find_ends_angles_cross(skel, anisotropy)

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
        

def merged_a_tif(tif_path, out_path=None, tif_name=None, logger=None, imsaved=True):
    root, name = os.path.split(tif_path)
    name = name.split('.')[0]
    if out_path==None:
        out_path = root
    
    # logger = set_logger(log_file)
    # print("Processing:", name)
    logger.info("Processing:{}".format(name))

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

    # print(merged_s)
    # print(len(np.unique(merged_tif))-1)
    # logger.info('\n'.join(merged_s))
    for m in merged_s:
        logger.info(m)
    logger.info("{} ----> {}\n\n".format(len(np.unique(tif))-1, len(np.unique(merged_tif))-1))

    if imsaved:
        if tif_name == None:
            tifffile.imsave(os.path.join(out_path, name+"_merged.tif"), merged_tif.astype(np.uint16))
            # tifffile.imsave(os.path.join(out_path, name+"_ends.tif"), ends_array.astype(np.uint16))
            # tifffile.imsave(os.path.join(out_path, name+"_skels.tif"), skel_array.astype(np.uint16))
        else:
            tifffile.imsave(os.path.join(out_path, tif_name+"_"+name+"_merged.tif"), merged_tif.astype(np.uint16))

    return merged_tif

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_logger(log_file, name="log_metric"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

def merge_some():
    # file_root = "/media/jjx/Biology/logs/test_301_80_dsmc_emb/visual_results/"
    file_root = "/media/jjx/Biology/logs/test_301_80_dsc_new/visual_results/"
    out_path = "./results/test_301_80_dsc_new"
    mkdir_if_not_exist(os.path.join(out_path))

    log_file = "./results/test_301_80_dsc_new/log_merged_"+"test_301_80_dsc_new.txt"
    logger = set_logger(log_file)

    file_paths = os.listdir(file_root)
    # file_paths = ["test_120_80_dsc_new_embseg"]
    for file_path in file_paths:

        logger.info("{}:".format(file_path))

        _ = merged_a_tif(os.path.join(file_root, file_path, "ins_pred.tif"), out_path, file_path, logger)
        # _ = merged_a_tif(os.path.join(file_root, "ins_pred.tif"), out_path, file_path, logger)


if __name__ == "__main__":

#     # a = np.zeros((100, 100, 100))
#     # a[50, 50, 14:67] = 1
#     # a[50, 50, 70:94] = 2
#     # a[50, 24:50, 69] = 3

#     # a[50, 50, 14:67] = 1
#     # a[50, 50, 70:94] = 2
#     # a[50, 48, 35:68] = 3
#     # labels = a

#     # file_path = "/media/jjx/Biology/logs/test_modified_multiseg_tcl_df/visual_results/5700_35350_4150_0/ins_pred.tif"
#     # file_path = "/media/jjx/Biology/logs/test/visual_results/5700_35350_4150_0/ins_pred.tif"
#     # file_path = "/media/jjx/Biology/logs/test_301_80/visual_results/5700_35350_4150_0/ins_pred_filter.tif"

#     file_path = "/media/fcheng/NeuralTrackcf/eval/realData/fov64_DF_finetune/5700_35350_3900_pred.tif"
#     out_path = "./results/"

#     mkdir_if_not_exist(out_path)
#     _ = merged_a_tif(file_path, out_path, "5700_35350_3900_pred")
    merge_some()

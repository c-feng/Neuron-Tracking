import numpy as np
import copy
from scipy.spatial.distance import squareform, pdist
from skimage.external import tifffile
import kimimaro
import pdb
from treelib import Node, Tree

def fov_connect(fov_ins_array):
    def parent(edges, i):
        coords = np.where( edges == i )
        edge = edges[ coords[0][0] ]
        if edge[0] == i:
            return edge[1] + 1
        return edge[0] + 1
     
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
            dust_threshold=10,
            anisotropy=(200,200,1000), # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=True, # default False
            parallel=2, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )
    ends_dict = {}

    fov_ins_skel_array = np.zeros_like(fov_ins_array)
    ends_array = np.zeros_like(fov_ins_array)
    for label_ in skels:
        skel = skels[label_]

        coords = (skel.vertices / np.array([200, 200, 1000])).astype(int)
        fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label_

        coords = coords.tolist()
        edges = skel.edges.tolist()

        ftree = Tree()
        cur_ = edges[0][0]
        ftree.create_node(cur_, cur_, data = coords[0])

        cur_list = [cur_]

        while(len(edges) > 0 and len(cur_list) > 0):
            _cur_list = []
            edges_ = edges[:]
            #print(cur_list)
            for cur_ in cur_list:
                next_inds = np.where(np.array(edges_) == cur_)[0]
                if len(next_inds) == 0:continue
                for next_ind in next_inds:
                    edge_ = edges_[next_ind]
                    edges.remove(edge_)
                    #print(cur_, edge_)

                    if edge_[0] == cur_:
                        next_ = edge_[-1]
                    else:
                        next_ = edge_[0]

                    _cur_list.append(next_)
                    ftree.create_node(next_, next_, data = coords[next_], parent = cur_)
                edges_ = edges[:]

            cur_list = _cur_list

        ends = [x.data for x in ftree.leaves()]
        ends.append(coords[0])

        ends_dict[label_] = ends
        
        ends_ = np.array(ends)
        ends_array[ends_[:, 0], ends_[:, 1], ends_[:, 2]] = 1
        #ends_array = dilation(ends_array, ball(1))

    return fov_ins_skel_array, ends_array, ends_dict

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

def mergeByendpoints(label_eps, threshold=7):
    """
    """
    labels_ = label_eps.keys()
    eps_ = label_eps.values()

    # trans endpoints, labels
    labels = []
    eps = []
    for l, ep in label_eps.items():
        for e in ep:
            labels.append(l)
            eps.append(e)

    dist = squareform(pdist(eps))
    np.fill_diagonal(dist, np.inf)

    # 相同实例点的距离 设为np.inf
    for i in range(dist.shape[0]):
        dist[i, labels==labels[i]] = np.inf

    # 计算最近邻的点
    min0 = np.min(dist, axis=0)
    idx0 = np.argmin(dist, axis=0)
    min1 = np.min(dist, axis=1)
    idx1 = np.argmin(dist, axis=1)

    # 只保留满足阈值条件的最近邻点
    valid_min0 = np.where(min0<threshold)[0]
    valid_min1 = np.where(min1<threshold)[0]

    # valid_idx0 = idx0[np.where(min0<threshold)[0]]
    # valid_idx1 = idx1[np.where(min1<threshold)[0]]

    # 将最近邻点配对
    maybeSame = []
    for _, (vm0, _) in enumerate(zip(valid_min0, valid_min1)):
        # if vm0 == idx1[idx0[vm0]]:
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


if __name__ == "__main__":
        
    labels = tifffile.imread("./ins_pred.tif")
    fov_ins_skel_array, ends_array, ends_dict = fov_connect(labels)
    merged_s = mergeByendpoints(ends_dict)

    merged_tif = labels.copy()

    # 相同实例, 赋给相同的label
    for ms in merged_s:
        label = ms[0]
        for l in ms[1:]:
            merged_tif[merged_tif==l] = label

    print(merged_s)
    tifffile.imsave("merged.tif", merged_tif.astype(np.uint8))
    tifffile.imsave("ends.tif", ends_array.astype(np.uint8))
    tifffile.imsave("skel.tif", fov_ins_skel_array.astype(np.uint8))

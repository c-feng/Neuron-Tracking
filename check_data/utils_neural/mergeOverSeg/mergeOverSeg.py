import numpy as np
import os
import time
import copy
from queue import Queue
from multiprocessing import Pool
from skimage.external import tifffile
from skimage.morphology import remove_small_objects
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from functools import partial

def findendpoints(mask):
    axis = np.stack( np.where(mask > 0), axis=0).transpose()

    axis0_min = np.min(axis[:, 0])
    axis0_max = np.max(axis[:, 0])

    axis1_min = np.min(axis[:, 1])
    axis1_max = np.max(axis[:, 1])
    if axis.shape[1] == 3:
        axis2_min = np.min(axis[:, 2])
        axis2_max = np.max(axis[:, 2])

    a = np.mean(list(filter(lambda i: axis0_min==i[0], axis.tolist())), axis=0)
    b = np.mean(list(filter(lambda i: axis0_max==i[0], axis.tolist())), axis=0)
    c = np.mean(list(filter(lambda i: axis1_min==i[1], axis.tolist())), axis=0)
    d = np.mean(list(filter(lambda i: axis1_max==i[1], axis.tolist())), axis=0)
    if axis.shape[1] == 3:
        e = np.mean(list(filter(lambda i: axis2_min==i[2], axis.tolist())), axis=0)
        f = np.mean(list(filter(lambda i: axis2_max==i[2], axis.tolist())), axis=0)

    if axis.shape[1] == 3:
        points = np.stack([a, b, c, d, e, f], axis=0)
    else:
        points = np.stack([a, b, c, d], axis=0)
    index = list(combinations(range(len(points)), 2))
    dist = pdist(points)
    idx = index[np.argmax(dist)]

    return points[idx, :], dist

def wraper(label, mask):
    p, _ = findendpoints(mask==label)
    return p

def mutilprocess_func(tif, labels, num_proc=12):
    with Pool(processes=num_proc) as pool:
        ps = pool.map(partial(wraper, mask=tif), labels)
    return ps

def merge_lists(labels_list):
    """ 将标签对融合在一起,
        存在一个list中
    """
    list_arr = np.array(labels_list)
    list_arr = np.concatenate([list_arr, np.zeros((len(list_arr), 1))], axis=1)

    # 根据(a0, a1) a1的值来找融合关系
    merged = []
    for i, a in enumerate(list_arr):
        if a[2] == 0:
            m = [a[0], a[1]]
            list_arr[i][2] = 1

            idx = np.where(list_arr[:, 0]==a[1])[0]
            if len(idx) > 0:
                queue = Queue()
                for j in idx:
                    queue.put(j)
                
                while not queue.empty():
                    idx = queue.get()
                    m += [list_arr[idx][1]]
                    list_arr[idx][2] = 1
                    
                    idx = np.where(list_arr[:, 0]==list_arr[idx][1])[0]
                    if len(idx) > 0:
                        for j in idx:
                            queue.put(j)

            merged.append(m)
    
    # 根据(a0, a1) a0的值再次融合
    l0 = set()
    for i in merged:
        l0.add(i[0])
    l0 = list(l0)
    l0.sort()
    merged_ = []
    for l in l0:
        temp = set()
        t = filter(lambda x: x[0]==l, merged)
        temp.update(*t)
        temp = list(temp)
        temp.sort()
        merged_.append(temp)

    return merged_

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


if __name__ == "__main__":
        
    # file_path = r"./Syn_chaos_300_5_2_4_6_080_pred.tif"
    file_path = r"H:\temp\realData\5700_35600_4150_pred.tif"

    root = os.path.dirname(file_path)
    name = os.path.basename(file_path)

    tif = tifffile.imread(file_path)

    tif_remove = remove_small_objects(tif.astype(int), min_size=25)
    labels = np.unique(tif_remove)[1:]
    print(labels, len(labels))
    # print(np.unique(tif_remove), len(np.unique(tif_remove)))

    endpoints = []
    start = time.time()
    # for label in labels:
    #     ins = tif_remove == label
    #     p, _ = findendpoints(ins)
    #     endpoints.append(p)
    # 获取每个实例的端点
    endpoints = mutilprocess_func(tif_remove, labels, num_proc=4)

    print('***********', "time:{}s".format(time.time()-start))
    print(np.array(endpoints).shape)

    a_eps = np.array(endpoints)[:, 0, :]
    b_eps = np.array(endpoints)[:, 1, :]
    ab_eps = np.concatenate([a_eps, b_eps], axis=0)
    num = len(a_eps)  # 实例个数

    # 计算所有端点之间的距离
    dist = squareform(pdist(ab_eps))
    np.fill_diagonal(dist, np.inf)

    # 计算最近邻的点
    min0 = np.min(dist, axis=0)
    idx0 = np.argmin(dist, axis=0)
    min1 = np.min(dist, axis=1)
    idx1 = np.argmin(dist, axis=1)

    # 只保留满足阈值条件的最近邻点
    threshold = 5
    valid_min0 = np.where(min0<threshold)[0]
    valid_min1 = np.where(min1<threshold)[0]

    valid_idx0 = idx0[np.where(min0<threshold)[0]]
    valid_idx1 = idx1[np.where(min1<threshold)[0]]
    
    # 将最近邻点配对
    maybeSame = []
    for _, (vm0, _) in enumerate(zip(valid_min0, valid_min1)):
        # if vm0 == idx1[idx0[vm0]]:
        temp = [vm0%num, idx0[vm0]%num]
        temp.sort()
        maybeSame.append(temp)

    # 去除两两配对中的重复
    s = set()
    for i in maybeSame:
        s.add(tuple(i))
    s = sorted(s, key=lambda k:k[0])

    # 融合为一个链式关系
    merged_s = merge_lists(s)
    merged_s = merge_lists_v1(s)
    merged_s = [[int(i) for i in ms] for ms in merged_s]

    # 合并label
    merged_tif = tif_remove.copy()

    for ms in merged_s:
        label = labels[ms[0]]
        for l in ms[1:]:
            merged_tif[merged_tif==labels[l]] = label
    
    # tifffile.imsave(os.path.join(root, name.split('.')[0] + "_merged_v1.tif"), merged_tif.astype(np.float16))

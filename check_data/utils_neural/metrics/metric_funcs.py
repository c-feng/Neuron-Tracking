import numpy as np
from scipy.spatial.distance import cdist

## ##    ********************    ## ##
##            PAT metric            ##
## ##    ********************    ## ##
def match(ins_i, inses, threshold=5):
    coords = np.stack(np.where(ins_i), axis=0).T
    lens_i = len(coords)

    match_labels = []
    ratios = []
    for j in np.unique(inses)[1:]:
        inses_j = (inses == j).astype(int)
        j_coords = np.stack(np.where(inses_j), axis=1).T

        dist = cdist(coords, j_coords)
        min_dist = np.min(dist, axis=1)
        # min_idx = np.argmin(dist, axis=1)

        valid_min = np.where(min_dist < threshold)[0]
        if len(valid_min) == 0:
            match_labels.append(None)
            ratios.append(0)
        else:
            match_labels.append(j)
            ratios.append(len(valid_min) / lens_i)
        
    r = np.max(ratios)
    m_l = match_labels[np.argmax(ratios)]

    return m_l, r

def match_fp(ins_j, inses, threshold=5):
    coords = np.stack(np.where(ins_i), axis=0).T
    lens_j = len(coords)

    match_labels = []
    ratios = []
    for i in np.unique(inses)[1:]:
        inses_i = (inses == i).astype(int)
        i_coords = np.stack(np.where(inses_i), axis=0).T

        dist = cdist(coords, i_coords)
        min_dist = np.min(dist, axis=1)

        valid_min = np.where(min_dist < threshold)[0]
        if len(valid_min) == 0:
            match_labels.append(None)
            ratios.append(1.)
        else:
            match_labels.append(i)
            ratios.append(len(valid_min))

    r = (lens_j - np.max(ratios)) / lens_j
    m_l = match_labels[np.argmax(ratios)]

    return m_l, r

def pat_TP(pred_mask, gt_mask, logger=None):
    """ TP(g_i) = len(g_i ∩ D(g_i)) / len(g_i)
        其中: D(g_i) = argmax(len(g_i ∩ g^_j)), g_i为ground truth中的一根实例, g^_j为预测中的一个连通域
        TP ratio是ground truth中被正确预测出来的片段的比例
    """
    labels = np.unique(gt_mask)[1:]
    
    TPs = []
    label_pairs = []
    for label in labels:
        ins = (gt_mask == label).astype(int)
        m, r = match(ins, pred_mask)
        if logger is not None:
            logger.info("label:{}({}), match:{}({}), precision:{}".format(label,
                                 np.sum(ins>0), m, np.sum(pred_mask==m), r))
        TPs.append(r)
        label_pairs.append([label, m])
    
    return np.mean(TPs), TPs, label_pairs

def pat_FP(pred_mask, gt_mask, logger=None):
    """ FP(g^_j) = len(g^_j - (g^_j ∩ D'(g^_j))) / len(g^_j)
        其中: D'(g^_j) = argmax(len(g^_j ∩ g_i)), g_i为ground truth中的一根实例, g^_j为预测中的一个连通域
        FP ratio是pred中错误片段的比例
    """
    labels = np.unique(pred_mask)
    
    FPs = []
    label_pairs = []
    for label in labels:
        ins = (pred_mask == label).astype(int)
        m, r = match_fp(ins, gt_mask)
        if logger is not None:
            logger.info("label:{}({}), match:{}({}), precision:{}".format(label, 
                                 np.sum(ins>0), m, np.sum(gt_mask==m), r))
        FPs.append(r)
        label_pairs.append([label, m])
    
    return np.mean(FPs), FPs, label_pairs

def pat_precision(pred_mask, gt_mask):
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]

    _, TPs, tp_pairs = pat_TP(pred_mask, gt_mask)
    _, FPs, fp_pairs = pat_FP(pred_mask, gt_mask)

    accurate = []
    for i, p in enumerate(tp_pairs):
        j = np.where(pred_labels==p[1])[0][0]
        fp_p = fp_pairs[j]
        if p[0] == fp_p[1]:
            if TPs[i] >= 0.9 and FPs[j] <= 0.1:
                accurate.append(p)
    
    return len(accurate) / len(gt_labels), len(accurate)
## ##  ******************** ## ##


## ##    ********************    ## ##
##          MES/ADE metric          ##
## ##    ********************    ## ##
# MES = (S_G - S_miss) / (S_G + S_extra)
# ADE: Average-Displacement-Error. 平均位移误差
def match_mes(inses1, inses2, threshold=5):
    """ 在inses2中找到inses1中每个实例的匹配ins
    """
    pass


def mes(skeleton1, skeleton2, threshold=0.2):
    """ Trace MES is defined as the ratio of the gold standard length reduced by
        the false negtive length(gt中未被检测到的部分) to the gold standard length
        reduced by the false positive length(pred中多余的部分).
    """
    pass
## ##  ******************** ## ##


## ##    ********************    ## ##
##            FFN metric            ##
## ##    ********************    ## ##
def metric_a_single_skel(skel, B_coords, label, threshold):
    """ Input:
            skel: 单根骨架化的实例
            B_coords: (N, 4), 记录预测中所有实例的x, y, z, label
        Return:
            min_idx: 单个ground truth实例的每个骨架节点, 在预测中的匹配点
            a_labels: 单个ground truth实例的每个骨架节点, 在预测中的标签
    """
    # edges = skel.edges
    coords = skel.vertices

    # 计算gt中一个skeleton的节点和所有预测节点的距离, 并作匹配
    dist = cdist(coords, B_coords[:, :2])
    min_dist = np.min(dist, axis=1)
    min_idx = np.argmin(dist, axis=1)

    # 小于一定距离的 被视作有效匹配
    valid_mask = (min_dist < threshold).astype(int)

    # gt中一个skeleton的所有结点 对应有效匹配 的节点 的距离
    # 非有效匹配的距离 置为无穷大
    min_dist[valid_mask==0] = np.inf

    # gt中一个skeleton的所有结点 对应有效匹配 在预测中的点
    # 非有效匹配的节点, 对应标签 置为-1
    min_idx[valid_mask==0] = -1

    # 一个预测节点 只能 匹配一个gt节点, 其他gt节点视作未被检测到
    idx_unique = np.unique(min_idx)
    for i in idx_unique:
        if i == -1: continue
        # 找出一个预测节点 匹配多个的gt节点
        re_idx = np.where(min_idx==i)[0]
        if len(re_idx) > 1:
            m_i = np.argmin(min_dist[re_idx])
            m_i = re_idx[m_i]

            not_m_i = []
            for j in re_idx:
                if j != m_i:
                    not_m_i.append(j)
            
            min_dist[not_m_i] = np.inf
            min_idx[not_m_i] = -1
    
    # skeleton中每个节点的 预测标签
    a_labels = np.zeros(len(coords))
    a_labels[min_idx!=-1] = B_coords[min_idx[min_idx!=-1], 2]

    return a_labels



def skel_match(skelsA, skelsB, shape, threshold=5):
    """ an edge e is defined as correctly reconstructed if both of its nodes
        belong to the same object in te
    """
    # skelsB_array = np.zeros(shape)

    # 将skelsB, vertices提取, 并赋label, 得到[num_points, 4], 4:x, y, z, label
    B_coords = []
    for i in skelsB:
        n, d = skelsB[i].vertices.shape

        # 增加一列, 记录label信息
        v = i * np.ones([n, d+1])
        v[:, :3] = skelsB[i].vertices
        B_coords += v.tolist()
    B_coords = np.array(B_coords)
        # skelsB_array[B_coords[0], B_coords[1], B_coords[2]] = i

    match = {}
    for label in skelsA:
        skel = skelsA[label]
        edges = skel.edges
        coords = skel.vertices
        
        # 计算gt中一个skeleton的节点和所有预测节点的距离, 并作匹配
        dist = cdist(coords, B_coords[:3])
        min_dist = np.min(dist, axis=1)
        min_idx = np.argmin(dist, axis=1)

        # 小于一定距离的 被视作有效匹配
        valid_mask = (min_dist < threshold).astype(int)
        
        # gt中一个skeleton的所有结点 对应有效匹配 的节点 的距离
        min_dist[valid_mask == 0] = np.inf
        
        # gt中一个skeleton的所有结点 对应有效匹配 的节点 的标签
        min_idx[valid_mask == 0] = -1

        # 一个预测节点 只能 匹配一个gt节点, 其他gt节点视作未被检测到
        idx_unique = np.unique(min_idx)
        for i in idx_unique:
            if i == -1: continue
            # temp = min_idx[min_idx == i]
            # 找出一个预测节点 匹配的多个gt节点
            re_idx = np.where(min_idx==i)[0]
            if len(re_idx) > 1:
                m_i = np.argmin(dist[re_idx])
                m_i = re_idx[m_i]
                
                not_m_i = []
                for j in re_idx:
                    if j != m_i:
                        not_m_i.append(j)
                
                dist[not_m_i] = np.inf
                min_idx[not_m_i] = -1
        
        match[label] = [dist.tolist(), min_idx.tolist()]
    
        # 统计split, merge, omitted, correct
        # omitted, if R(e) = 0
        # split, if R(A(e)) != R(B(e))
        # 
        split = []
        merge = []
        correct = []
        for edge in edges:
            A = edge[0]
            B = edge[1]
            # if 


def debug_test():
    skelA_array = None
    skel_arr = np.zeros((20, 20), dtype=np.float16)
#    skel_arr[3, 4:17] = 1
    skel_arr[3:13, 16] = 2

    skelB_arr = np.zeros((20, 20), dtype=np.float16)
    skelB_arr[3, 4:16] = 2
    skelB_arr[2:13, 12] = 4
    B_coords = np.zeros((np.sum(skelB_arr>0), 3))
    t = np.stack(np.where(skelB_arr==2), axis=0).T
    num = len(t)
    B_coords[:num, :2] = t
    B_coords[:num, 2] = 2
    t = np.stack(np.where(skelB_arr==4), axis=0).T
    B_coords[num:, :2] = t
    B_coords[num:, 2] = 4

    class Skel_Class():
        def __init__(self, vertices):
            self.vertices = vertices
    v = np.stack(np.where(skel_arr>0), axis=0).T
    skel = Skel_Class(v)

    a_labels = metric_a_single_skel(skel, B_coords, None, 5)

    match_mask = np.zeros_like(skel_arr, dtype=np.float16)
    for i, l in enumerate(a_labels):
        match_mask[v[i][0], v[i][1]] = l
    
    print("")

debug_test()








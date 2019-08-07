import numpy as np
from scipy.spatial.distance import cdist
from queue import Queue
import copy
import pdb

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
        j_coords = np.stack(np.where(inses_j), axis=0).T


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
    coords = np.stack(np.where(ins_j), axis=0).T
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
            logger.info("label:{}({}), match:{}({}), tp ratio:{}".format(label,
                                 np.sum(ins>0), m, np.sum(pred_mask==m), r))
        TPs.append(r)
        label_pairs.append([label, m])
    
    return np.mean(TPs), TPs, label_pairs

def pat_FP(pred_mask, gt_mask, logger=None):
    """ FP(g^_j) = len(g^_j - (g^_j ∩ D'(g^_j))) / len(g^_j)
        其中: D'(g^_j) = argmax(len(g^_j ∩ g_i)), g_i为ground truth中的一根实例, g^_j为预测中的一个连通域
        FP ratio是pred中错误片段的比例
    """
    labels = np.unique(pred_mask)[1:]
    
    FPs = []
    label_pairs = []
    for label in labels:
        ins = (pred_mask == label).astype(int)
        m, r = match_fp(ins, gt_mask)
        if logger is not None:
            logger.info("label:{}({}), match:{}({}), fp ratio:{}".format(label, 
                                 np.sum(ins>0), m, np.sum(gt_mask==m), r))
        FPs.append(r)
        label_pairs.append([label, m])
    
    return np.mean(FPs), FPs, label_pairs

def pat_precision(pred_mask, gt_mask, logger=None):
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]

    _, TPs, tp_pairs = pat_TP(pred_mask, gt_mask, logger)
    _, FPs, fp_pairs = pat_FP(pred_mask, gt_mask, logger)

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
def match_a_single_skel(skel, B_coords, label, threshold):
    """ Input:
            skel: 单根骨架化的实例
            B_coords: (N, 4), 记录B中所有实例的骨架点x, y, z, label
        Return:
            min_idx: A中单个实例的每个骨架节点, 在B中的匹配点
            a_labels: A中单个实例的每个骨架节点, 在B中的匹配点的label标签
    """
    # edges = skel.edges
    coords = skel.vertices

    # 计算A中一个skeleton的节点和B中所有节点的距离, 并作匹配
    # 由于直接取得最小值点, 所以不存在B中多个节点匹配一个skeleton节点的情况
    dist = cdist(coords, B_coords[:, :3])
    min_dist = np.min(dist, axis=1)
    min_idx = np.argmin(dist, axis=1)

    # 小于一定距离的 被视作有效匹配
    valid_mask = (min_dist < threshold).astype(int)

    # A中一个skeleton的所有结点 对应有效匹配 的节点 的距离
    # 非有效匹配的距离 置为无穷大
    min_dist[valid_mask==0] = np.inf

    # A中一个skeleton的所有结点 对应有效匹配 在预测中的点编号
    # 非有效匹配的节点, 对应编号 置为-1
    min_idx[valid_mask==0] = -1

    # 一个B节点 只能 匹配一个skeleton节点, 其他skeleton节点视作未被检测到
    idx_unique = np.unique(min_idx)
    for i in idx_unique:
        if i == -1: continue
        # 找出一个B节点 匹配的多个skeleton节点
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
    
    # skeleton中每个节点对应的B中的匹配点的label标签, 没有匹配的点对应的label置0
    a_labels = np.zeros(len(coords))
    a_labels[min_idx!=-1] = B_coords[min_idx[min_idx!=-1], 3]

    return a_labels, min_idx

def is_merged(a, i, match_labels):
    """ 判断点a是否为merged, a的匹配点所在的实例, 同时还存在其它点被匹配
        给另一个skel, 即为merge
    """
    ml_a = match_labels[i][a]
    for j, match_label in enumerate(match_labels):
        if j == i: continue
        for ml in match_label:
            if ml == ml_a:
                return True
    
    return False

def distinguish_merged(class_es, match_labels):
    """ 判断e是否为merged, e的两个端点只要有一个点的匹配点所在的实例, 同时还存在
        其他点被匹配给另一个skel, 即为merge
    """
    merges = []
    for i, class_e in enumerate(class_es):
        ms = []
        for e in class_e:
            a, b, _ = e
            if match_labels[i][a] != 0 and match_labels[i][b] !=0:
                if is_merged(a, i, match_labels) or is_merged(b, i, match_labels):
                    ms.append(e)
        merges.append(ms)
    
    for ms, class_e in zip(merges, class_es):
        for m in ms:
            for i in range(len(class_e)):
                if m == class_e[i]:
                    class_e[i][-1] = 2
    
    # return class_es

def distinguish_merged_v1(class_es, match_labels):
    """ 判断e是否为merged, e的两个端点只要有一个点的匹配点所在的实例, 同时还存在
        其他点被匹配给另一个skel, 即为merge
    """
    merges = []
    for i, class_e in enumerate(class_es):
        ms = []
        for e in class_e:
            a, b, _ = e
            if match_labels[i][a] != 0 and match_labels[i][b] !=0:
                if is_merged(a, i, match_labels) or is_merged(b, i, match_labels):
                    ms.append(e)
        merges.append(ms)
    
    # 去除merges中的重复记录
    merges_set = []
    for merge in merges:
        mss = set()
        for ms in merge:
            mss.add(ms[-1])
        merges_set.append(list(mss))

    for ms, class_e in zip(merges_set, class_es):
        for m in ms:
            idx = np.where(np.array(class_e)[:, -1]==m)[0]
            sel = np.random.choice(idx, 1)[0]
            class_e[sel][-1] = 2

    # return class_es

def classify_edge(edges, match_labels):
    """ omitted if R(e)=0, 0
        split if R(A(e)) != R(B(e)), 1
        merged if R(e) =R(e_m) and S(e) != S(e_m), 2
        correct if none of the above, 3
    """
    class_e = []
    omitted = []
    split = []
    merged = []
    correct = []
    for e in edges:
        A, B = e
        # edges是gt中的skeleton
        if match_labels[A] == 0 or match_labels[B] == 0:
            # omitted
            class_e.append([A, B, 0])
            omitted.append([A, B])
        elif match_labels[A] != match_labels[B]:
            # split
            class_e.append([A, B, 1])
            split.append([A, B])
        elif match_labels[A] == match_labels[B]:
            class_e.append([A, B, 3])
            correct.append([A, B])
        else:  # 没有限制条件的话, merged统计是不对的
            # merged 在预测中匹配点所在的连通域 同时也匹配了其他skeleton
            class_e.append([A, B, 2])
            merged.append([A, B])

    return class_e

def find_CRCs(class_e):
    """ For a given skeleton S, let CE(S) denote the set of correct edges.
        "correctly reconstructed components"(CRCs)--subsets of edges corresponding to
        valid (without a merger) segments in R.
            CRC(S, L) = {e: e ∈ CE(S) and R(e)=L}
    """
    crcs = []

    idx = np.where(np.array(class_e)[:, -1] == 3)[0]
    correct_e = np.zeros([len(idx), 4]) # [A, B, c, flag], flag指明当前edge是否被分配给crc
    correct_e[:, :3] = np.array(class_e)[idx, :]
    while not np.all(correct_e[:, -1]==1):
        crc = []
        for i, ce in enumerate(correct_e):
            if ce[-1] == 0: break
        crc.append(ce[:3])
        ce[-1] = 1
        a, b, _, _ = ce
        
        # 找到下一个包含点a, b的edge
        idxa_, _ = np.where(correct_e[:, :2]==a)
        idxb_, _ = np.where(correct_e[:, :2]==b)

        # 剔除当前edge位置
        idxa, idxb = [], []
        for j in idxa_:
            if j != i:
                idxa.append(j)
        for j in idxb_:
            if j != i:
                idxb.append(j)

        # 将待搜索节点加入队列
        find_queue = Queue()
        for j in idxa:
            find_queue.put([j, a])
        for j in idxb:
            find_queue.put([j, b])
        
        while not find_queue.empty():
            ind, ab = find_queue.get()
            assert correct_e[ind, -1] == 0, "Something Wrong!!!"
            
            crc.append(correct_e[ind, :3])
            correct_e[ind, -1] = 1
            
            a, b, _, _ = correct_e[ind]
            next_ab = b if ab == a else a
            idx_, _ = np.where(correct_e[:, :2]==next_ab)
            idx = []
            for j in idx_:
                if j != ind:
                    idx.append(j)
            for j in idx:
                if correct_e[j, -1] == 0:
                    find_queue.put([j, next_ab])

        crcs.append(crc)

    return crcs


def cal_ERL(class_e, coords):
    """ ERL(expected run length) is the expected size of the segment that contains a
        randomly selected skeleton node.
            ERL(S) = ∑_L ||CRC(S, L)|| * ||CRC(S, L)|| / ||S||, 其中||S|| = ∑||e||.
        则：
            ERL({S_k}) = ∑_k w_k * ERL(S_k), 其中w_k = ||S_k|| / ∑_i||S_i|| 
        For a given skeleton S, let CE(S) denote the set of correct edges.
        "correctly reconstructed components"(CRCs)--subsets of edges corresponding to
        valid (without a merger) segments in R.
            CRC(S, L) = {e: e ∈ CE(S) and R(e)=L}
    """
    # 将edges分割为多个CRC, 
    CRCs = find_CRCs(class_e)
    
    # 计算||S||, 即骨架点的总长度
    total_len = 0
    for e in class_e:
        total_len += np.linalg.norm(coords[int(e[0])] - coords[int(e[1])])

    # ERL_S = 0
    # # 计算单个skeletonde ERL
    # for crc in CRCs:
    #     ERL_S_i = 0
    #     len_S_i = 0
    #     for e in crc:
    #         len_S_i += np.linalg.norm(coords[e[0]] - coords[e[1]])
        
    #     # w_i = 
    #     # ERL_S_i = 
    ERL = 0
    for crc in CRCs:
        len_crc = 0
        for e in crc:
            len_crc += np.linalg.norm(coords[int(e[0])] - coords[int(e[1])])

        ERL += len_crc * (len_crc / total_len)
    
    return ERL, total_len

def cal_skels_ERL(class_es, coords):
    """ ERL(expected run length) is the expected size of the segment that contains a
        randomly selected skeleton node.
            ERL(S) = ∑_L ||CRC(S, L)|| * ||CRC(S, L)|| / ||S||, 其中||S|| = ∑||e||.
        则：
            ERL({S_k}) = ∑_k w_k * ERL(S_k), 其中w_k = ||S_k|| / ∑_i||S_i|| 
        For a given skeleton S, let CE(S) denote the set of correct edges.
        "correctly reconstructed components"(CRCs)--subsets of edges corresponding to
        valid (without a merger) segments in R.
            CRC(S, L) = {e: e ∈ CE(S) and R(e)=L}
    """
    ERL_S = 0
    S_len = 0
    for class_e, coord in zip(class_es, coords):
        for e in class_e:
            S_len += np.linalg.norm(coord[int(e[0])] - coord[int(e[1])])
    
    for class_e, coord in zip(class_es, coords):
        erl, S_i_len = cal_ERL(class_e, coord)
        ERL_S += erl * (S_i_len / S_len)
    
    return ERL_S


def skels_metric(skelsA, skelsB, threshold=5):
    """ an edge e is defined as correctly reconstructed if both of its nodes
        belong to the same object in the reconstruction
    """
    labels_A = list(skelsA.keys())
    labels_A.sort()
    # labels_B = list(skelsB.keys())
    # labels_B.sort()

    # 将skelsB, vertices提取, 并赋label, 得到[num_points, 4], 4:x, y, z, label
    B_coords = []
    for i in skelsB:
        n, d = skelsB[i].vertices.shape

        # 增加一列, 记录label信息
        v = i * np.ones([n, d+1])
        v[:, :3] = skelsB[i].vertices
        B_coords += v.tolist()
    B_coords = np.array(B_coords)
    
    # 将skelsA, vertices提取, 并赋label, 得到[num_points, 4], 4:x, y, z, label
    # A_coords = []
    # for i in skelsA:
    #     n, d = skelsA[i].vertices.shape

    #     # 增加一列, 记录label信息
    #     v = i * np.ones([n, d+1])
    #     v[:, :3] = skelsA[i].vertices
    #     A_coords += v.tolist()
    # A_coords = np.array(A_coords)

    # # 增加一列, 记录A中每个骨架点在B中找到的匹配节点的类别
    # A_coords_ml = np.concatenate([A_coords, np.zeros((len(A_coords), 1))], axis=1)

    # 对skelsA中的每个骨架实例的每个edge做分类, 分为omitted, merged, split, correct
    match_labels = []
    # match_idxBs = []
    class_es = []
    for label in labels_A:
        skel = skelsA[label]
        edges = skel.edges
        
        # A中单个实例的每个骨架节点, 在B中匹配的节点的标签
        a_labels, _ = match_a_single_skel(skel, B_coords, label, threshold)
        class_e = classify_edge(edges, a_labels)
        
        match_labels.append(a_labels)
        # match_idxBs.append(a_idxB)
        class_es.append(class_e)
    
    class_es_v1 = copy.deepcopy(class_es)
    class_es_ = copy.deepcopy(class_es)
    # class_es_v1 = distinguish_merged_v1(class_es_v1, match_labels)
    # class_es_ = distinguish_merged(class_es, match_labels)
    distinguish_merged_v1(class_es_v1, match_labels)
    distinguish_merged(class_es_, match_labels)

    # skelsA 节点坐标
    coordsA = []
    for label in labels_A:
        coordsA.append(skelsA[label].vertices / (200,200,1000))
    ERL = cal_skels_ERL(class_es_, coordsA)

    # 统计split:1, merge:2, omitted:0, correct:3
    # omitted, if R(e) = 0
    # split, if R(A(e)) != R(B(e))
    # 
    num_edges = 0
    omits_num = []
    splits_num = []
    mergeds_num = []
    corrects_num = [] 
    for class_e in class_es_v1:
        o, s, m, c = 0, 0, 0, 0
        for ce in class_e:
            if ce[-1] == 0:
                o += 1
            elif ce[-1] == 1:
                s += 1
            elif ce[-1] == 2:
                m += 1
            elif ce[-1] == 3:
                c += 1
            else:
                raise Exception('class error')
        num_edges += len(class_e)
        omits_num.append(o)
        splits_num.append(s)
        mergeds_num.append(m)
        corrects_num.append(c)
    
    statistics = {}
    statistics["omits_num"] = omits_num
    statistics["splits_num"] = splits_num
    statistics["merged_num"] = mergeds_num
    statistics["corrects_num"] = corrects_num
    statistics["omit_ratio"] = np.sum(omits_num) / num_edges
    statistics["split_ratio"] = np.sum(splits_num) / num_edges
    statistics["merged_ratio"] = np.sum(mergeds_num) / num_edges
    statistics["correct_rotio"] = np.sum(corrects_num) / num_edges

    statistics["erl"] = ERL

    return statistics


def debug_test():
    skelA_array = None
    skel_arr = np.zeros((20, 20))
    skel_arr[3, 4:17] = 1
    # skel_arr[3:13, 19] = 2

    skelB_arr = np.zeros((20, 20))
    skelB_arr[3, 4:16] = 2
    skelB_arr[3:13, 18] = 4
    B_coords = np.zeros((np.sum(skelB_arr>0), 4))
    t = np.stack(np.where(skelB_arr==2), axis=0).T
    num = len(t)
    B_coords[:num, :3] = t
    B_coords[:num, 3] = 2
    t = np.stack(np.where(skelB_arr==4), axis=0).T
    B_coords[num:, :3] = t
    B_coords[num:, 3] = 4

    class Skel_Class():
        def __init__(self, vertices):
            self.vertices = vertices
    v = np.stack(np.where(skel_arr>0), axis=0).T
    skel = Skel_Class(v)

    a_labels = match_a_single_skel(skel, B_coords, None, 2)

    match_mask = np.zeros_like(skel_arr)
    for i, l in enumerate(a_labels):
        match_mask[v[i][0], v[i][1]] = l
    








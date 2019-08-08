import numpy as np
from skimage.morphology import dilation, disk

def find_maxOverlaps(preds, ins):
    """ 在preds中, 找到与ins重叠最大的实例预测
        preds: 多个实例mask
        ins: 单个实例
    """
    #gap = 2
    mask = np.zeros(ins.shape)
    coords = np.stack(np.where((preds>0) == (ins>0)), axis=0).T
    if len(ins.shape) == 2:
        mask[coords[:, 0], coords[:, 1]] = preds[coords[:, 0], coords[:, 1]]
    elif len(ins.shape) == 3:
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = preds[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    overlap_labels = np.unique(mask)
    if len(overlap_labels) == 1:
        return None
    
    max_overlap = 0
    max_label = 0
    for label in overlap_labels[1:]:
        if max_overlap < np.sum(mask==label):
            max_overlap = np.sum(mask==label)
            max_label = label
    
    max_ins = np.zeros(ins.shape)
    max_ins[preds==max_label] = preds[preds==max_label]

    return max_ins

def find_intersection(ins1, ins2):
    """ 找到ins1, ins2的重叠区域
    """
    intersec = np.zeros(ins1.shape)
    intersec = np.array(((ins1>0) == (ins2>0)) & (ins1>0) & (ins2>0)).astype(int)
    return intersec

def cal_TP(preds, gts):
    """ 按论文"PAT - Probabilistic Axon Tracking for Densely Labeled Neurons in Large 3D Micrographs"中的计算方法
        计算TP
        preds: 实例预测, (w, h, d)
        gts: 实例labels, (w, h, d)

        return:
            TPs: (num_gts, )
    """
    labels = np.unique(gts)[1:]
    TPs = []
    max_Overlap_label = []
    for i, label in enumerate(labels):
        gt_ins = (gts==label).astype(int)
        D_g_i = find_maxOverlaps(preds, gt_ins)
        if D_g_i is None:
            max_Overlap_label.append(0)
            TPs.append(0)
            continue
        TP = np.sum(find_intersection(D_g_i, gt_ins) > 0) / np.sum(gt_ins > 0)

        max_Overlap_label.append(np.unique(D_g_i)[-1])
        TPs.append(TP)
    
    return TPs, max_Overlap_label

def cal_FP(preds, gts):
    """ 计算FP"PAT - Probabilistic Axon Tracking for Densely Labeled Neurons in Large 3D Micrographs"
        preds: 实例预测, (w, h, d)
        gts: 实例labels, (w, h, d)

        return:
            FPs: (num_preds)
    """
    labels = np.unique(preds)[1:]
    FPs = []
    max_Overlap_label = []
    for i, label in enumerate(labels):
        pred_ins = (preds==label).astype(int)
        D_g = find_maxOverlaps(gts, pred_ins)
        if D_g is None:
            max_Overlap_label.append(0)
            FPs.append(1.)
            continue
        FP = (np.sum(pred_ins>0) - np.sum(find_intersection(D_g, pred_ins))) / \
                np.sum(pred_ins>0)
        
        max_Overlap_label.append(np.unique(D_g)[-1])
        FPs.append(FP)
    
    return FPs, max_Overlap_label

def cal_precision(preds, gts, tp_threshold=0.9, fp_threshold=0.1):
    """ Detected accurately if (g_i, g_j) with TP > 0.9, FP < 0.1
        precision: correctly detected pates
    """
    TPs, tplabels = cal_TP(preds, gts)
    FPs, fplabels = cal_FP(preds, gts)
    
    correct_pred = set()
    correspond_pred = []
    # correspond_gt = []
    for i, (tp, label) in enumerate(zip(TPs, tplabels)):
        if ( fplabels[int(label)-1] == i+1
             and tp >= tp_threshold 
             and label > 0 
             and FPs[int(label)-1] <= fp_threshold ):
            correct_pred.add(i+1)
            correspond_pred.append(label)
        else:
            correspond_pred.append(0.)
    # for _, (fp, label) in enumerate(zip(FPs, fplabels)):
    #     if fp <= fp_threshold and label > 0 and TPs[int(label)-1] >= tp_threshold:
    #         correct_pred.add(label)
    #         correspond_gt.append(label)
    #     else:
    #         correspond_gt.append(0.)
    result = [[i, correspond_pred[i-1]] for i in correct_pred]

    return len(correct_pred) / len(TPs), result #list(correct_pred), list(correspond_pred)#, list(correspond_gt)

if __name__ == "__main__":
    # test cal_TP
    gts = np.zeros((10, 10))
    gts[2:8, 4] = 1
    gts[0:9, 8] = 2
    preds = np.zeros((10, 10))
    preds[4:6, 3:6] = 3
    preds[7:9, 4:9] = 2
    preds[0:5, 8] = 1
    tps, tplabels = cal_TP(preds, gts)
    fps, fplabels = cal_FP(preds, gts)
    prec, r = cal_precision(preds, gts, 0.3, 0.81)

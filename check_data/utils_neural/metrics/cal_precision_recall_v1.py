import os
import sys
import numpy as np
from skimage.external import tifffile
from scipy.spatial.distance import cdist
import pdb


def match(ins_i, mask, threshold=5):
    """ ins_i: 单个实例mask
        mask: 多个实例mask
    """
    coords = np.stack(np.where(ins_i), axis=0).T
    lens = len(coords)

    c = np.stack(np.where(mask), axis=0).T
    
    dist = cdist(coords, c)

    mask_d = (dist < threshold).astype(np.int16)
    match_p = np.sum(mask_d, axis=1)
    
    precision = np.sum(match_p>0) / lens

    return precision

def match_v1(ins_i, inses, threshold=5):
    coords = np.stack(np.where(ins_i), axis=0).T
    
    inses_c = np.stack(np.where(inses), axis=0).T
    inses_labels = inses[inses_c[:, 0], inses_c[:, 1], inses_c[:, 2]]

    dist = cdist(coords, inses_c)
    
    min_dist = np.min(dist, axis=1)  # 找到每个点匹配的最近邻点
    min_idx = np.argmin(dist, axis=1)

    valid_min = np.where(min_dist < threshold)[0]
    if len(valid_min) == 0:
        return None, 0
    remove_repeat = list(set(min_idx[valid_min])) 
    valid_labels = inses_labels[remove_repeat]

    labels = np.unique(valid_labels)
    ratios = [np.sum(valid_labels==x)/np.sum(inses==x) for x in labels]
    match_label = labels[np.argmax(ratios)]


    return match_label, np.max(ratios)


def cal_precision(pred_mask, gt_mask):
    labels = np.unique(pred_mask)[1:]

    precisions = []
    for label in labels:
        ins = (pred_mask == label).astype(int)
        _, p = match_v1(ins, gt_mask)
        print("label:{}, precision:{}".format(label, p))
        precisions.append(p)
    
    return np.mean(precisions), precisions

def cal_recall(pred_mask, gt_mask):
    labels = np.unique(gt_mask)[1:]
    
    recalls = []
    for label in labels:
        ins = (gt_mask == label).astype(int)
        _, p = match_v1(ins, pred_mask)
        print("label:{}, recall:{}".format(label, p))
        recalls.append(p)
    
    return np.mean(recalls), recalls


if __name__ == "__main__":
    # pred_path = r"H:\metric\gtree\3450_31350_5150_pred.tif"
    # gt_path = r"H:\metric\gtree\3450_31350_5150_gt.tif"
    pred_root = r"H:\metric\gtree\ins"
    gt_root = r"H:\dataset\Neural\data_modified\ins_modified"

    names = os.listdir(pred_root)

    precs = []
    recalls = []
    for name in names:

        pred_path = os.path.join(pred_root, name)
        gt_path = os.path.join(gt_root, name)

        pred = tifffile.imread(pred_path)
        gt = tifffile.imread(gt_path)

        print(name, ":")
        mp, ps = cal_precision(pred, gt)
        print("precision:", mp)
        # print("#"*10)

        mr, rs = cal_recall(pred, gt)
        print("recall:", mr)
        print("\n")

        precs.append(mp)
        recalls.append(mr)
    
    print("Average precision:", np.mean(precs))
    print("Average recall:", np.mean(recalls))
import numpy as np
from skimage.external import tifffile
from scipy.spatial.distance import cdist
import time
import pdb

def match(ins_i, mask, threshold=5):
    """ ins_i: 单个实例mask
        mask: 多个实例mask
    """
    coords = np.stack(np.where(ins_i), axis=0).T
    lens = len(coords)

    if len(ins_i.shape) == 3:
        c = np.array(np.unravel_index(np.arange(ins_i.size), shape=ins_i.shape)).reshape(3, -1).T
    elif len(a.shape) == 2:
        c = np.array(np.unravel_index(np.arange(ins_i.size), shape=ins_i.shape)).reshape(2, -1).T
    start = time.time()
    dist = cdist(coords, c)
    print("time:", time.time()-start)

    sel = np.stack(np.where(dist<threshold), axis=0).T
    pdb.set_trace()

    sel_points = mask[sel[0, :], sel[1, :], sel[2, :]]
    sel_labels = np.unique(sel_points[sel_points>0])
    percent = [np.sum(sel_points==x)/lens for x in sel_labels]
    match_label = sel_labels[np.argmax(percent)]

    return match_label, np.max(percent)


def cal_precision(pred_mask, gt_mask):
    labels = np.unique(pred_mask)[1:]

    precisions = []
    for label in labels:
        ins = (pred_mask == label).astype(int)
        _, p = match(ins, gt_mask)
        precisions.append(p)
    
    return np.mean(precisions)

def cal_recall(pred_mask, gt_mask):
    labels = np.unique(gt_mask)[1:]

    # precision

if __name__ == "__main__":
    pred_path = r"./3450_31350_5150_pred.tif"
    gt_path = r"./3450_31350_5150_gt.tif"
    
    pred = tifffile.imread(pred_path)
    gt = tifffile.imread(gt_path)

    print(cal_precision(pred, gt))
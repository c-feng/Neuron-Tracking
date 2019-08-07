import os
import sys
import logging
import glob
import errno
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
    lens = len(coords)
    
    inses_c = np.stack(np.where(inses), axis=0).T
    inses_labels = inses[inses_c[:, 0], inses_c[:, 1], inses_c[:, 2]]

    dist = cdist(coords, inses_c)
    
    min_dist = np.min(dist, axis=1)  # 找到每个点匹配的最近邻点
    min_idx = np.argmin(dist, axis=1)

    valid_min = np.where(min_dist < threshold)[0]
    if len(valid_min) == 0:
        return None, 0
    # remove_repeat = list(set(min_idx[valid_min])) 
    # valid_labels = inses_labels[remove_repeat]
    valid_labels = inses_labels[min_idx[valid_min]]

    labels = np.unique(valid_labels)
    # ratios = [np.sum(valid_labels==x)/np.sum(inses==x) for x in labels]  # 除以gt长度
    ratios = [np.sum(valid_labels==x)/lens for x in labels]  # 除以pred
    match_label = labels[np.argmax(ratios)]


    return match_label, np.max(ratios)

def match_recall(ins_i, inses, threshold=5):
    coords = np.stack(np.where(ins_i), axis=0).T
    lens = len(coords)
    
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
    ratios = [np.sum(valid_labels==x)/lens for x in labels]
    match_label = labels[np.argmax(ratios)]

    return match_label, np.max(ratios)


def cal_precision(pred_mask, gt_mask, logger):
    labels = np.unique(pred_mask)[1:]

    precisions = []
    for label in labels:
        ins = (pred_mask == label).astype(int)
        m, p = match_v1(ins, gt_mask)
        # print("label:{}, match:{}, precision:{}".format(label, m, p))
        logger.info("label:{}({}), match:{}({}), precision:{}".format(label, np.sum(ins>0), m, np.sum(gt_mask==m), p))
        precisions.append(p)
    
    return np.mean(precisions), precisions

def cal_recall(pred_mask, gt_mask, logger):
    labels = np.unique(gt_mask)[1:]
    
    recalls = []
    for label in labels:
        ins = (gt_mask == label).astype(int)
        m, p = match_recall(ins, pred_mask)
        # print("label:{}, match:{}, recall:{}".format(label, m, p))
        logger.info("label:{}({}), match:{}({}), recall:{}".format(label, np.sum(ins), m, np.sum(pred_mask==m), p))
        recalls.append(p)
    
    return np.mean(recalls), recalls

def metric_a_tif(pred_path, gt_path, logger=None):
    pred = tifffile.imread(pred_path)
    gt = tifffile.imread(gt_path)
    
    _, name = os.path.split(pred_path)
    name = name.split('.')[0]

    # print(name, ":")
    logger.info("{}:".format(name))
    mp, ps = cal_precision(pred, gt, logger)
    # print("precision:", mp)
    logger.info("Precision:{}".format(mp))
    # print("#"*10)

    mr, rs = cal_recall(pred, gt, logger)
    # print("recall:", mr)
    # print("\n")
    logger.info("Recall:{}".format(mr))
    logger.info("\n")

    return mp, mr


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

def metric_some():
    pred_root = "/media/fcheng/merge_tif/results/test_301_80_dsc_new/"
    gt_root = "/media/jjx/Biology/logs/test_301_80_dsc_new/visual_results/"

    log_file = "./log_metric_" + "test_301_80_dsc_new.txt"
    logger = set_logger(log_file)

    logger.info("GT path:{}\n".format(gt_root))
    precs = []
    recalls = []
    names = glob.glob(os.path.join(pred_root, "*.tif"))
    names = [os.path.split(name)[1] for name in names]
    # names = ["ins_pred_filter_merged.tif"]
    for name in names:

        pred_path = os.path.join(pred_root, name)
        tif_name = "_".join(name.split('_')[:4])
        gt_path = os.path.join(gt_root, tif_name, "ins_gt.tif")
        # gt_path = os.path.join(gt_root, name)

        mp, mr = metric_a_tif(pred_path, gt_path, logger)

        precs.append(mp)
        recalls.append(mr)

    logger.info("Average Precision:{}".format(np.mean(precs)))
    logger.info("Average Recall:{}".format(np.mean(recalls)))


metric_some()
# if __name__ == "__main__":
#     # pred_path = r"H:\metric\gtree\3450_31350_5150_pred.tif"
#     # gt_path = r"H:\metric\gtree\3450_31350_5150_gt.tif"
#     # pred_root = r"H:\metric\gtree\ins"
#     # gt_root = r"H:\dataset\Neural\data_modified\ins_modified"

#     pred_root = "/media/fcheng/NeuralTrackcf/eval/realData/fov64_DF_finetune/"
#     gt_root = "/media/jjx/Biology/data/data_modified/ins_modified"

#     # pred_root = "/media/jjx/Biology/data/gtree_pred_test_301/ins/"
#     # gt_root = "/media/jjx/Biology/data/data_modified_test_301/ins/"
    
#     # pred_root = "/media/fcheng/merge_tif/results/"
#     # gt_root = "/media/jjx/Biology/logs/test_301_80/visual_results/5700_35350_4150_0/"

#     # pred_root = "/media/fcheng/merge_tif/results/test_301_80_dsmc_emb/"
#     # gt_root = "/media/jjx/Biology/logs/test_301_80_2/visual_results/"

#     # log_file = './log_metric_gtree_pred_test_301.txt'
#     log_file = './log_metric_5700_35350_3900_pred.txt'
#     logger = set_logger(log_file)

#     logger.info("\n\nGT path:{}".format(gt_root))
#     precs = []
#     recalls = []
#     # names = os.listdir(pred_root)
#     names = ["5700_35350_3900_pred.tif"]
#     for name in names:

#         pred_path = os.path.join(pred_root, name)
#         # gt_path = os.path.join(gt_root, "ins_gt.tif")
#         # gt_path = os.path.join(gt_root, name)
#         gt_path = os.path.join(gt_root, "_".join(name.split('_')[:3])+".tif")


#         mp, mr = metric_a_tif(pred_path, gt_path, logger)

#         precs.append(mp)
#         recalls.append(mr)

#     logger.info("Average Precision:{}".format(np.mean(precs)))
#     logger.info("Average Recall:{}".format(np.mean(recalls)))

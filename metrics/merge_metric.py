import os
import logging
import glob
import numpy as np
from mergeByendpoints_v1 import merged_a_tif
from cal_precision_recall_v1 import metric_a_tif

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

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def merge(log_name):
    # file_root = "/media/jjx/Biology/logs/test_301_80_dsmc_emb/visual_results/"
    file_root = "/media/jjx/Biology/logs/"+log_name+"/visual_results/"
    out_path = "./results/"+log_name
    mkdir_if_not_exist(os.path.join(out_path))

    log_file = "./results/"+log_name+"/log_merged_"+log_name+".txt"
    logger = set_logger(log_file, "log_merge")

    file_paths = os.listdir(file_root)
    # file_paths = ["test_120_80_dsc_new_embseg"]
    for file_path in file_paths:

        logger.info("{}:".format(file_path))

        _ = merged_a_tif(os.path.join(file_root, file_path, "ins_pred.tif"), out_path, file_path, logger)
        # _ = merged_a_tif(os.path.join(file_root, "ins_pred.tif"), out_path, file_path, logger)

def metric(log_name):
    pred_root = "./results/"+log_name
    gt_root = "/media/jjx/Biology/logs/"+log_name+"/visual_results/"

    log_file = "./results/"+log_name+"/log_metric_" +log_name+".txt"
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

log_name = "test_301_48_embseg"
merge(log_name)
metric(log_name)
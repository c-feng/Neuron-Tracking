import os
import numpy as np
import glob
from tqdm import tqdm
import logging
from skimage.external import tifffile

from cal_metrics.metric_funcs import pat_precision


def set_logger(log_file, name="log_metric_pat"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

def func1():
    root_path = "/home/jjx/Biology/logs/test_dsmc_emb_center_crop_synthesize/visual_results/"
    out_path = "./results/test_dsmc_emb_center_crop_synthesize/"
    os.makedirs(out_path, exist_ok=True)

    log_file = os.path.join(out_path, "metric_PAT_gap_10_test_dsmc_emb_center_crop_synthesize.txt")
    logger = set_logger(log_file, "log_metric_pat")

    names = os.listdir(root_path)
    names.sort()
    precs = []
    for name in tqdm(names):
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))

        for ins_path in ins_paths:
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split("_")[:-1])

            seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_seg.tif"))
            gt = tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif"))
            # over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_over_seg.tif")))
            # under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_under_seg.tif")))
            # cross_seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_cross_seg.tif"))
            seg = seg[10:-10, 10:-10, 10:-10]
            gt = gt[10:-10, 10:-10, 10:-10]

            logger.info("{}_{}:".format(name, ins_name))
            precision, _ = pat_precision(seg, gt, logger)
            precs.append(precision)
            logger.info("{}_{} precision: {}".format(name, ins_name, precision))
            logger.info("")

    logger.info("Average of total {} is {}".format(len(precs), np.mean(precs)))

def func2():
    root_path = "/home/jjx/Biology/logs/test_synthesize_embcls_tcl_64_center_crop_merge_0.5/visual_results/"
    out_path = "./results/test_synthesize_embcls_tcl_64_center_crop_merge_0.5/"
    os.makedirs(out_path, exist_ok=True)

    log_file = os.path.join(out_path, "metric_PAT_test_synthesize_embcls_tcl_64_center_crop_merge_0.5.txt")
    logger = set_logger(log_file, "log_metric_pat")

    names = os.listdir(root_path)
    names.sort()
    precs = []
    for name in tqdm(names):
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))

        for ins_path in ins_paths:
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split("_")[:-1])

            seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_merge.tif"))
            gt = tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif"))
            # over_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_over_seg.tif")))
            # under_seg = orderedLabel(tifffile.imread(os.path.join(root_path, name, ins_name+"_under_seg.tif")))
            # cross_seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_cross_seg.tif"))
            seg = seg[10:-10, 10:-10, 10:-10]
            gt = gt[10:-10, 10:-10, 10:-10]

            logger.info("{}_{}:".format(name, ins_name))
            precision, _ = pat_precision(seg, gt, logger)
            precs.append(precision)
            logger.info("{}_{} precision: {}".format(name, ins_name, precision))
            logger.info("")

    logger.info("Average of total {} is {}".format(len(precs), np.mean(precs)))

if __name__ == "__main__":
    func2()
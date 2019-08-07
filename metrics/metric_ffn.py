import os
import numpy as np
import glob
from tqdm import tqdm
import logging
from skimage.external import tifffile
import kimimaro

from cal_metrics.metric_funcs import skels_metric
import pdb


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

def skeleton_func(ins_mask, nums_cpu=1, anisotropy=(200,200,1000), return_arr=False):
    skels = kimimaro.skeletonize(
        ins_mask, 
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
            dust_threshold=0,
            anisotropy=anisotropy, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=False, # default False
            parallel=nums_cpu, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )
    # for i in skels:
    #     skels[i].vertices = skels[i].vertices / np.array(anisotropy)
    ins_array = np.zeros_like(ins_mask)
    
    if return_arr:
        for label in skels:
            skel = skels[label]
            # ends, vecs = ends_cal(skel, anisotropy)

            coords = (skel.vertices / np.array(anisotropy)).astype(int)

            ins_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label

        return skels, ins_array
    else:
        return skels

def func1():
    root_path = "/home/jjx/Biology/logs/test_dsmc_emb_center_crop_synthesize/visual_results/"
    out_path = "./results/test_dsmc_emb_center_crop_synthesize/"

    os.makedirs(out_path, exist_ok=True)

    log_file = os.path.join(out_path, "metric_FFN_test_dsmc_emb_center_crop_synthesize.txt")
    logger = set_logger(log_file, "log_metric_ffn")

    names = os.listdir(root_path)
    names.sort()

    omit_ratio = []
    split_ratio = []
    merged_ratio = []
    correct_ratio = []
    ERLs = []
    for name in tqdm(names):
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))

        for ins_path in ins_paths:
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split("_")[:-1])

            seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_seg.tif"))
            gt = tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif"))

            seg_skel = skeleton_func(seg, nums_cpu=1, anisotropy=(200,200,1000), return_arr=False)
            gt_skel = skeleton_func(gt, nums_cpu=1, anisotropy=(200,200,1000), return_arr=False)

            statistics = skels_metric(gt_skel, seg_skel)
            omit_ratio.append(statistics["omit_ratio"])
            split_ratio.append(statistics["split_ratio"])
            merged_ratio.append(statistics["merged_ratio"])
            correct_ratio.append(statistics["correct_rotio"])
            ERLs.append(statistics["erl"])
            # tifffile.imsave(os.path.join(out_path, name+"_"+ins_name+"_gt_skel_arr.tif"), seg_skel_arr)
            logger.info("{}_{}:".format(name, ins_name))
            logger.info("omit_ratio: {}".format(omit_ratio[-1]))
            logger.info("split_ratio: {}".format(split_ratio[-1]))
            logger.info("merged_ratio: {}".format(merged_ratio[-1]))
            logger.info("correct_ratio: {}".format(correct_ratio[-1]))
            logger.info("expected run length: {} Voxel".format(ERLs[-1]))
            logger.info("")

    logger.info("Average omit_ratio of total {} is {}".format(len(omit_ratio), np.mean(omit_ratio)))
    logger.info("Average split_ratio of total {} is {}".format(len(split_ratio), np.mean(split_ratio)))
    logger.info("Average merged_ratio of total {} is {}".format(len(merged_ratio), np.mean(merged_ratio)))
    logger.info("Average correct_ratio of total {} is {}".format(len(correct_ratio), np.mean(correct_ratio)))
    logger.info("Average expect run length of total {} is {} Voxel".format(len(ERLs), np.mean(ERLs)))

def func2():
    root_path = "/home/jjx/Biology/logs/test_dsc_emb_center_crop/visual_results/"
    out_path = "./results/statistic_gt/"

    os.makedirs(out_path, exist_ok=True)

    log_file = os.path.join(out_path, "metric_statistic_gt.txt")
    logger = set_logger(log_file, "log_metric_ffn")

    names = os.listdir(root_path)
    names.sort()

    omit_ratio = []
    split_ratio = []
    merged_ratio = []
    correct_ratio = []
    ERLs = []
    lens = []
    for name in tqdm(names):
        ins_paths = glob.glob(os.path.join(root_path, name, "*_gt.tif"))

        for ins_path in ins_paths:
            rd = os.path.basename(ins_path).split(".")[0]
            ins_name = "_".join(rd.split("_")[:-1])

            seg = tifffile.imread(os.path.join(root_path, name, ins_name+"_seg.tif"))
            gt = tifffile.imread(os.path.join(root_path, name, ins_name+"_gt.tif"))

            seg_skel = skeleton_func(seg, nums_cpu=1, anisotropy=(200,200,1000), return_arr=False)
            gt_skel = skeleton_func(gt, nums_cpu=1, anisotropy=(200,200,1000), return_arr=False)


            len_skels = []
            for i in gt_skel:
                skel = gt_skel[i]
                coords = skel.vertices / (200,200,1000)

                len_s = 0
                for e in skel.edges:
                    len_s += np.linalg.norm(coords[int(e[0])] - coords[int(e[1])])
                len_skels.append(len_s)
            
            logger.info("{}_{}:".format(name, ins_name))
            logger.info("average length: {}".format(np.mean(len_skels)))
            logger.info("")
            lens.append(np.mean(len_skels))

    logger.info("Average of total {} is {}".format(len(lens), np.mean(lens)))

if __name__ == "__main__":
    func1()



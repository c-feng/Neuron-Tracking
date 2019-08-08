import numpy as np
import os.path as osp

from skimage.external import tifffile
from skimage.morphology import dilation, ball
from glob import glob

from ..utils.serialization import read_json
from ..utils.osutils import mkdir_if_missing 

def process(ins_p, info_p, \
        tiffs_dir, gts_dir, ends_dir, juncs_dir, ins_dir, centerlines_dir):
    ins = tifffile.imread(ins_p)
    ins = dilation(ins, ball(1))

    tiff = (ins > 0).astype(np.uint16)

    seg = (np.logical_and(ins > 0, ins != 1000)).astype(np.uint8)
    ins[seg == 0] = 0 
    #seg = (ins > 0).astype(np.uint8)
    
    end = np.zeros_like(ins, dtype = np.uint8)
    end_info = read_json(info_p)

    end_coords = np.array(end_info["end"])
    cross_coords = np.array(end_info["cross"])
    
    end[end_coords[:,0], end_coords[:,1], end_coords[:,2]] = 1
    end[cross_coords[:,0], cross_coords[:,1], cross_coords[:,2]] = 2

    fname = osp.basename(ins_p)
    print(fname)
    tiff_p = osp.join(tiffs_dir, fname)
    gt_p = osp.join(gts_dir, fname)

    end_p = osp.join(ends_dir, fname)

    junc_p = osp.join(juncs_dir, fname)
    ins_p = osp.join(ins_dir, fname)
    centerline_p = osp.join(centerlines_dir, fname)

    tifffile.imsave(tiff_p, tiff)
    tifffile.imsave(gt_p, seg)

    tifffile.imsave(end_p, end)
    
    tifffile.imsave(centerline_p, seg)
    tifffile.imsave(junc_p, seg)
    tifffile.imsave(ins_p, ins)

target_dir = "/home/jjx/Biology/data_synthesize"
ins_ps = glob("/media/fcheng/datasets/SyntheticData/1x10_3/*.tif")
info_ps = ["{}.json".format(osp.splitext(x)[0]) for x in ins_ps]

tiffs_dir = osp.join(target_dir, "tiffs") 
mkdir_if_missing(tiffs_dir)

gts_dir =  osp.join(target_dir, "gts") 
mkdir_if_missing(gts_dir)

ends_dir = osp.join(target_dir, "ends") 
mkdir_if_missing(ends_dir)

juncs_dir = osp.join(target_dir, "juncs")
mkdir_if_missing(juncs_dir)

ins_dir = osp.join(target_dir, "ins")
mkdir_if_missing(ins_dir)

centerlines_dir = osp.join(target_dir, "centerlines")
mkdir_if_missing(centerlines_dir)

#print(ins_ps[0], info_ps[0])

for ins_p, info_p in zip(ins_ps, info_ps):
    process(ins_p, info_p, \
        tiffs_dir, gts_dir, ends_dir, juncs_dir, ins_dir, centerlines_dir)

import numpy as np
import argparse
import os.path as osp

from skimage.external import tifffile
from glob import glob

from neuralTrack.utils.osutils import mkdir_if_missing

parser = argparse.ArgumentParser(description="Split tiffs into patches")

parser.add_argument('--source-dir', type=str, metavar='PATH')


shift = [1, 1, 1]

args = parser.parse_args()

fdir = args.source_dir
fpath_tiffs = glob(osp.join(fdir, "*/*.tif"))

for fpath_tiff in fpath_tiffs:
    print(fpath_tiff)
    tiff = tifffile.imread(fpath_tiff)
    tiff_shift_dir = "{}_shift".format(osp.dirname(fpath_tiff))
    mkdir_if_missing(tiff_shift_dir)

    fpath_tiff_shift = osp.join(tiff_shift_dir, osp.basename(fpath_tiff))
    
    inds_sel = np.arange(tiff.size)[(tiff > 0).flatten()]
    
    tiff_shift = np.zeros_like(tiff)
    coords_sel = np.array(np.unravel_index(inds_sel, tiff.shape))
    coords_sel_shift = coords_sel + np.array(shift)[:, None]
    xs_, ys_, zs_ = coords_sel_shift
    xs, ys, zs = coords_sel 
    
    xs_mask = np.logical_and(xs_ > 0, xs_ < tiff.shape[0])
    ys_mask = np.logical_and(ys_ > 0, ys_ < tiff.shape[0])
    zs_mask = np.logical_and(zs_ > 0, zs_ < tiff.shape[0])

    mask = xs_mask & ys_mask & zs_mask

    tiff_shift[xs_[mask], ys_[mask], zs_[mask] ] = tiff[xs[mask], ys[mask], zs[mask]]
    
    tifffile.imsave(fpath_tiff_shift, tiff_shift)



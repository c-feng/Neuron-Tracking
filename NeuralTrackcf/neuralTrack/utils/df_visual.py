from .direct_field import diret_field_cal,distance_transform
from skimage.external import tifffile
from sys import argv
import os.path as osp
import numpy as np

gt_path = argv[-1]
colors = np.array([1,2000],dtype = np.uint16)
gt = tifffile.imread(gt_path)
print(gt.shape)
edt,inds = distance_transform(gt)

gt_df = np.zeros(gt.shape,dtype = np.uint16)

mask = np.logical_and(edt>0,edt<=1)
#mask = np.logical_and(edt=0)
#mask = edt ==0
#print(mask.shape)
gt_df[mask] = 1
#print()
#gt_df = colors[np.logical_and(edt>0,edt<=1)]
print(gt.sum(),gt_df.sum())
tifffile.imsave(osp.splitext(osp.basename(gt_path))[0]+"_df.tif",colors[gt_df])
tifffile.imsave(osp.basename(gt_path),colors[gt])





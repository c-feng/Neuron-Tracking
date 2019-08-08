# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:28:10 2018

@author: Jueyean
"""

from skimage.external import tifffile
import numpy as np
import os.path as osp
import scipy.misc
import glob

colors = np.array([0,2000],np.uint16)

#ins_path = r"C:\Users\cf__e\Desktop\data_select\tif\5950_24100_3650.tif"
#tiff_dir = "C:/Users/cf__e/Desktop/check_data/data_80_modified/ins/"
tiff_dir = "C:/Users/cf__e/Desktop/check_data/data_80_modified/ins/"
ins_paths = glob.glob(osp.join(tiff_dir, '*.tif'))
output_path = "C:/Users/cf__e/Desktop/check_data/data_80_modified/"

for ins_path in ins_paths:
    fname = osp.splitext(osp.basename(ins_path))[0]
    fname_dir = osp.dirname(ins_path)
    ins = tifffile.imread(ins_path)
    
    ins_inds = np.unique(ins)[1:]
    print(ins_inds)
    ins_visual = []
    for i in ins_inds:
        #ins_i = np.zeros(ins.shape,dtype = np.bool)
        ins_i = ins ==i
        ins_i = ins_i.astype(np.uint16)
        fname_i = "{}_{}.tif".format(fname,i)
        #tifffile.imsave(osp.join(fname_dir,fname_i),colors[ins_i])
        
        ins_i = np.sum(ins_i,axis = 0)>0
        ins_i = ins_i.astype(np.int)
        #print(ins_i.shape)
        #print(np.sum(ins_i))
        
        
        ins_visual.append(colors[ins_i])
    fname_x = "{}_x.jpg".format(fname)
    ins_visual = np.array(ins_visual)
    ins_visual = np.pad(ins_visual, ((0,0),(1,1), (1,1)), 'constant', constant_values=2000)
    #print(ins_visual.shape, len(ins_visual))
    ins_visual = np.hstack(ins_visual)
    print(fname_x)
    scipy.misc.imsave(osp.join(output_path,fname_x),ins_visual)
       
    gt = ins>0
    gt = gt.astype(np.int)
    fname_gt = "{}_gt.tif".format(fname)
    #tifffile.imsave(osp.join(fname_dir,fname_gt),colors[gt])



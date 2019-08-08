# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:28:10 2018

@author: Jueyean
"""

from skimage.external import tifffile
import numpy as np
import os.path as osp
import scipy.misc
import matplotlib.pyplot as plt
import glob

colors = np.array([0,500],np.uint16)


def vis_square(data, file_path=None):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 0), (0, 0))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    padding = (((0, 0),
               (1, 1), (1, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data, cmap=plt.cm.gray); plt.axis('off')
    if file_path is not None:
        plt.imsave(file_path, data, cmap=plt.cm.gray)
    plt.show()


#ins_path = r"C:\Users\cf__e\Desktop\data_select\tif\5950_24100_3650.tif"
#tiff_dir = "C:/Users/cf__e/Desktop/check_data/data_80_modified/ins/"
tiff_dir = r"H:\ins_filter_1"
ins_paths = glob.glob(osp.join(tiff_dir, '*.tif'))[:]
output_path = r"H:\ins_filter_1\vis"

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
        tifffile.imsave(osp.join(output_path,fname_i),colors[ins_i])
        
        ins_i = np.sum(ins_i,axis = 0)>0
        ins_i = ins_i.astype(np.int)
        #print(ins_i.shape)
        #print(np.sum(ins_i))
        
        
        ins_visual.append(colors[ins_i])
    fname_x = "{}_x.jpg".format(fname)
    ins_visual = np.array(ins_visual)
    #ins_visual = np.pad(ins_visual, ((0,0),(1,1), (1,1)), 'constant', constant_values=2000)
    vis_square(ins_visual, osp.join(output_path,fname_x))
    print(ins_visual.shape, len(ins_visual))
    #ins_visual = np.hstack(ins_visual)
    print(fname_x)
    #scipy.misc.imsave(osp.join(output_path,fname_x),ins_visual)

    gt = ins>0
    gt = gt.astype(np.int)
    fname_gt = "{}_gt.tif".format(fname)
    tifffile.imsave(osp.join(output_path,fname_gt),colors[gt])



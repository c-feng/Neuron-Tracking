import json
import os
import random

import numpy as np

from multiprocessing import Pool
from functools import partial
from skimage import io
from skimage.external import tifffile

from .serialization import read_json,write_json
from .infos import DimsInfo
from .transform import image_pad

def coords_filter_single(sample_ind,sample_info,dims_info= None):
    gt_path = sample_info[-1]
    tiff = io.imread(sample_info[0])
    gt = io.imread(gt_path)
    gt_shape = tiff.shape
    gt.resize(gt_shape)
    del tiff
    seg_l  = dims_info.seg_l
    seg_r = dims_info.seg_r
    mask_range = dims_info.range_mask_get(gt_shape)
    mask_gt = gt.astype(np.bool)
    mask = np.logical_and(mask_range,mask_gt)
    inds = np.arange(gt.size)[mask.flatten()]
    
    mask_b = np.logical_and(mask_range,mask_gt == False)
    inds_b = np.arange(gt.size)[mask_b.flatten()]

    inds_b = np.random.choice(inds_b,inds.size*10)

    seg_l  = dims_info.seg_l
    seg_r = dims_info.seg_r
    segments = []
    for i in inds:
        segments.append([sample_ind,i,1])
    for i in inds_b:
        segments.append([sample_ind,i,0])

    return segments 

def coords_filter(samples_ind,samples_info,dims_info):
    cpu_nums = os.cpu_count()//2
    segments = []
    samples_num = len(samples_info)
    for sample_ind,sample_info in zip(samples_ind,samples_info):
        segments += coords_filter_single(sample_ind,sample_info,dims_info)
    return segments 

def content_cal(i,gt_shape,gt,sample_ind,dims_info):
    seg_l  = dims_info.seg_l
    seg_r = dims_info.seg_r
    coord = np.unravel_index(i,gt_shape)
    coord = np.array(coord)
    coord += dims_info.pad_size  
    #print(coord)
    coord_l = coord + 1 - seg_l
    coord_r = coord + 1 + seg_r  
    segment_gt = gt[coord_l[0]:coord_r[0],coord_l[1]:coord_r[1],coord_l[2]:coord_r[2]]

    n = np.sum(segment_gt)

    beta = round(float(n) / segment_gt.size,4)
    return [sample_ind,i,n,beta]
def rate_sample_single(sample_ind,sample_info,num_segment = 10000,dims_info = None,rate = 0.1):
    tiff = tifffile.imread(sample_info[0])
    gt = tifffile.imread(sample_info[-1])
    
    gt.resize(tiff.shape)
    gt_padded = image_pad(gt,dims_info.pad_size)    
    inds = np.arange(gt.size)
    range_mask = dims_info.range_mask_get(tiff.shape)
    inds_f = inds[np.logical_and(gt > 0,range_mask).flatten()]
    inds_b = inds[np.logical_and(gt == 0,range_mask).flatten()]
    if len(inds_f) >0:
        num_segment_b = int(rate*num_segment)
        num_segment_f = num_segment - num_segment_b 
        inds_sel_f = np.random.choice(inds_f,num_segment_f)
        inds_sel_b = np.random.choice(inds_b,num_segment_b)
        #inds_sel = list(inds_sel_b) + list(inds_sel_f)
        segments = []
        for i in inds_sel_f:
            segments.append([sample_ind,i,1])
        for i in inds_sel_b:
            segments.append([sample_ind,i,0])
        #segments = [[sample_ind,i,1] for i in inds_sel] 
        return segments 
    else:
        num_segment_b = int(rate*num_segment)
        segments = [[sample_ind,i,0] for i in inds_sel_b] 
        return segments 
def equally_sample_single(sample_ind,sample_info,num_segment = 1000,dims_info =None,rate = 0.1):
    tiff = io.imread(sample_info[0])
    gt = io.imread(sample_info[-1])
    gt.resize(tiff.shape)
    gt_padded = image_pad(gt,dims_info.pad_size)

    inds = np.arange(gt.size)

    range_mask = dims_info.range_mask_get(tiff.shape)
    inds_f = inds[np.logical_and(gt > 0,range_mask).flatten()]
    inds_b = inds[np.logical_and(gt == 0,range_mask).flatten()]
    
    
    if len(inds_f) >0:

        num_segment_b = int(rate*num_segment)
        num_segment_f = num_segment - num_segment_b 
        #print("Now samples {} segments from {} segments".format(num_segment,len(inds_f)))
        inds_sel_f = np.random.choice(inds_f,num_segment_f)
        inds_sel_b = np.random.choice(inds_b,num_segment_b)
        segments = []
        for i in inds_sel_f:
            segments.append([sample_ind,i,1])
        for i in inds_sel_b:
            segments.append([sample_ind,i,0])
        return segments 
    else:
        num_segment_b = int(rate*num_segment)
        segments = [[sample_ind,i,0] for i in inds_sel_b] 
        return segments 
    
    
    #inds_sel = list(inds_f) + list(inds_b)
    #inds_sel = list(inds_f)

    #temp = [content_cal(i,tiff.shape,gt_padded,sample_ind,dims_info) for i in inds_sel]
    
    #segments = np.array([x[:3] for x in temp])
    
    #beta = round(np.nanmean(np.array([x[-1] for x in temp])),4)
    
    #np.random.shuffle(segments)
def equally_sample(samples_ind,samples_info,dims_info,nums_segment = 10000,rate = 0.1):
    nums_sample = len(samples_ind)
    nums_segment_per_sample = np.zeros(nums_sample).astype("int")
    nums_segment_per_sample.fill(nums_segment // nums_sample)
    for i in np.random.choice(np.arange(nums_sample),nums_segment % nums_sample):
        nums_segment_per_sample[i] += 1
    
    segments = []
    segments_list = []
    cpu_nums = os.cpu_count()//2

    with Pool(cpu_nums) as pool:
        segments_list = pool.starmap(partial(equally_sample_single,dims_info = dims_info,rate = rate)\
                ,zip(samples_ind,samples_info,nums_segment_per_sample))
     
    segments = [y for x in segments_list for y in x]
    #beta = round(np.nanmean(np.array([x[-1] for x in segments_list])),4)
    
    #print("equally sample finished ! Now has sampled {} segments. The beta is {}"\
    #        .format(len(segments),beta))
    print("equally sample finished ! Now has sampled {} segments."\
            .format(len(segments)))
    
    return np.array(segments),segments_list   

def grid_sample_single(sample_ind,sample_info,dims_info= None):
    tiff = io.imread(sample_info[0])
    gt = io.imread(sample_info[-1])
    gt.resize(tiff.shape)
    
    roi_mask = tiff >0
    range_mask = dims_info.range_mask_get(tiff.shape)
    grid_mask = dims_info.grid_mask_get(tiff.shape)
    grid_mask = np.logical_and(range_mask,grid_mask) 
    mask = np.logical_and(grid_mask,roi_mask)

    inds = np.arange(tiff.size)
    inds_sel = inds[mask.flatten()]
    segments = [[sample_ind,x,0] for x in inds_sel]
    
    beta = np.sum(gt).astype(np.float)/gt.size
    
    return segments,beta 
    #segments += [[i,x,0] for x in inds_sel]
    #segments_list.append([[i,x,0] for x in inds_sel])
def grid_sample(samples_ind,samples_info,dims_info):
    segments = []
    segments_list = []
    cpu_nums = os.cpu_count()//4
    #print(cpu_nums,samples_ind,samples_info)
    with Pool(cpu_nums) as pool:
        segments_list = pool.starmap(partial(grid_sample_single,dims_info = dims_info)\
                ,zip(samples_ind,samples_info))
    segments = [y for x in segments_list for y in x[0] ]
    beta = round(np.mean(np.array([x[-1] for x in segments_list])),4)
    print("grid sample finished ! Now has {} segments. The Beta is {}".format(len(segments),beta))
    return segments,segments_list,beta 



if __name__ == "__main__":
    samples_info_json = "/home/jjx/Biology/DataNeuron/data/info.json"
    samples_info = read_json(samples_info_json)["imgs"][:3]
    gts_map = ["negative","postive"]
    dims_info = DimsInfo(133,80,40,1,gts_map)
    '''for i,sample_info in enumerate(samples_info):
        segments = coords_filter_single(i,sample_info,dims_info)
        segments_array = np.array(segments)
        print(segments_array.shape)
        hists,edges = np.histogram(segments_array[:,2],bins = 5)
        print(hists,edges)'''

    segments  = rate_sample_single(0,samples_info[0],1000,dims_info)
    print(len(segments))
    
    segments,_ = equally_sample([0,1,2],samples_info,dims_info)
    print(len(segments),segments.shape)
    segments,_,beta = grid_sample([0,1,2],samples_info,dims_info)
    print(len(segments),beta)

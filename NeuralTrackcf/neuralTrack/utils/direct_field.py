import numpy as np
import os

from scipy import ndimage
from multiprocessing import Pool
from functools import partial

def distance_transform(binary_metric):
    binary_metric_reverse = np.logical_not(binary_metric)
    edt, inds = ndimage.distance_transform_edt(binary_metric_reverse, return_indices=True)
    return edt,inds
def diret_field_cal(edt,inds,thres = 3):
    shapes = edt.shape
    sizes = edt.size
    coords = np.array(np.unravel_index(np.arange(sizes),dims = shapes))
    #print(coords.shape)
    direct_field = inds.reshape(inds.shape[0],-1) - coords
    direct_field = direct_field.transpose()
    #direct_field = direct_field.reshape(*shapes,-1)

    #print(direct_field)
    edt = edt.flatten()
    mask_gt_thres = edt >thres 
    mask_lt_thres = np.logical_and(edt <= thres,edt >0)
    direct_field[mask_gt_thres] = 0
    #print(shapes) 
    #gts = np.zeros(sizes)
    #gts[edt == 0] = 1
    #gts[mask_lt_thres] = 1
    #print(mask_lt_thres)
    #gts = gts.reshape(shapes)

    direct_field[mask_lt_thres] = direct_field[mask_lt_thres] / edt[mask_lt_thres][:,None]
    
    direct_field = direct_field.reshape(-1,direct_field.shape[-1]).transpose()
    #direct_field = direct_field.reshape()

    direct_field = direct_field.reshape(-1,*shapes)
    #direct_field = np.concatenate((direct_field,gts[None,:]),axis = 0)
    return direct_field
def batch_direct_field_cal(batch_gts,thres = 3):
    cpu_nums = os.cpu_count()//12
    
    with Pool(cpu_nums) as pool:
        infos = pool.starmap(distance_transform,zip(batch_gts))
    #print(infos)
    batch_edt,batch_inds = zip(*infos)
    with Pool(cpu_nums) as pool:
        batch_direct_field = pool.starmap(partial(diret_field_cal,thres = thres),\
                zip(batch_edt,batch_inds))
    #batch_direct_field = np.concatenate(batch_direct_field,axis = 0)
    batch_direct_field = np.array(batch_direct_field,dtype = np.float)
    return batch_direct_field 

def batch_direct_field_cal(batch_gts,thres = 10):
    batch_direct_field = []
    for batch_gt in batch_gts:
        edt,inds = distance_transform(batch_gt)
        direct_field = diret_field_cal(edt,inds,thres)
        batch_direct_field.append(direct_field)
    batch_direct_field = np.array(batch_direct_field)
    return batch_direct_field 
if __name__ == "__main__":
    import torch
    a = np.zeros((20,64,64,64))
    a[1,1,1] = 1
    #edt,inds = distance_transform(a)
    #print(inds.shape)
    #print(edt)
    #direct_field = diret_field_cal(edt,inds,1)
    a = torch.tensor(a)
    direct_field = batch_direct_field_cal(a,1)
    #print(direct_field)
    print(direct_field.shape)

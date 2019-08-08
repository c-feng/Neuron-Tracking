from skimage.measure import label
from skimage.util import pad
import numpy as np

def ins_center_crop(seg,center_coord,crop_size = [32,32,32]):
    center_coord_array =  np.array(center_coord)
    crop_size_array = np.array(crop_size)

    crop_size_array_l = [ x//2 if x%2 == 0 else x//2 + 1 for x in crop_size ]
    crop_size_array_r = [ x//2 for x in crop_size ]

    coord_l = np.clip(center_coord_array - crop_size_array_l , 0, None)
    coord_r = np.clip(center_coord_array + crop_size_array_r , None, np.array(seg.shape))
    
    #crop_patch = seg[coord_l[0]:coord_r[0],coord_l[1]:coord_r[1],coord_l[2]:coord_r[2]]  
    #crop_patch_label = label(crop_patch)

    #seg_label = label(seg)
    #labels_ = np.unique(seg_label)[1:]
    #label_ = seg_label[center_coord[0],center_coord[1],center_coord[2]]

    crop_patch = seg[coord_l[0]:coord_r[0],coord_l[1]:coord_r[1],coord_l[2]:coord_r[2]]  
    crop_patch_label = label(crop_patch)
    labels_ = np.unique(crop_patch_label)[1:]
    label_ = crop_patch_label[center_coord[0] - coord_l[0],\
            center_coord[1] - coord_l[1], center_coord[2] - coord_l[2]]

    print(labels_,label_) 
    if label_ == 0:
        return np.zeros(crop_size,dtype = np.uint16)

    mask = crop_patch_label == label_

    filter_patch = np.zeros_like(crop_patch_label)
    filter_patch[mask] = 1

    pad_shape_r = crop_size_array - np.array(filter_patch.shape)
    pad_shape_l = [0,0,0]
    
    pad_shape = list(zip(pad_shape_l,pad_shape_r))
    #print(pad_shape)
    filter_patch_pad = pad(filter_patch,pad_shape,"constant",constant_values = 0)
    return filter_patch_pad

    

    
def seg_center_crop(seg,center_coord,crop_size = [32,32,32]):
    crop_size_array = np.array(crop_size)
    center_coord_array =  np.array(center_coord)
    
    crop_size_array_l = [ x//2 if x%2 == 0 else x//2 + 1 for x in crop_size ]
    crop_size_array_r = [ x//2 for x in crop_size ]

    coord_l = np.clip(center_coord_array - crop_size_array_l , 0, None)
    coord_r = np.clip(center_coord_array + crop_size_array_r , None, np.array(seg.shape))

    crop_patch = seg[coord_l[0]:coord_r[0],coord_l[1]:coord_r[1],coord_l[2]:coord_r[2]]  

    pad_shape_r = crop_size_array - np.array(crop_patch.shape)
    pad_shape_l = [0,0,0]
    
    pad_shape = list(zip(pad_shape_l,pad_shape_r))
    crop_patch_pad = pad(crop_patch,pad_shape,"constant")
    return crop_patch_pad

    

    

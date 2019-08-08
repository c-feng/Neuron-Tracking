import numpy as np
from skimage.morphology import dilation,ball,cube

def range_mask_generate(crop_size, patch_size):
    mask = np.zeros(patch_size, dtype = bool)
    crop_size_array = np.array(crop_size)
    patch_size_array = np.array(patch_size)
    
    range_l = crop_size_array//2
    range_r = crop_size_array - (crop_size_array - crop_size_array//2)

    mask[range_l[0]:-range_r[0], range_l[1]:-range_r[1], range_l[2]:-range_r[2]] = 1
    return mask

def dilated_mask_generate(coord, shape, mode, thres):
    mask = np.zeros(shape, dtype = bool)
    if len(shape) ==3:
        x,y,z = coord 
        mask[x, y, z] = 1
    else:
        x,y = coord
        mask[x, y] = 1
    mask = dilation(mask, mode(*thres))
    return mask


def batch_dilated_mask_generate(coords, shape, mode, thres):
    masks = []
    for coord in coords:
        mask = dilated_mask_generate(coord, shape, mode, *thres)
        masks.append(mask[None])
    masks = np.concatenate(masks, axis = 0)
    return masks
    


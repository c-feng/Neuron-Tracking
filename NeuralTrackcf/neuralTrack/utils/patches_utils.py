import torch
import torch.nn.functional as F
import numpy as np

def patches_split(img,patches_size,patches_stride,pad_mode = "constant"):
    patches_size_array = np.array(patches_size, dtype = int)
    patches_stride_array  = np.array(patches_stride, dtype = int)
    img_size_array = np.array(img.shape,dtype = int)

    
    patches_max_ind = []
    for img_size_,patches_size_,patches_stride_ in zip(img_size_array,patches_size_array,patches_stride_array):
        max_ind = abs(img_size_ - patches_size_) // patches_stride_\
                if abs(img_size_ - patches_size_) % patches_stride_ == 0\
                else abs(img_size_ - patches_size_) // patches_stride_ + 1
        patches_max_ind.append(max_ind)
    patches_max_ind_array = np.array(patches_max_ind)

    pad_r_array = patches_stride_array * patches_max_ind_array + patches_size_array - img_size_array 
    #print(pad_r_array)
    pad_r_array = pad_r_array.tolist()
    pad = [0,pad_r_array[2],0,pad_r_array[1],0,pad_r_array[0]]

    img_pad = F.pad(img,pad,mode = pad_mode)

    img_patches_grid = img_pad.unfold(0,patches_size[0],patches_stride[0]).\
                unfold(1,patches_size[1],patches_stride[1]).\
                unfold(2,patches_size[2],patches_stride[2])
    
    grid_size = img_patches_grid.size()[:3]

    img_patches = img_patches_grid.reshape(-1,*patches_size)
    #print(img_patches) 
    coords_grid = np.unravel_index(np.arange(img_patches.size(0)),grid_size)
    coords_patches = np.array(coords_grid) * patches_stride_array[:,None]# 3 * n
    coords_patches = coords_patches.transpose()# n * 3

    return img_patches, coords_patches, grid_size

def patches_merge(img_patches,patches_size,patches_stride,coords_patches,mode = "and"):
    patches_size_array = np.array(patches_size, dtype = int)
    patches_stride_array  = np.array(patches_stride, dtype = int)

    img_pad_size = np.array(coords_patches[-1]) + patches_size_array 
    #print(img_pad_size)
    mask_filled = torch.zeros(*img_pad_size.tolist(),dtype = torch.uint8)
    img_pad = torch.zeros(*img_pad_size.tolist(), dtype = img_patches.dtype)

    for coords_patch ,img_patch in zip(coords_patches, img_patches):
        x_l, y_l, z_l = coords_patch 
        x_r, y_r, z_r = coords_patch + patches_size_array

        if mode == "and":
            mask_patch_filled = mask_filled[x_l:x_r,y_l:y_r,z_l:z_r]
            
            img_pad[x_l:x_r,y_l:y_r,z_l:z_r] += img_patch
            mask_filled_not_changed = img_pad[x_l:x_r,y_l:y_r,z_l:z_r] == 2

            img_pad[x_l:x_r,y_l:y_r,z_l:z_r][mask_patch_filled] = 0
            img_pad[x_l:x_r,y_l:y_r,z_l:z_r][mask_filled_not_changed] = 1

        elif mode == "or":
            img_pad[x_l:x_r,y_l:y_r,z_l:z_r][img_patch > 0] = img_patch[img_patch > 0]

        mask_filled[x_l:x_r,y_l:y_r,z_l:z_r] = 1

    return img_pad

if __name__ == "__main__":
    img = torch.zeros(8,8,8)
    img[2:4,3:4,1:4] = 1
    patches_size = [3,3,3]
    patches_stride = [2,2,2]
    img_patches,coords_patches = patches_split(img,patches_size,patches_stride)
    print(img_patches.size())
    img_pad = patches_merge(img_patches,patches_size,patches_stride,img.size(),coords_patches,mode = "and") 
    print(img_pad.size())
    print(torch.sum(img != img_pad[:8,:8,:8]))

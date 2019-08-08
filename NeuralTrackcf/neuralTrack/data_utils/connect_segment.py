from skimage.external import tifffile
from skimage.measure import label,regionprops
from skimage.morphology import skeletonize_3d,dilation,ball
import numpy as np
import os.path as osp
import scipy.misc
from glob import glob

from neuralTrack.utils.osutils import mkdir_if_missing

colors = np.array([0,2000],np.uint16)

def range_mask_generate(coord,shape,thres = 16):
    x,y,z = coord
    x_r = x + thres if x + thres <= shape[0] else shape[0]
    x_l = x - thres if x - thres >=0 else 0

    y_r = y + thres if y + thres <= shape[1] else shape[1]
    y_l = y - thres if y - thres >=0 else 0

    z_r = z + thres if z + thres <= shape[2] else shape[2]
    z_l = z - thres if z - thres >=0 else 0
    
    mask = np.zeros(shape,dtype = bool)
    mask[x_l:x_r,y_l:y_r,z_l:z_r] = 1
    return mask

def ins_seg(img_path,junc_path,target_dir,flag_visual = False):
    
    fname = osp.splitext(osp.basename(img_path))[0]
    fname_dir = osp.join(target_dir,fname) 
    
    mkdir_if_missing(fname_dir)
    
    fname_tiff_dir = osp.join(fname_dir,"tiffs")
    fname_gt_dir = osp.join(fname_dir,"gts")

    if flag_visual:
        fname_tiff_visual_dir = osp.join(fname_dir,"tiffs_visual")
        fname_gt_visual_dir = osp.join(fname_dir,"gts_visual")
        mkdir_if_missing(fname_tiff_visual_dir)
        mkdir_if_missing(fname_gt_visual_dir)
    
    mkdir_if_missing(fname_tiff_dir)
    mkdir_if_missing(fname_gt_dir)


    img = tifffile.imread(img_path)
    img = (img > 1)
    junc = tifffile.imread(junc_path)

    ins = label(img,connectivity=1)
    ins_inds = np.unique(ins)[1:]
    ins_visual = []
    
    inds = np.arange(img.size) 

    for i in ins_inds:
        ins_i = ins == i
        if np.sum(ins_i) < 8000:
            continue
        print(i,np.sum(ins_i))
        ins_i = ins_i.astype(np.uint8)
        gt_i = np.zeros_like(ins_i)

        junc_mask = np.logical_and(ins_i,junc)
        if np.sum(junc_mask) == 0:continue
        inds_sel = inds[junc_mask.flatten()]
        mask = np.zeros_like(junc_mask)
        
        for ind in inds_sel:
            coord = np.unravel_index(ind,img.shape)
            mask += range_mask_generate(coord,img.shape)
        
        mask = np.logical_and(mask,ins_i) 

        fname_i = "{}_{}.tif".format(fname,i)
        gt_i[mask] = 1
        
        if flag_visual:
            tifffile.imsave(osp.join(fname_tiff_visual_dir,fname_i),colors[ins_i])
            tifffile.imsave(osp.join(fname_gt_visual_dir,fname_i),colors[gt_i])

        tifffile.imsave(osp.join(fname_tiff_dir,fname_i),ins_i)
        tifffile.imsave(osp.join(fname_gt_dir,fname_i),gt_i)

        ins_i = np.sum(ins_i,axis = 0)>0
        ins_i = ins_i.astype(np.int)
    
        ins_visual.append(colors[ins_i])
    if len(ins_visual) == 0:return 
    fname_x = "{}_x.jpg".format(fname)
    ins_visual = np.hstack(ins_visual)
    scipy.misc.imsave(osp.join(fname_dir,fname_x),ins_visual)
   


target_dir = "/home/jjx/Biology/DirectField/data_300_modified_ins"
mkdir_if_missing(target_dir)

img_paths = glob("/home/jjx/Biology/DirectField/data_300_modified_VOX/visual_results/*/*_pred.tif")
junc_paths = glob("/home/jjx/Biology/DirectField/data_300_modified_junc/centers/*")
img_paths.sort()
junc_paths.sort()


for img_path,junc_path in zip(img_paths,junc_paths):
    print(img_path)
    print(junc_path)
    ins_seg(img_path,junc_path,target_dir,True)
    #break

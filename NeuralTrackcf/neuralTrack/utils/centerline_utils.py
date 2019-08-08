import numpy as np
import os.path as osp
import math
from functools import partial

from collections import Counter
from skimage.external import tifffile
from glob import glob

from .serialization import  read_json,write_json
from .osutils import mkdir_if_missing
from .direct_field import distance_transform,diret_field_cal
from .coords_utils import endpoint_cal,coords_trans
from .mask_utils import range_mask_generate


def line_segment(coords,end_coords):
    if len(end_coords) == 0:
        return []
    if len(end_coords) == 2:
        return [coords]

    lines = []
    line = []
    #print(end_coords,len(coords))
    for coord in coords:
        
        if coord in end_coords:
            if len(line) == 0:
                continue
            else:
                lines.append(line)
                line = []
        else:
            line.append(coord)    
    return lines   

def ins_mask_cal(coords,ins):
    coords_array = np.array(coords)
    labels_ = ins[coords_array[:,0],coords_array[:,1],coords_array[:,2]]
    labels_ = labels_.tolist()
    if 0 in labels_:
        labels_.remove(0)

    most_common,num_most_common = Counter(labels_).most_common(1)[0]  
    
    mask = ins == most_common
    return mask

def center_radius_cal(coord,ins_mask,thres):
    inds = np.arange(ins_mask.size)
    
    mask = range_mask_generate(coord,ins_mask.shape,thres)
    
    volume = np.sum(ins_mask[mask])
    volume_rate = volume / np.sum(mask)

    radius = math.pow(3 * volume / (4 * math.pi), 1.0 / 3)
    radius = int(radius)
    xs,ys,zs = np.unravel_index(inds[mask.flatten()], mask.shape)
    x,y,z = int(xs.mean()),int(ys.mean()),int(zs.mean())

    print(thres,np.sum(mask),volume_rate,radius)
    return [x,y,z],radius

"""def connect_2_points(p1=[34,50,178], p2=[100,100,106], size=[301,301,301]):
    """ Connect two points in the spaces.
        Filling the gap between two points
    """

    mask = np.zeros(shape=size)

    vector = np.array(p2) - np.array(p1)
    length = np.linalg.norm(vector)

    if length <= 1: return mask

    vector_unit = vector / length

    mask[p2[0],p2[1],p2[2]] = 1 
    
    for i in range(int(round(length)) + 1):
        x, y, z = np.array(p1) + vector_unit * i
        
        x_i = int(round(x))
        y_i = int(round(y))
        z_i = int(round(z))
        
        print("p0 {} {} {} p1 {} {} {} new {} {} {}".format(*p1, *p2, x_i, y_i, z_i))
        mask[x_i, y_i, z_i] = 1  

    return mask"""

def connect_2_points(p1, p2, size):
    mask = np.zeros(size, dtype = bool)
    p_list = cut_2_points(p1, p2)
    p_array = np.array(p_list, dtype = int)
    mask[p_array[:,0], p_array[:,1], p_array[:,2]] = 1
    return mask

def cut_2_points(p1=[34,50,178], p2=[100,100,106]):
    

    p_list = []
    p1 = np.array(p1)
    p2 = np.array(p2)
    p_list = [p1, p2]
    length = np.linalg.norm(p2 - p1)
    if length <= 1:
        return p_list
    pc = (p1 + p2)/2
    #print(p1, pc, p2)

    p_list += cut_2_points(p1, pc)
    p_list += cut_2_points(pc, p2)
    
    return p_list

def centerline_cal(swc, ins, factors, target_size, thres, unit):
    end_coords,coords = endpoint_cal(swc,factors,target_size,unit)

    end_coords = end_coords.tolist()
    coords = coords.tolist()

    centerline_mask = np.zeros_like(ins)
    radius_mask = np.zeros_like(ins)

    print(end_coords)
    print(len(coords))
    if len(coords) <= 10:
        #final_mask = np.concatenate([centerline_mask[None],radius_mask[None]],axis = 0)
        #return final_mask
        return centerline_mask,radius_mask
    lines = line_segment(coords, end_coords)
    ins_mask = ins_mask_cal(coords, ins) 

    if np.sum(ins_mask) == 0 :
        return centerline_mask,radius_mask

    for line in lines:
        centers = []
        radius = []

        '''for coord in line:
            center, radiu = center_radius_cal(coord, ins_mask, thres)
            centers.append(center)
            radius.append(radiu)
    
        for p1,p2,r1 in zip(centers[:-1], centers[1:], radius[:-1]):
            print(p1,p2)
            mask = connect_2_points(p1,p2,ins.shape)
            print(np.sum(mask))
            centerline_mask[mask > 0] = 1
            radius_mask[mask > 0] = r1'''
        for p1,p2 in zip(line[:-1],line[1:]):
            c1 = p1
            c2 = p2
            r1 = 2
            r2 = 2
            mask = connect_2_points(c1,c2,ins.shape)
            centerline_mask[mask > 0] = 1
            radius_mask[mask > 0] = (r1 + r2)//2


    return centerline_mask,radius_mask

def single_run(swcs,ins_p,factors,target_size,thres = 3,unit = (0.2,0.2,1)):
    ins = tifffile.imread(ins_p)
    x_, y_, z_ = ins.shape

    centerline = np.zeros_like(ins)
    radius = np.zeros_like(ins)

    for swc in swcs :
        centerline_mask, radius_mask = centerline_cal(swc, ins, factors, target_size, thres, unit) 
        #final_mask += centerline_cal(swc, ins, factors, target_size, thres, unit) 
        mask_ = centerline_mask > 0
        centerline[mask_] = centerline_mask[mask_]
        radius[mask_] = radius_mask[mask_]

    return centerline, radius 


if __name__ == "__main__":
    swcs = glob("/home/jjx/Biology/Modified_Selected_Dataset/swcs/5700_35350_4150*")
    #ins_p = glob("/home/jjx/Biology/DirectField/data_300_modified/ins_modified/5700_35350_4150*")
    ins_p = glob("/home/jjx/Biology/DirectField/data_300_modified_junc/ins/5700_35350_4150*")
    factors = np.array([301,301,301])/300
    centerline, radius = single_run(swcs,ins_p,factors,300)

    colors = np.array([0,500],np.uint16)
    tifffile.imsave("centerline.tif",colors[centerline])

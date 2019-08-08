import numpy as np
import os.path as osp
import math
from functools import partial

from collections import Counter
from skimage.external import tifffile
from glob import glob

from neuralTrack.utils.serialization import  read_json,write_json
from neuralTrack.utils.osutils import mkdir_if_missing
from neuralTrack.utils.direct_field import distance_transform,diret_field_cal
from neuralTrack.utils.coords_trans import coords_trans
from neuralTrack.utils.ins_modified import endpoint_cal
from neuralTrack.utils.junc_generate import range_mask_generate


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

def connect_2_points(p1=[34,50,178], p2=[100,100,106], size=[300,300,300]):
    """ Connect two points in the spaces.
        Filling the gap between two points
    """

    mask = np.zeros(shape=size)
    direction_v = np.array( [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]] )
    direction = direction_v / ( np.power(np.power(direction_v, 2).sum(), 0.5) )
    length = int(np.power(np.power(direction_v, 2).sum(), 0.5)) 

    mask[p2[0],p2[1],p2[2]] = 1 
    #print(length)
    
    for i in range(length):
        x_new = int(direction[0] * i) + p1[0]
        y_new = int(direction[1] * i) + p1[1]
        z_new = int(direction[2] * i) + p1[2]
        if length > 2:
            print(x_new,y_new,z_new)
        mask[x_new, y_new, z_new] = 1  # 200

    # debug
    # mask[int(p1[0]), int(p1[1]), int(p1[2])] = 2000
    # mask[int(p2[0]), int(p2[1]), int(p2[2])] = 2000
    # tifffile.imsave("test.tif", mask.astype(np.int16))
    return mask

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
            #c1,r1 = center_radius_cal(p1,ins_mask,thres)
            #c2,r2 = center_radius_cal(p2,ins_mask,thres)
            #print(c1,c2)
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

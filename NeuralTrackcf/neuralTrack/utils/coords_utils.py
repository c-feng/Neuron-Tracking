import numpy as np
from treelib import Node, Tree
import sys

def connect_2_points(p1, p2, size):
    """
        fill 2 points with recursion
    """
    mask = np.zeros(size, dtype = bool)
    p_list = cut_2_points(p1, p2)
    p_array = np.array(p_list, dtype = int)
    #filter coords out of size
    p_mask = coords_filter(p_array, size)
    p_array = p_array[p_mask]
    
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


def coords_filted_get(swc, unit, sep, target_size):
    coords, labels, ids, pars = coords_get(swc, unit, sep)

    mask = coords_filter(coords, target_size)
    coords = coords[mask]
    labels = labels[mask]
    ids = ids[mask]
    pars = pars[mask]

    return coords, labels, ids, pars


def coords_get(swc, units, sep):
    """
        get the coords from a swc file
        
        swc: swc path
        units: unit of coords exp:[1, 1, 1]
        sep: sep used in swc file exp: ","
    
    """
    with open(swc) as f:
        infos = np.array([np.fromstring(line.strip(),dtype = float,sep = sep) for line in f.readlines()])
    coordnates = infos[:,2:5]
    #print(swc)
    coordnates = coordnates/np.array(units)
    coordnates = coordnates.astype(int)
    coordnates = coordnates[:,[2,1,0]]

    labels = infos[:,5]
    labels = labels.astype(int)
    
    ids = infos[:, 0].astype(int)
    pars = infos[:, 6].astype(int)

    labels += 1

    return coordnates, labels, ids, pars

def coords_trans(swc_path, factors, target_size, units, sep):
    coords, labels, ids, pars = coords_get(swc_path,units, sep)
    coords = np.array(coords)
    labels = np.array(labels)
    ids = np.array(ids)
    pars = np.array(pars)
    
    coords = coords/factors
    coords = coords.astype(int)

    return coords, labels, ids, pars

def coords_filter(coords, target_size):
    """
        filter coords out of target_size
    """
    mask_z = np.logical_and(coords[:,0]>=0,coords[:,0]<target_size[2])
    mask_y = np.logical_and(coords[:,1]>=0,coords[:,1]<target_size[1])
    mask_x = np.logical_and(coords[:,2]>=0,coords[:,2]<target_size[0])
    mask = np.logical_and(mask_x,mask_y)
    mask = np.logical_and(mask,mask_z)

    return mask


def endpoint_cal(swc_p, unit , sep = ","):
    """
        generate a multiBranch Tree from the swc file

    """
    print(unit, sep)
    coords, labels, ids, pars = coords_get(swc_p, unit, sep)
    #coords += 1
    if len(coords) == 0:
        print("{} is something wrong".format(swc_p))
        sys.exit(0)

    ftree = Tree()
    ftree.create_node(ids[0], ids[0], data = coords[0])

    for coord_, id_, par_ in zip(coords[1:], ids[1:], pars[1:]):
        #print(id_, par_)
        ftree.create_node(id_, id_, data = coord_, parent = par_)

    endpoint_coords = [x.data for x in ftree.leaves()]
    endpoint_coords.append(coords[0])

    branch_coords = [x.data for x in ftree.all_nodes() if len(ftree.children(x.tag)) >= 2]
    
    endpoint_coords = np.array(endpoint_coords)
    branch_coords = np.array(branch_coords)
    coords = np.array(coords)
    return endpoint_coords, branch_coords, coords, ftree

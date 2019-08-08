import numpy as np
import os.path as osp

from scipy.sparse import coo_matrix

def array_to_coo(coo_array):
    size = coo_array.shape
    size_str = ' '.join(str(x) for x in size)
    
    mask = coo_array > 0
    coo_inds = np.arange(coo_array.size)[mask.flatten()]
    coo_coords = np.unravel_index(coo_inds, size)
    coo_data = coo_array[mask]

    coo_str = ' '.join("{} {} {} {}".format(x, y, z, v) \
            for x, y, z, v in zip(*coo_coords, coo_data))
    
    return coo_str, size_str
    #with open(coo_p, "w+") as f:
    #    f.write(size_str + "\n" + coo_str)

def coo_to_array(coo_str, size_str):
    #assert osp.isfile(coo_p)
    #with open(coo_p) as f:
    #    size_str, coo_str = f.read().split("\n")

    size = np.fromarrays(size_str, dtype = int, sep = ' ')
    coo_flatten = np.fromarrays(coo_str, dtype = int, sep = ' ')
    coo_infos = coo_flatten.reshape(-1, 4)
    coo_array = np.zeros(size, dtype = np.int)
    coo_array[coo_infos[:, 0], coo_infos[:, 1], coo_infos[:, 2]] = coo_infos[:, 3]
    #coo = coo_matrix((coo_infos[:, -1], coo_infos[:, 0], \
    #        coo_infos[:, 1], coo_infos[:, 2]), shape = size)
    return coo_array




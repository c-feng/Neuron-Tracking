import numpy as np
import torch
import skimage
import kimimaro

from treelib import Node, Tree
from skimage.morphology import dilation, ball
from skimage.external import tifffile



labels = tifffile.imread("../ins_pred.tif")

def fov_connect(fov_ins_array):
    def parent(edges, i):
        coords = np.where( edges == i )
        edge = edges[ coords[0][0] ]
        if edge[0] == i:
            return edge[1] + 1
        return edge[0] + 1
     
    skels = kimimaro.skeletonize(
        fov_ins_array, 
        teasar_params={
            'scale': 4,
            'const': 500, # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100, # physical units
            'soma_acceptance_threshold': 3500, # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300, # physical units
            'max_paths': None, # default None
                },
            dust_threshold=50,
            anisotropy=(200,200,1000), # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=True, # default False
            parallel=2, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )
    ends_dict = {}

    fov_ins_skel_array = np.zeros_like(fov_ins_array)
    ends_array = np.zeros_like(fov_ins_array)
    for label_ in skels:
        skel = skels[label_]

        coords = (skel.vertices / np.array([200, 200, 1000])).astype(int)
        fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label_

        coords = coords.tolist()
        edges = skel.edges.tolist()

        ftree = Tree()
        cur_ = edges[0][0]
        ftree.create_node(cur_, cur_, data = coords[0])

        cur_list = [cur_]

        while(len(edges) > 0 and len(cur_list) > 0):
            _cur_list = []
            edges_ = edges[:]
            #print(cur_list)
            for cur_ in cur_list:
                next_inds = np.where(np.array(edges_) == cur_)[0]
                if len(next_inds) == 0:continue
                for next_ind in next_inds:
                    edge_ = edges_[next_ind]
                    edges.remove(edge_)
                    #print(cur_, edge_)

                    if edge_[0] == cur_:
                        next_ = edge_[-1]
                    else:
                        next_ = edge_[0]

                    _cur_list.append(next_)
                    ftree.create_node(next_, next_, data = coords[next_], parent = cur_)
                edges_ = edges[:]

            cur_list = _cur_list

        ends = [x.data for x in ftree.leaves()]
        ends.append(coords[0])

        ends_dict[label_] = ends
        
        ends_ = np.array(ends)
        ends_array[ends_[:, 0], ends_[:, 1], ends_[:, 2]] = 1
        #ends_array = dilation(ends_array, ball(1))

    return fov_ins_skel_array, ends_array, ends_dict 

fov_ins_skel_array, ends_array, ends_dict = fov_connect(labels)
print(ends_dict)
tifffile.imsave("skel.tif", fov_ins_skel_array)
tifffile.imsave("ends.tif", ends_array)



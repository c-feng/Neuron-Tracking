import kimimaro
import numpy as np
from skimage.external import tifffile


def ends_cal(skel, anisotropy=(200,200,1000)):
    edges = skel.edges
    coords = (skel.vertices / np.array([200, 200, 1000])).astype(int)
    
    ends = []
    vecs = []
    for edge  in edges:
        l, r = edge
        if np.sum(edges == l) == 1:
            ends.append(coords[l])
            vec = coords[l] - coords[r]
            vecs.append(vec)
        elif np.sum(edges == r) == 1:
            ends.append(coords[r])
            vec = coords[r] - coords[l]
            vecs.append(vec)
        else:
            pass
    ends = np.array(ends)
    vecs = np.array(vecs)
    return ends, vecs

def skels_cal(fov_ins_array, nums_cpu=1, anisotropy=(200,200,1000)):
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
            dust_threshold=0,
            anisotropy=anisotropy, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            progress=False, # default False
            parallel=nums_cpu, # <= 0 all cpu, 1 single process, 2+ multiprocess
            )

    fov_ins_skel_array = np.zeros_like(fov_ins_array)
    fov_ins_ends_array = np.zeros_like(fov_ins_array)
    ends_dict = {}
    vecs_dict = {}
    for label in skels:
        skel = skels[label]
        ends, vecs = ends_cal(skel, anisotropy)
        # ends, vecs = find_ends_angles_cross(skel, anisotropy)

        coords = (skel.vertices / np.array(anisotropy)).astype(int)
        
        fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label
        fov_ins_ends_array[ends[:,0], ends[:,1], ends[:,2]] = label
        ends_dict[label] = ends
        vecs_dict[label] = vecs

    return ends_dict, vecs_dict, fov_ins_skel_array, fov_ins_ends_array, skels

def skel_ends(tif):

    labels_eps = {}

    labels = np.unique(tif)[1:]

    ends_dict, vecs_dict, skels_array, ends_array, skels = skels_cal(tif)
    # ins_skel_array, skel = skels_cal(ins_i)

    # ends, vecs = ends_cal(skel)
    # labels_eps[label] = [ends.tolist(), vecs.tolist()]
    for label in labels:
        labels_eps[label] = [ends_dict[label].tolist(), vecs_dict[label].tolist()]

    return labels_eps, ends_array, skels_array, skels

def find_ends_angles_cross(skel, anisotropy=(200,200,1000)):
    # skel = skels[label]
    vertices = (skel.vertices / anisotropy).astype(int)
    edges = skel.edges

    # 找出端点和交叉点
    num_points = len(vertices)
    end_points = []
    cross_points = []
    angles = []
    for i in range(num_points):
        cnt = 0
        for edge in edges:
            if i in edge:
                cnt += 1

        if cnt == 1:
            end_points.append(vertices[i].tolist()+[i])
            idx0, idx1 = np.where(edges==i)
            idy1 = 0 if idx1 else 1
            angle = vertices[edges[idx0, idy1]] - vertices[i]
            angles.append(angle)
        if cnt > 2:
            cross_points.append(vertices[i].tolist()+[i])

    if len(cross_points) > 0:
        sel = prune_skeleton(edges, end_points, cross_points)
    else:
        return np.array(end_points)[:, :-1], np.array(angles)

    end_points = np.array(end_points)[sel, :-1]
    angles = np.array(angles)[sel]

    return end_points, angles

def prune_skeleton(edges, end_points, cross_points, min_length=20):
    assert len(cross_points) > 0, "the cross_points is empty"
    reserve_ends = []
    # for cp in cross_points:
    cross_id = np.array(cross_points)[:, -1]

    for i, ep in enumerate(end_points):
        id_ = ep[-1]
        idx0, idx1 = np.where(edges==id_)
        idy1 = 0 if idx1 else 1
        length = 1
        while edges[idx0, idy1] not in cross_id:
            id_ = edges[idx0, idy1]
            idx0s, _ = np.where(edges==id_)

            for j in idx0s:
                if j != idx0:
                    idx0 = j
                    break

            idx1 = np.where(edges[idx0]==id_)[0]
            idy1 = 0 if idx1 else 1
            length += 1
        
        if length > min_length:
            reserve_ends.append(i)

    return reserve_ends


if __name__ == "__main__":
        
    tif = tifffile.imread("/media/fcheng/temp/ins_pred.tif")

    labels_eps, ends_array, skels_array, skels = skel_ends(tif)
    # skels的keys值, 对应skels_array和labels中的标签, skels_array是labels的骨架化
    anisotropy=np.array((200,200,1000))
    label = 5
    skel = skels[label]
    end_points, angle = find_ends_angles_cross(skel)

    # 保存end_points, cross_points
    ends_tif = np.zeros_like(tif)
    cross_tif = np.zeros_like(tif)
    for i in end_points:
        x, y, z = np.array(i, dtype=int)
        ends_tif[x, y, z] = 100

    # for i in cross_points:
    #     x, y, z = np.array(i[0], dtype=int)
    #     cross_tif[x, y, z] = 100

    print("")


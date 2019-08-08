import os
import numpy as np
import glob
import kimimaro
from skimage.external import tifffile
from skimage.morphology import dilation, cube

def kimimaro_func(fov_ins_array, nums_cpu=1, anisotropy=(200,200,1000)):
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

    return skels

def find_end_cross(vertices, edges, anisotropy=(200, 200, 1000)):
    """ 确定骨架点中的端点和分叉点
    """
    vertices_end_cross = []  # 0, 1endpoint, 2crosspoint
    vertices[:, :3] = (vertices[:, :3] / anisotropy).astype(int)
    # edges = skel.edges

    num_points = len(vertices)
    for i in range(num_points):
        cnt = 0
        for edge in edges:
            if i in edge:
                cnt += 1
        
        if cnt == 0:
            vertices_end_cross.append(vertices[i, :3].tolist()+[i, -1])  # -1需要去除的点
        elif cnt == 1:
            vertices_end_cross.append(vertices[i, :3].tolist()+[i, 1])
        elif cnt == 2:
            vertices_end_cross.append(vertices[i, :3].tolist()+[i, 0])
        elif cnt > 2:
            vertices_end_cross.append(vertices[i, :3].tolist()+[i, 2])

    return np.array(vertices_end_cross)

def prune_skeleton(edges, v_e_c, min_length=20):
    """ 将骨架点中的小分叉(length<min_length)进行裁剪
        edges: 原始的连接关系
        v_e_c: 坐标, 端点、分叉点信息

        return:
            edges_new: 小分叉的连接关系去除
            v_c_e_new: 坐标, 端点、分叉点信息, -1表示需要删除
    """

    end_sel = np.where(v_e_c[:, -1]==1)[0]
    cross_ids = np.where(v_e_c[:, -1]==2)[0]
    reserve_points = []
    del_points = []
    cross_change = []
    for s in end_sel:
        dps = []

        id_ = v_e_c[s, -2]
        idx0, idx1 = np.where(edges==id_)
        idx0, idx1 = idx0[0], idx1[0]
        idy1 = 0 if idx1 else 1
        length = 1
        dps.append(id_)
        while edges[idx0, idy1] not in cross_ids:
            id_ = edges[idx0, idy1]
            idx0s, _ = np.where(edges==id_)
            for j in idx0s:
                if j != idx0:
                    idx0_ = j
                    break
                else:
                    idx0_ = None
            # 生成骨架点时, 可能会断开, 造成虚假的端点
            if idx0_ == None:
                break
            else:
                idx0 = idx0_

            idx1 = np.where(edges[idx0]==id_)[0]
            idy1 = 0 if idx1 else 1
            length += 1
            dps.append(id_)
        
        if length > min_length:
            reserve_points += dps
        else:
            del_points += dps
            v_e_c_sel = np.where(v_e_c[:, -2]==edges[idx0, idy1])[0]
            cross_change.append(v_e_c_sel[0])

    if len(del_points) > 0:
        edges_new = []
        # v_e_c[cross_change, -1] = 0
        for edge in edges:
            if edge[0] not in del_points and edge[1] not in del_points:
                edges_new.append(edge)  # 已删除小分叉

        v_e_c_new = find_end_cross(v_e_c, edges_new, (1,1,1))  # 重新判断端点和交叉点
    else:
        return edges, v_e_c

    return np.array(edges_new), np.array(v_e_c_new)

def skeleon2direct(skel, tif_shape=(301, 301, 301), anisotropy=(200, 200, 1000), min_length=20):
    """ 由骨架点出发, 裁剪小分叉, 生成沿神经信号的方向向量, 
        input: 
            skel: 骨架化实例
        outputs:
            direct_mask: 方向向量, (tif_shape, 3)
            coords: 裁剪小分叉后的骨架点坐标
            ends: 裁剪小分叉后的端点坐标
    """

    v_e_c = find_end_cross(skel.vertices, skel.edges)
    if np.sum(v_e_c[:, -1]==2) > 0:
        edges_new, v_e_c_new = prune_skeleton(skel.edges, v_e_c, min_length)
    else:
        edges_new, v_e_c_new = skel.edges, v_e_c

    if np.sum(v_e_c_new[:, -1]==2) > 0:
        direct_mask = direct_cal(v_e_c_new, edges_new, tif_shape, min_length)
    else:
        # direct_mask = np.zeros([*tif_shape, 3])
        direct_mask = []
    
    coords = []
    ends = []
    cross = []
    for i in v_e_c_new:
        if i[-1] != -1:
            coords.append(i[:3])
        if i[-1] == 1:
            ends.append(i[:3])
        if i[-1] == 2:
            cross.append(i[:3])
    
    return np.array(direct_mask), np.array(coords, dtype=int), np.array(ends, dtype=int), np.array(cross, dtype=int)

def direct_cal(v_e_c, edges, tif_shape, min_length=20):

    # direct_mask = np.zeros([*tif_shape, 3])
    directs = []
    cross_id = np.where(v_e_c[:, -1]==2)[0]
    end_points = v_e_c[v_e_c[:, -1]==1]

    for ep in end_points:
        direct = []
        id_ = int(ep[-2])
        idx0, idx1 = np.where(edges==id_)
        idx0, idx1 = idx0[0], idx1[0]
        idy1 = 0 if idx1 else 1
        direct.append((v_e_c[edges[idx0, idy1], :3] - v_e_c[id_, :3]).tolist()
                       + v_e_c[id_, :3].tolist())

        while edges[idx0, idy1] not in cross_id:
            id_ = int(edges[idx0, idy1])
            idx0s, _ = np.where(edges==id_)
            for j in idx0s:
                if j != idx0:
                    idx0_ = j
                    break
                else:
                    idx0_ = None
            if idx0_ == None:
                break
            else:
                idx0 = idx0_
            idx1 = np.where(edges[idx0]==id_)[0]
            idy1 = 0 if idx1 else 1
            direct.append((v_e_c[edges[idx0, idy1], :3] - v_e_c[id_, :3]).tolist()
                           + v_e_c[id_, :3].tolist())
        
        direct = np.array(direct)
        direct_ = average_direct(direct[:, :3])
        direct[:, :3] = direct_
        directs += direct.tolist()
        # direct = np.array(direct, dtype=int)
        # for j in direct:
        #     direct_mask[j[3], j[4], j[5], ...] = j[:3]

    return directs

def average_direct(direct_list, moving_len=11):
    """ 对原始计算出来的方向场, 进行平滑处理, 取相邻几个点的平均值
    """
    half = moving_len // 2
    aver_list = []
    for i, _ in enumerate(direct_list):
        aver = []
        for j in range(max(i-half, 0), min(i+half+1, len(direct_list))):
            aver.append(direct_list[j])
        aver_list.append(np.mean(aver, axis=0).tolist())

    return aver_list

# def skeleon2direct(skel, tif_shape=(301, 301, 301), anisotropy=(200, 200, 1000)):
#     vertices = (skel.vertices / anisotropy).astype(int)
#     edges = skel.edges

#     # 找出端点和交叉点
#     num_points = len(vertices)
#     end_points = []  # 端点
#     cross_points = []  # 分叉点
#     angles = []  # 端点的中心线方向
#     for i in range(num_points):
#         cnt = 0
#         for edge in edges:
#             if i in edge:
#                 cnt += 1

#         if cnt == 1:
#             end_points.append(vertices[i].tolist()+[i])
#             idx0, idx1 = np.where(edges==i)
#             # idy1 = 0 if idx1 else 1
#             # angle = vertices[edges[idx0, idy1]] - vertices[i]
#             # angles.append(angle)
#         if cnt > 2:
#             cross_points.append(vertices[i].tolist()+[i])

#     if len(cross_points) > 0:
#         direct_mask = direct_cal(vertices, edges, end_points, cross_points, tif_shape)
#     else:
#         direct_mask = np.zeros([*tif_shape, 3])
    
#     return direct_mask

# def direct_cal(vertices, edges, end_points, cross_points, tif_shape, min_length=20):
#     direct_mask = np.zeros([*tif_shape, 3])

#     cross_id = np.array(cross_points)[:, -1]

#     for ep in end_points:
#         direct = []
#         id_ = ep[-1]
#         idx0, idx1 = np.where(edges==id_)
#         idx0, idx1 = idx0[0], idx1[0]
#         idy1 = 0 if idx1 else 1
#         length = 1
#         direct.append((vertices[edges[idx0, idy1]] - vertices[id_]).tolist()
#                       + vertices[id_].tolist())
#         while edges[idx0, idy1] not in cross_id:
#             id_ = edges[idx0, idy1]
#             idx0s, _ = np.where(edges==id_)
#             for j in idx0s:
#                 if j != idx0:
#                     idx0 = j
#                     break
#             idx1 = np.where(edges[idx0]==id_)[0]
#             idy1 = 0 if idx1 else 1
#             length += 1
#             direct.append((vertices[edges[idx0, idy1]] - vertices[id_]).tolist()
#                            + vertices[id_].tolist())

#         if length > min_length:
#             for j in direct:
#                 direct_mask[j[3], j[4], j[5], ...] = j[:3]


#     return direct_mask

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
    fov_ins_cross_array = np.zeros_like(fov_ins_array)
    directs_mask = np.zeros([*(fov_ins_array.shape), 3])
    ends_dict = {}
    for label in skels:
        skel = skels[label]
        # ends, vecs = ends_cal(skel, anisotropy)
        # ends, vecs = find_ends_angles_cross(skel, anisotropy)
        direct_mask, coords, ends, cross = skeleon2direct(skel, None, min_length=15)
        
        fov_ins_skel_array[coords[:, 0], coords[:, 1], coords[:, 2]] = label
        fov_ins_ends_array[ends[:,0], ends[:,1], ends[:,2]] = label
        ends_dict[label] = ends
        if len(direct_mask) != 0:
            directs_mask[direct_mask[:, 3].astype(int), direct_mask[:, 4].astype(int), direct_mask[:, 5].astype(int), ...] = direct_mask[:, :3]
        
        if len(cross) != 0:
            fov_ins_cross_array[cross[:, 0], cross[:, 1], cross[:, 2]] = label

    return ends_dict, fov_ins_skel_array, fov_ins_ends_array, fov_ins_cross_array, directs_mask

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# def test():
#     out_path = "/media/fcheng/temp/test_301_80_vox/"
#     mkdir_if_not_exist(out_path)

#     paths = "/media/jjx/Biology/logs/test_301_80_vox/visual_results/"
#     tif_paths = glob.glob(os.path.join(paths, "*"))
#     print(tif_paths)
#     # names = [os.path.split(name)[1] for name in names]
#     for tif_path in tif_paths:
#         name = tif_path.split('/')[-1]
#         print("Processing: {}".format(name))
#         tif = tifffile.imread(os.path.join(tif_path, name+".tif"))
#         tif = dilation(tif, cube(5))
#         _, _, _, direct_mask = skels_cal(tif)

#         direct_vis = ~np.all(direct_mask==[0,0,0], axis=3)

#         tifffile.imsave(os.path.join(out_path, name+"_direct_vis.tif"), direct_vis.astype(np.float16))
#         print("{}_direct_vis.tif have saved".format(name) )

def vis_direct():
    # direct_mask_path = "/media/fcheng/temp/test_301_80_vox/5700_35350_4150_0_direct_vis.tif"
    # direct_mask = tifffile.imread(direct_mask_path)
    paths = "/media/fcheng/temp/test_301_80_vox/visual_results/"
    tif_paths = glob.glob(os.path.join(paths, "*"))
    out_path = "/media/fcheng/temp/test_301_80_vox/direct_vis"
    mkdir_if_not_exist(out_path)

    for tif_path in tif_paths[:1]:
        name = tif_path.split('/')[-1]
        print("Processing:{}".format(name))
        tif = tifffile.imread(os.path.join(tif_path, name+".tif"))
        tif = dilation(tif, cube(5))

        skels = kimimaro_func(tif)

        direct_vis = np.zeros_like(tif)
        ends_array = np.zeros_like(tif)
        for label in skels:
            skel = skels[label]
            directs, _, ends, _ = skeleon2direct(skel, None, min_length=15)
            
            ends_array[ends[:,0], ends[:,1], ends[:,2]] = label
            for e in ends:
                sel = np.where(np.all(directs[:, 3:]==e, axis=1))[0]
                if len(sel) == 0: continue
                direct = directs[sel[0]][:3]
                direct_grow(e, direct, direct_vis, tif.shape)

        tifffile.imsave(os.path.join(out_path, name+"end_direct_vis_smooth.tif"), direct_vis.astype(np.float16))
        tifffile.imsave(os.path.join(out_path, name+"ends.tif"), ends_array.astype(np.float16))

def direct_grow(start, direct, mask, max_shape, length=20):
    for i in range(length):
        x_new = np.round(direct[0] * i + start[0]).astype(int)
        y_new = np.round(direct[1] * i + start[1]).astype(int)
        z_new = np.round(direct[2] * i + start[2]).astype(int)

        x_new = np.clip(x_new, a_min=0, a_max=max_shape[0]-1)
        y_new = np.clip(y_new, a_min=0, a_max=max_shape[1]-1)
        z_new = np.clip(z_new, a_min=0, a_max=max_shape[2]-1)
        
        mask[x_new, y_new, z_new] = 1



# vis_direct()
if __name__ == "__main__":
        
    file_path = "/media/jjx/Biology/data/data_modified/ins_modified/5700_35350_3900.tif"
    # tif = tifffile.imread("/media/fcheng/temp/ins_pred.tif")
    tif = tifffile.imread(file_path)
    # tif = dilation(tif, cube(5))

    # skels = kimimaro_func(tif)
    # # skels的keys值, 对应skels_array和labels中的标签, skels_array是labels的骨架化
    # anisotropy=np.array((200,200,1000))
    # label = 5
    # skel = skels[label]
    # end_points, angle = find_ends_angles_cross(skel)
    # direct_mask = skeleon2direct(skel, tif.shape)
    # dm = skeleon2direct(skel, tif.shape, min_length=15)
    _, skel_array, ends_array, direct_mask = skels_cal(tif)

    a = direct_mask == [0, 0, 0]
    b = np.all(a, axis=3)
    c = ~b

    print("")
    

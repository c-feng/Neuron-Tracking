import os
import sys
import torch
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH, "../../"))
from pointnet2_lib.pointnet2 import pointnet2_utils
import pdb

def sample_points(points, sample_nums=15000):
    """ Input is Nx3 points
        Output  is sample_nums x 3 points
    """
    nums = len(points)
    if nums > sample_nums:
        sel = np.random.choice(nums, size=sample_nums, replace=False)
        sample_points = points[sel]

    # if nums <= sample_nums:
    #     r_nums = sample_nums - nums

    #     # 数量不足, 使用(0,0,0)来弥补
    #     add_points = np.zeros((r_nums, 3))
    #     sample_points = np.concatenate([points, add_points], axis=0)
    #     sel = None
    else:
        sel = np.arange(0, nums)
        if nums < sample_nums:
            extra_sel = np.random.choice(sel, sample_nums - nums, replace=True)
            sel = np.concatenate([sel, extra_sel], axis=0)
        sample_points = points[sel]

    return sample_points, sel

def coords_to_volume(coords, vsize):
    """ input is N x 3 points.
        output is vsize*vsize*vsize
    """
    vol = np.zeros((vsize, vsize, vsize))
    locations = coords.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1
    return vol

def tif_to_points(tif, sample_nums=15000):
    """ Input is instance tif voxel
        Output is sample_nums x 4, [x, y, z, label]
    """
    coords = np.stack(np.where(tif > 0), axis=0).T
    # coords_l = np.concatenate([coords, tif[coords[:, 0], coords[:, 1], coords[:, 2]][:, None] ], axis=1)
    sp, sel = sample_points(coords, sample_nums)
    if sel is not None:
        sp_l = np.concatenate([sp, tif[coords[sel, 0], coords[sel, 1], coords[sel, 2]][:, None] ], axis=1)
    else:
        labels = tif[coords[:, 0], coords[:, 1], coords[:, 2]]
        labels = np.concatenate([labels, np.zeros((sample_nums-len(coords)))], axis=0)
        sp_l = np.concatenate([sp, labels[:, None]], axis=1)
    
    # sp_l[:, :3] = sp_l[:, :3] / vsize
    return  sp_l

def tif_to_points_by_FPS(tif, sample_nums=15000):
    """ Input is instance tif voxel
        Output is sample_nums x 4, [x, y, z, label]
        Use the farthest point sampling(FPS)
    """
    coords = np.stack(np.where(tif > 0), axis=0).T.astype(np.float32)
    # coords = (coords - (149.5, 149.5, 149.5)) / 149.5
    coords = coords.astype(np.float32)

    # FPS 
    coords = torch.from_numpy(coords).cuda()[None]
    xyz_flipped = coords.transpose(1, 2).contiguous()
    
    # 点的数量不够时, idx=0被用来填充
    idx = pointnet2_utils.furthest_point_sample(coords, sample_nums)
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, idx).transpose(1, 2).contiguous()

    idx = idx.cpu().numpy()[0]
    new_xyz = new_xyz.cpu().numpy()[0]
    coords = coords.cpu().numpy()[0].astype(int)

    sp_l = np.concatenate([new_xyz, tif[coords[idx, 0], coords[idx, 1], coords[idx, 2]][:, None]], axis=1)
    return sp_l

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def func1():
    from tqdm import tqdm
    from neural_dataset import NeuralDataset
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # data_root = "/home/jjx/Biology/data/data_modified/"
    data_root = "/home/jjx/Biology/data/data_synthesize/"
    out_path = "/home/fcheng/Neuron/pointnet/data/syn_data_FPS_32768/train"
    os.makedirs(out_path, exist_ok=True)

    dataset = NeuralDataset(data_root, mode="train", sample_shape=[300, 300, 300], sample_num=1)
    dataset.resetSample()
    print("reset compeleted")

    for i in range(dataset.__len__()):
        _, ins, info = dataset.readData(i)
        # sp_l = tif_to_points(ins, 8192)
        sp_l = tif_to_points_by_FPS(ins, 32768)

        sp_l = sp_l.astype(np.float)
        # sp_l[:, :3] = pc_normalize(sp_l[:, :3])
        sp_l[:, :3] = (sp_l[:, :3] - (149.5, 149.5, 149.5)) / 149.5

        name = info[0]
        np.save(os.path.join(out_path, name+".npy"), sp_l)


if __name__ == "__main__":
    # from neural_dataset import NeuralDataset
    # from skimage.external import tifffile
    # import pdb

    # data_root = "/home/jjx/Biology/data/data_modified/"
    # out_path = "/home/fcheng/Neuron/pointnet/data/debug/sample_points"
    # os.makedirs(out_path, exist_ok=True)

    # dataset = NeuralDataset(data_root, sample_shape=[300, 300, 300], sample_num=1)
    # dataset.resetSample()
    # print("reset completed")

    # nums = dataset.__len__()
    # for i in tqdm(range(nums)):
    #     _, b, info = dataset.readData(i)
    #     coords = np.stack(np.where(b>0), axis=0).T.astype(int)
    #     sp, _ = sample_points(coords)
    #     vol = coords_to_volume(sp, 300)

    #     name = info[0]
    #     print(name)
    #     print("{} ---> {}\n".format(len(coords), len(sp)))
        
    #     # tifffile.imsave(os.path.join(out_path, name+'_sp.tif'), vol.astype(np.float16))

    ######################################
    func1()

    ####################################
    # data_root = "/home/fcheng/Neuron/pointnet/data/real_data"

    # npy_files = os.listdir(data_root)
    
    # for path in npy_files:
    #     npy = np.load(os.path.join(data_root, path))
    #     print(path)
    #     print(npy.shape)


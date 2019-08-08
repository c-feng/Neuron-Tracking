import numpy as np
import os

import torch
import pdb

def sample_pointBylabel(gt, label, shape, patch_shape, condition=True, num=None):
    coords = np.stack(np.where(gt==label), axis=0).T
    margins = np.array(patch_shape) // 2
    
    # 选出满足条件的点
    if condition:
        sel = np.all((coords-margins+1-np.array(patch_shape)%2 >= 0) & 
                    (coords+margins < np.array(shape)) > 0, axis=1)
        valid_coords = coords[sel]
    else:
        valid_coords = coords

    # 选出满足数量的点
    if num is None:
        return valid_coords
    else:
        num = min(num, len(valid_coords))
    sel = np.random.choice(len(valid_coords), num, replace=False)
    valid_coords = valid_coords[sel.tolist(), :]
    return valid_coords

def sample_point(gt, num=None):
    coords = np.stack(np.where(gt > 0), axis=0).T
    
    if num is None:
        return coords
    else:
        num = min(num, len(coords))
    sel = np.random.choice(len(coords), num, replace=False)
    sel_coords = coords[sel.tolist(), :]
    labels = gt[sel_coords[:, 0], sel_coords[:, 1], sel_coords[:, 2]]
    return sel_coords, labels

def prepare_seeddata(gt, patch_shape, sample_num=1):
    """ 由labels 产生一些初始点, 作为网络推断的起始
    """
    if isinstance(gt, (np.ndarray, list, tuple)):
        shape = np.array(gt).shape
    elif isinstance(gt, torch.Tensor):
        shape = list(gt.shape)
    
    labels = np.unique(gt)[1:]
    ins_coords = []
    for i, label in enumerate(labels):
        coords = sample_pointBylabel(gt, label, shape, patch_shape, sample_num)
        ins_coords.append(coords)

    return np.array(ins_coords)

def fromSeed2Data(seed_coord, full_data, patch_shape):
    """ seed为中心点
    """
    if isinstance(full_data, (np.ndarray, list, tuple)):
        shape = np.array(full_data).shape
    elif isinstance(full_data, torch.Tensor):
        shape = list(full_data.shape)
    patch = np.zeros(patch_shape)
    
    sel = [slice(max(s, 0), e+1) for s, e in zip(
                np.array(seed_coord)+1-np.array(patch_shape)//2-np.array(patch_shape)%2,
                np.array(seed_coord)+np.array(patch_shape)//2)]
    if len(sel) == 2:
        patch = full_data[..., sel[0], sel[1]].copy()
    elif len(sel) == 3:
        patch = full_data[..., sel[0], sel[1], sel[2]].copy()
    else:
        print("the data have shape of {}".format(shape))
    return patch

def findCenterCoord(seeds, label, full_shape, patch_shape):
    """ seed位置不在中心, 重新随机crop返回中心点坐标
    """
    centers = []
    for seed in seeds:
        low = np.maximum(0, np.array(seed)-np.array(patch_shape)+1)  # (0,...)
        high = np.minimum(np.array(full_shape), 
                          np.array(seed)+np.array(patch_shape))  # (..., 300)
        high = high - np.array(patch_shape) + 1
        coord0 = []
        for i in range(len(full_shape)):
            try:
                coord0.append(np.random.randint(low[i], high[i], 1)[0])
            except ValueError:
                pdb.set_trace()
        center = np.array(coord0) + np.array(patch_shape)//2 + np.array(patch_shape)%2 - 1
        centers.append(center)
    return np.array(centers).tolist()

def sample_batch(gts, patch_shape, num=1):
    """ gts: (N, w, h, d)
        return: (max_label, N, 3)
    """
    max_label = 0
    labels_list = []
    for i in gts:
        label_ = np.unique(i)[1:].tolist()
        if label_ == []:  # modified 5.13.0:40
            label_ = [0]  # modified 5.13.0:40
        labels_list.append(label_)
        max_label = max(max_label, len(label_))
    try:
        labels_pad = [x * (max_label // len(x)) + x[:max_label % len(x)] \
                                    for x in labels_list]
    except ZeroDivisionError:
        pdb.set_trace()
    labels_pad = np.array(labels_pad).T  # (num, batch_size)
    
    coords = []
    for labels in labels_pad:
        coord = []
        for gt, label in zip(gts, labels):
            c = sample_pointBylabel(gt, label, gt.shape, patch_shape, False, num)
            c = findCenterCoord(c, label, gt.shape, patch_shape)
            coord += [np.squeeze(c).tolist()]
        coords += [coord]
    if num > 1:
        coords = np.array(coords)
        coords = np.concatenate(np.split(coords, num, -2), axis=0).squeeze()
    return np.array(coords), np.tile(labels_pad, (num, 1))

def get_batchCropData(coords, data, patch_shape):
    """ 通过给定的坐标采样crop图片, 
    """
    batch_patch = []
    for i, coord in enumerate(coords):
        p = fromSeed2Data(coord, data[i], patch_shape)
        batch_patch.append(p)
    
    return np.stack(batch_patch, axis=0)

def CenterGaussianHeatMap(shape, center, sigma=1):
    dims = len(shape)
    lins = []
    for i in range(dims):
        lins += [np.linspace(0, shape[i]-1, shape[i]).tolist()]
    
    coords = np.stack(np.meshgrid(*lins), axis=-1)
    D = 1. * np.sum(np.power(coords - center, 2), axis=-1)
    E = 2.0 * sigma * sigma
    Exponent = D / E
    heatmap = np.exp(-Exponent)
    return heatmap.swapaxes(0, 1)

def get_SeedMap(seed, label, seg_map, patch_shape, is_center=True):
    """ 由给定的中心点, 产生高斯热图
        is_center: 表示高斯热图的中心为seed
    """
    
    if is_center:
        center = np.array(patch_shape) // 2 - 1 + np.array(patch_shape) % 2
    else:
        patch = fromSeed2Data(seed, seg_map, patch_shape)
        # center = np.array(patch_shape) // 2 - 1 + np.array(patch_shape) % 2
        coords = np.stack(np.where(patch==label), axis=0).T
        sel = np.random.choice(len(coords), 1)[0]
        center = coords[sel]
    
    heatmap = CenterGaussianHeatMap(patch_shape, center, sigma=0.7)
    return heatmap

def get_batchSeedMap(seed, label, seg_map, patch_shape, is_center=False):
    hms = []
    for s, l, m in zip(seed, label, seg_map):
        hms.append(get_SeedMap(s, l, m, patch_shape, is_center))
    return np.stack(hms, axis=0)

def sample_seedfromfullseg(full_seg, ins_mask, labels_pad, patch_shape):
    """ 在训练时从整体分割的map中, 采样种子点, 来训练fov
    """
    def sample_single(full_seg, ins_mask, labels):
        num_seeds = len(labels)
        if np.any(labels==0):
            coords = np.stack(np.where(full_seg==0), axis=0).T
            sel = np.random.choice(len(coords), size=num_seeds, replace=False)
            
            return coords[sel, :].tolist(), labels
        else:
            coords = np.stack(np.where(full_seg>0), axis=0).T
            coords = list(filter(lambda x: ins_mask[x[0], x[1], x[2]]>0, coords))
            if len(coords) >= num_seeds:
                sel = np.random.choice(len(coords), size=num_seeds, replace=False)
                coords = np.array(coords)[sel, :]
            elif len(coords) > 0:
                sel = np.arange(len(coords))
                sel = sel * (num_seeds // len(sel)) + sel[:num_seeds % len(sel)]
                coords = np.array(coords)[sel, :]
            else:
                coords = np.stack(np.where(ins_mask>0), axis=0).T
                sel = np.random.choice(len(coords), size=min(len(coords), num_seeds), replace=False)
                sel = sel * (num_seeds // len(sel)) + sel[:num_seeds % len(sel)]
                coords = coords[sel, :]
            crop_coords = findCenterCoord(coords, label, full_seg.shape, patch_shape)
            return crop_coords, coords.tolist(), ins_mask[coords[:, 0], coords[:, 1], coords[:, 2]].tolist()

    batch_centers = []
    batch_seed = []
    batch_label = []
    for seg, ins, label in zip(full_seg, ins_mask, labels_pad.T):
        seeds_center, seeds, labels = sample_single(seg, ins, label)
        batch_centers.append(seeds_center)
        batch_seed.append(seeds)
        batch_label.append(labels)
    
    return batch_seed, batch_label


# 可视化
def trans3Dto2D(tensor):
    if isinstance(tensor, torch.Tensor):
        a = tensor.data.numpy()
    else:
        a = tensor
    if len(a.shape) == 5:
        N, c, w, h, d = a.shape
    elif len(a.shape) == 4:
        N, w, h, d = a.shape

    for i in range(min(10,N)):
        t = np.squeeze(a[i])
        mask = t > 0.9
        img0 = np.sum(mask, axis = 0) > 0
        img1 = np.sum(mask, axis = 1) > 0
        img2 = np.sum(mask, axis = 2) > 0
        
        img0 = img0.astype(np.int)
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        img = np.vstack([img0, np.ones([1, w]), img1, np.ones([1, w]), img2])
        if i == 0:
            imgs = img
        else:
            imgs = np.hstack([imgs, np.ones([h*3+2, 1]), img])
    return np.expand_dims(imgs, axis=0)


if __name__ == "__main__":
    # test sample_pointBylabel
    # gt = np.random.randint(0, 4, size=(10,10))
    # v_c = sample_pointBylabel(gt, 1, gt.shape, (5,5))
    # coords = np.stack(np.where(gt==1), axis=0).T
    # print("")

    # test prepare_data
    # gt = np.random.randint(0, 4, (10, 10))
    # c = prepare_seeddata(gt, (4,4))

    # test fromSeed2Data
    # data = np.random.randint(0, 4, (10, 10))
    # seed = [3, 5]
    # p = fromSeed2Data(seed, data, [4, 4])

    # test smaple_batch
    # gts = np.random.randint(0, 6, size=(8, 10,10))
    # coords, labels_pad = sample_batch(gts, patch_shape=5, num=3)

    # test get_SeedMap
    # a = np.random.randint(0, 4, (10, 10))
    a = np.array([[3, 0, 2, 0, 0, 2, 0, 3, 0, 1],
       [2, 0, 1, 2, 3, 1, 2, 3, 0, 1],
       [0, 1, 3, 0, 2, 2, 0, 1, 0, 1],
       [3, 2, 3, 2, 3, 0, 0, 2, 3, 0],
       [0, 3, 3, 0, 3, 3, 2, 0, 0, 0],
       [1, 2, 3, 3, 2, 2, 2, 3, 1, 2],
       [1, 0, 3, 1, 3, 1, 3, 2, 1, 3],
       [2, 2, 3, 3, 3, 1, 2, 2, 1, 0],
       [2, 3, 3, 0, 3, 3, 1, 0, 0, 2],
       [2, 1, 2, 3, 1, 0, 2, 3, 0, 0]])
    map_ = get_SeedMap([2,3], a, [5, 5], False)
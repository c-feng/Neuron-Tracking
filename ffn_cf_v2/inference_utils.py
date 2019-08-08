import numpy as np
from itertools import product
import torch
from scipy.special import logit, expit

def prepare_data(labels, patch_shape, nums=3):
    """ Sampling points in all instances

        Args:
            labels: (N, w, h, d)
    """
    label = labels[0]
    label = label.cpu().data.numpy()
    classes = np.unique(label)
    locations = [set() for _ in range(len(classes)-1)]
    w, h, d, _ = patch_shape
    bounds = np.array(label.shape) - np.array([w, h, d])
    for x, y, z in product(range(0, bounds[0], 1), range(0, bounds[1], 1), range(0, bounds[2], 1)):
        mx, my, mz = x + w // 2, y + h // 2, z + d // 2
        if label[mx, my, mz] == 0: continue
        l, = np.where(classes==label[mx,my,mz])[0]
        locations[l-1].add((x, y, z))
    
    return equalize(locations, nums)

def equalize(locations, nums=3):
    smallest = min([len(c) for c in locations])
    smallest = min(smallest+nums , nums)
    # print([len(c)j for c in locations])
    new_locations = []
    for c in locations:
        if len(c) < nums: continue
        ind = np.random.choice(a=len(c), size=smallest, replace=False)
        new_locations.extend(np.array(list(c))[ind].tolist())
    indices = np.random.permutation(len(new_locations))
    return np.array(new_locations)[indices]

def is_valid_seed(location, segmentation):
    x, y, z = location
    if segmentation[..., x, y, z] > 0:
        return False
    return True

def in_min_boundary_dist(location, segmentation, mbd=[1,1,1]):
    x, y, z = location
    mbd = np.array(mbd)
    low = np.array(location) - mbd
    high = np.array(location) + mbd + 1
    sel = [slice(s, e) for s, e in zip(low, high)]
    if np.any(segmentation[sel] > 0):
        segmentation[..., x, y, z] = -1
        return True
    return False

def mask_subvol(subvol_shape):
    w, h, d, _ = subvol_shape
    subvol_mask = 0.05 * np.ones([1, 1, w, h, d], np.float32)
    # subvol_mask[0, 0, w//2, h//2, d//2] = 0.95
    return logit(subvol_mask)

def get_data(volume, center, subvol_shape):
    w, h, d, c = subvol_shape
    subvol = np.zeros(shape=[1, c, w, h, d], dtype=np.float32)
    lw, lh, ld = center - np.array([w, h, d], np.int32) // 2
    hw, hh, hd = center + np.array([w, h, d], np.int32) // 2 + 1
    subvol[0, 0, :, :, :] = volume[0, 0, lw:hw, lh:hh, ld:hd]
    return subvol

def set_data(volume, center, subvol, disco_seed_threshold=5):
    """ 将当前输出的概率图返回到patch_mask中
        Args:
            volume: patch_mask, [N, 1, W, H, D]
            center: 当前FOV的中心点在volume中的位置
            subvol: 输出概率图mask, [N, 1, W, H, D]
    """
    _, c, w, h, d = subvol.shape
    lw, lh, ld = center - np.array([w, h, d], np.int32) // 2
    hw, hh, hd = center + np.array([w, h, d], np.int32) // 2 + 1
    if np.mean(subvol >= logit(0.9)) > disco_seed_threshold:
        old_seed = volume[:, :, lw:hw, lh:hh, ld:hd]
        mask = ((subvol > old_seed) & (old_seed < logit(0.5)))
        subvol[mask] = old_seed[mask]
    volume[:, :, lw:hw, lh:hh, ld:hd] = subvol

def get_new_locs(mask, delta, tmove):
    new_locs = []
    dx, dy, dz = delta
    _, c, x, y, z= np.array(mask.shape) // 2
    submask = mask[0, 0, x - dx: x+dx+1, y-dy:y+dy+1,z-dz:z+dz+1]
    xminus, xplus = submask[0, :, :], submask[2*dx,:,:]
    yminus, yplus = submask[:, 0, :], submask[:,2*dy,:]
    zminus, zplus = submask[:, :, 0], submask[:,:,2*dz]

    i, j = np.unravel_index(xminus.argmax(), xminus.shape)
    if xminus[i, j] >= logit(tmove):
        new_locs.append((-dx, i - dy, j - dz, xminus[i, j]))
    i, j = np.unravel_index(xplus.argmax(), xplus.shape)
    if xplus[i, j] >= logit(tmove):
        new_locs.append((dx, i - dy, j - dz, xplus[i, j]))
    i, j = np.unravel_index(yminus.argmax(), yminus.shape)
    if yminus[i, j] >= logit(tmove):
        new_locs.append((i-dx,- dy, j - dz, yminus[i, j]))
    i, j = np.unravel_index(yplus.argmax(), yplus.shape)
    if yplus[i, j] >= logit(tmove):
        new_locs.append((i-dx, dy, j - dz, yplus[i, j]))
    i, j = np.unravel_index(zminus.argmax(), zminus.shape)
    if zminus[i, j] >= logit(tmove):
        new_locs.append((i-dx, j - dy,- dz, zminus[i, j]))
    i, j = np.unravel_index(zplus.argmax(), zplus.shape)
    if zplus[i, j] >= logit(tmove):
        new_locs.append((i-dx, j - dy, dz, zplus[i, j]))

    new_locs = sorted(new_locs, key=lambda l:l[-1], reverse = True)
    new_locs = np.array(new_locs)[:,0:3] if new_locs else []
    return new_locs
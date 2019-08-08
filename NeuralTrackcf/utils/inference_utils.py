import numpy as np
from scipy.special import logit, expit
import pdb
from skimage.morphology import dilation, cube, ball

from .utils import fromSeed2Data

def set_data(volume, center, subvol, covered=False):
    """ 将当前的fov预测结果, 添加到patch_mask中
        Args:
            volume: patch_mask, [N, 1, W, H, D]
            center: 当前Fov的中心点, 在volume中的坐标
            subvol: 当前Fov的预测结果
    """
    patch_shape = subvol.shape
    sel = [slice(max(s, 0), e+1) for s, e in zip(
            np.array(center)+1-np.array(patch_shape)//2-np.array(patch_shape)%2,
            np.array(center)+np.array(patch_shape)//2)]
    label = np.unique(subvol)[-1]
    if covered:
        volume[..., sel[0], sel[1], sel[2]][..., subvol>0] = subvol[subvol>0]
    else:
        mask = ((volume[..., sel[0], sel[1], sel[2]]==0) & (subvol>0)).astype(int)
        c = np.stack(np.where(mask>0), axis=0).T
        # volume[..., sel[0], sel[1], sel[2]] = (volume[..., sel[0], sel[1], sel[2]]==0) & subvol
        volume[..., sel[0], sel[1], sel[2]][..., c[:, 0], c[:, 1], c[:, 2]] = subvol[c[:, 0], c[:, 1], c[:, 2]]

def get_new_locations(mask, patch_mask, coord, pred, delta=[20, 20, 20], tmove=logit(0.95)):
    """ 由当前的fov 输出概率, patch_mask为已经分割完成的部分, coord为
        当前fov在patch_mask中的位置, 中心坐标
        mask: [41, 41, 41]
        patch_mask: [72, 72, 72]

    """
    # prob = -100 * np.ones(mask.shape)
    # prob[pred>0] = mask[pred>0]
    prob = mask

    dx, dy, dz = delta
    _, x, y, z = np.array(mask.shape) // 2
    submask = prob[0, x - dx: x+dx+1, y-dy:y+dy+1,z-dz:z+dz+1]

    xminus, xplus = submask[0, :, :], submask[2*dx,:,:]
    yminus, yplus = submask[:, 0, :], submask[:,2*dy,:]
    zminus, zplus = submask[:, :, 0], submask[:,:,2*dz]

    new_locs = []
    i, j = np.unravel_index(xminus.argmax(), xminus.shape)
    if xminus[i, j] >= tmove:
        new_locs.append((-dx, i - dy, j - dz, xminus[i, j]))
    i, j = np.unravel_index(xplus.argmax(), xplus.shape)
    if xplus[i, j] >= tmove:
        new_locs.append((dx, i - dy, j - dz, xplus[i, j]))
    i, j = np.unravel_index(yminus.argmax(), yminus.shape)
    if yminus[i, j] >= tmove:
        new_locs.append((i-dx,- dy, j - dz, yminus[i, j]))
    i, j = np.unravel_index(yplus.argmax(), yplus.shape)
    if yplus[i, j] >= tmove:
        new_locs.append((i-dx, dy, j - dz, yplus[i, j]))
    i, j = np.unravel_index(zminus.argmax(), zminus.shape)
    if zminus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy,- dz, zminus[i, j]))
    i, j = np.unravel_index(zplus.argmax(), zplus.shape)
    if zplus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy, dz, zplus[i, j]))

    if new_locs != []:
        subpatch = fromSeed2Data(coord, patch_mask, mask.shape[1:])
        for loc in new_locs:
            if np.any(subpatch[loc[0]-2:loc[0]+3, loc[1]-2:loc[1]+3, loc[2]-2:loc[2]+3] > 0):
                new_locs.remove(loc)

    new_locs = sorted(new_locs, key=lambda l:l[-1], reverse = True)
    new_locs = np.array(new_locs)[:,0:3] if new_locs else []
    return new_locs

def get_new_locations_v1(mask, patch_mask, coord, pred, delta=[20, 20, 20], tmove=1):
    """ 由当前的fov 输出概率, patch_mask为已经分割完成的部分, coord为
        当前fov在patch_mask中的位置, 中心坐标
        mask: [41, 41, 41]
        patch_mask: [72, 72, 72]

    """
    # prob = -100 * np.ones(mask.shape)
    # prob[pred>0] = mask[pred>0]
    prob = pred

    dx, dy, dz = delta
    _, x, y, z = np.array(mask.shape) // 2
    submask = prob[0, x - dx: x+dx+1, y-dy:y+dy+1,z-dz:z+dz+1]

    xminus, xplus = submask[0, :, :], submask[2*dx,:,:]
    yminus, yplus = submask[:, 0, :], submask[:,2*dy,:]
    zminus, zplus = submask[:, :, 0], submask[:,:,2*dz]

    new_locs = []
    i, j = np.unravel_index(xminus.argmax(), xminus.shape)
    if xminus[i, j] >= tmove:
        new_locs.append((-dx, i - dy, j - dz, xminus[i, j]))
    
    i, j = np.unravel_index(xplus.argmax(), xplus.shape)
    if xplus[i, j] >= tmove:
        new_locs.append((dx, i - dy, j - dz, xplus[i, j]))
    
    i, j = np.unravel_index(yminus.argmax(), yminus.shape)
    if yminus[i, j] >= tmove:
        new_locs.append((i-dx,- dy, j - dz, yminus[i, j]))
    
    i, j = np.unravel_index(yplus.argmax(), yplus.shape)
    if yplus[i, j] >= tmove:
        new_locs.append((i-dx, dy, j - dz, yplus[i, j]))
    
    i, j = np.unravel_index(zminus.argmax(), zminus.shape)
    if zminus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy,- dz, zminus[i, j]))
    
    i, j = np.unravel_index(zplus.argmax(), zplus.shape)
    if zplus[i, j] >= tmove:
        new_locs.append((i-dx, j - dy, dz, zplus[i, j]))

    if new_locs != []:
        subpatch = fromSeed2Data(coord, patch_mask, mask.shape[1:])
        for loc in new_locs:
            if subpatch[loc[0], loc[1], loc[2]] > 0:
                new_locs.remove(loc)

    new_locs = sorted(new_locs, key=lambda l:l[-1], reverse = True)
    new_locs = np.array(new_locs)[:,0:3] if new_locs else []
    return new_locs

def getfovLabel(coord, mask, seg, seed_mask, fov_shape):
    """ 由patch的当前分割结果, 结合整体分割map, 得到下一个
        未分割实例的seed点
    """
    mask_fov = fromSeed2Data(coord, mask, fov_shape)
    seg_fov = fromSeed2Data(coord, seg, fov_shape)
    seed_mask_fov = fromSeed2Data(coord, seed_mask, fov_shape)

    mask1 = np.zeros(mask_fov.shape)
    mask1[seg_fov>0] = mask_fov[seg_fov>0]
    # mask_fov[seg_fov==0] = 0  
    mask1 = dilation(mask1, cube(3))
    seed_map = ((mask1==0) & (seg_fov>0)).astype(int)  # 剩下未分割的区域
    
    seed_mask_fov_a = np.zeros(seed_mask_fov.shape)
    seed_mask_fov_a[seed_mask_fov>0] = seed_mask_fov[seed_mask_fov>0]
    # seed_mask_fov_a = dilation(seed_mask_fov_a, cube(5))
    # prev_seed = ((seed_map>0) & (seed_mask_fov>0)).astype(int)
    prev_seed = ((seed_map>0) & (seed_mask_fov_a>0)).astype(int)

    if np.sum(prev_seed) > 0:
        coords = np.stack(np.where(prev_seed>0), axis=0).T
    else:
        coords = np.stack(np.where(seed_map>0), axis=0).T
    sel = np.random.choice(len(coords), size=1)[0]

    coord = coords[sel]
    label = seed_mask_fov_a[..., coord[0], coord[1], coord[2]]

    return label, coord


import numpy as np
from itertools import product
import torch
from scipy.special import logit


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


# def batch_subvol(data, labels, subvol_shape, location):
#     w, h, d, c = subvol_shape
#     data_ = data.cpu()
#     labels_ = labels.cpu()
#     subvol_data = np.zeros(shape=[1, c, w, h, d], dtype=np.float32)
#     subvol_labels = np.zeros(shape=[1, 1, w, h, d], dtype=np.float32)
#     x, y, z = location
#     subvol_data[0] = data_[0, :, x:x+w, y:y+h, z:z+d]
#     subvol_labels[0, 0, :, :, :] = labels_[0, x:x+w, y:y+h, z:z+d] == labels_[0, x+w//2, y+h//2, z+d//2]
#     subvol_labels = 0.9 * subvol_labels + 0.05
#     return subvol_data, subvol_labels
def patch_subvol(data, labels, subvol_shape, deltas, location):
    w, h, d, c = subvol_shape
    dw, dh, dd = deltas  # 以当前点出发, 至少有一个fov的大小可以保证
    x, y, z = location

    data_ = data.cpu()
    labels_ = labels.cpu()
    label_mask = np.zeros(shape=labels_.shape, dtype=np.float32)
    label_mask[:] = labels_ == labels_[0, x+w//2, y+h//2, z+d//2]
    label_mask = 0.9 * label_mask + 0.05
    
    _, _, _, low_coords, upper_coords = cal_lower_upper(label_mask[0])
    low_coords = np.maximum(0, low_coords-deltas)
    upper_coords = np.minimum(299, upper_coords+deltas)
    pw, ph, pd = upper_coords - low_coords + 1
    lw, lh, ld = low_coords
    hw, hh, hd = upper_coords

    relative_loc = location + np.array(subvol_shape[:3])//2 - low_coords

    subvol_data = np.zeros(shape=[1, c, pw, ph, pd], dtype=np.float32)
    subvol_labels = np.zeros(shape=[1, 1, pw, ph, pd], dtype=np.float32)
    subvol_data[0] = data_[0, :, lw:hw+1, lh:hh+1, ld:hd+1]
    subvol_labels[0] = label_mask[:, lw:hw+1, lh:hh+1, ld:hd+1]
    return subvol_data, subvol_labels, relative_loc


# def mask_subvol(subvol_shape):
#     w, h, d, _ = subvol_shape
#     subvol_mask = 0.05 * np.ones([1, 1, w, h, d], np.float32)
#     subvol_mask[0, 0, w//2, h//2, d//2] = 0.95
#     return logit(subvol_mask)
def mask_subvol(subvol_shape, rel_loc):
    if len(subvol_shape) > 3:
        _, _, w, h, d = subvol_shape
    else:
        w, h, d = subvol_shape
    x, y, z = rel_loc
    subvol_mask = 0.05 * np.ones([1, 1, w, h, d], np.float32)
    subvol_mask[0, 0, x, y, z] = 0.95
    return logit(subvol_mask)

def get_data(volume, center, subvol_shape):
    w, h, d, c = subvol_shape
    subvol = np.zeros(shape=[1, c, w, h, d], dtype=np.float32)
    lw, lh, ld = center - np.array([w, h, d], np.int32) // 2
    hw, hh, hd = center + np.array([w, h, d], np.int32) // 2 + 1
    subvol[0, 0, :, :, :] = volume[0, 0, lw:hw, lh:hh, ld:hd]
    return subvol

def get_weights(fov_labels):
    sum_pos = np.sum(fov_labels==0.95)
    sum_neg = np.sum(fov_labels==0.05)
    # return torch.Tensor([sum_neg/sum_pos])
    # return sum_pos/(sum_neg+sum_pos)
    return sum_neg/sum_pos

def set_data(volume, center, subvol):
    """ 将当前输出的概率图返回到patch_mask中
        Args:
            volume: patch_mask, [N, 1, W, H, D]
            center: 当前FOV的中心点在volume中的位置
            subvol: 输出概率图mask, [N, 1, W, H, D]
    """
    _, c, w, h, d = subvol.shape
    lw, lh, ld = center - np.array([w, h, d], np.int32) // 2
    hw, hh, hd = center + np.array([w, h, d], np.int32) // 2 + 1
    # volume[0, :, lw:hw, lh:hh, ld:hd][softmax_sub[0]>0.9] = 0.95
    volume[0, :, lw:hw, lh:hh, ld:hd] = subvol[0, 0:, :, :, :]

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


def set_data_inference(volume, center, subvol):
    """ 将当前输出的概率图返回到patch_mask中
        Args:
            volume: patch_mask, [N, 1, W, H, D]
            center: 当前FOV的中心点在volume中的位置
            subvol: 输出概率图mask, [N, 1, W, H, D]
    """
    _, c, w, h, d = subvol.shape
    softmax_sub = subvol[0, 0:, :, :, :]
    lw, lh, ld = center - np.array([w, h, d], np.int32) // 2
    hw, hh, hd = center + np.array([w, h, d], np.int32) // 2 + 1
    vt_1 = volume[0, :, lw:hw, lh:hh, ld:hd]
    indx = (softmax_sub > vt_1) & (vt_1 < 0.5)
    volume[0, :, lw:hw, lh:hh, ld:hd][~indx] = softmax_sub[~indx]

def cal_lower_upper(mask):
    axis = np.stack(np.where(mask>0.5), axis=0).transpose()
    assert len(axis) > 0, print(len(axis))
    axis0_min = np.min(axis[:, 0])
    axis0_max = np.max(axis[:, 0])

    axis1_min = np.min(axis[:, 1])
    axis1_max = np.max(axis[:, 1])

    axis2_min = np.min(axis[:, 2])
    axis2_max = np.max(axis[:, 2])

    w = axis0_max - axis0_min + 1
    h = axis1_max - axis1_min + 1
    d = axis2_max - axis2_min + 1
    low_coords = [axis0_min, axis1_min, axis2_min]
    upper_coords = [axis0_max, axis1_max, axis2_max]

    return w, h, d, np.array(low_coords), np.array(upper_coords)

def trans3Dto2D(tensor):
    if isinstance(tensor, torch.Tensor):
        a = tensor.data.numpy()
    else:
        a = tensor
    if len(tensor.shape) > 3:
        a = np.squeeze(a)
    w, h, d = a.shape
    mask = a > 0.9
    img0 = np.sum(mask, axis = 0) > 0
    img1 = np.sum(mask, axis = 1) > 0
    img2 = np.sum(mask, axis = 2) > 0
    
    img0 = img0.astype(np.int)
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    img = np.hstack([img0, np.ones([w, 1]), img1, np.ones([w, 1]), img2])
    
    return np.expand_dims(img, axis=0)
    # return np.zeros([1,30,30])


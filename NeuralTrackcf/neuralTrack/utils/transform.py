import numpy as np
import numbers
import torch 

from skimage.transform import resize
from skimage.util import pad,crop
from skimage.morphology import ball, rectangle, dilation, cube
from scipy.ndimage.interpolation import zoom
from skimage.external import tifffile

from .mask_utils import dilated_mask_generate, range_mask_generate


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        #print(len(args))
        for i,t in enumerate(self.transforms):
            #print(i)
            img, gts = args
            #print(img.shape,gts.shape)
            args = t(*args)
        return args


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if label is None:
            return image,
        else:
            return image, label

class ToTensor(object):

    def __call__(self, img, gts = None):
        img = torch.from_numpy(img.astype(float))
        if gts is None:
            return img,
        else:
            gts = torch.from_numpy(gts.astype(int))
            gts = gts.long()
            return img, gts

class Rescale(object):
    def __init__(self,size = [300,300,300]):
        assert len(size) == 3
        self.size = np.array(size)

    def __call__(self, img, gts):
        assert len(gts.shape) == 4

        img = resize(img, self.size,mode='reflect',order = 0,anti_aliasing=False,preserve_range  = True)
        gts_ = []
        for gt in gts:
            gt = resize(gt,self.size,mode='reflect',order = 0,anti_aliasing=False,preserve_range  = True)
            gts_.append(gt[None])
            gts_ = np.concatenate(gts_)
        return img, gts_

def img_crop(img, coord, size = [80, 80, 80]):
    img_s = img.shape
    size_ = np.array(size)
    l_s = size_ // 2
    r_s = size_ - l_s
    x, y, z = coord
    
    x_l = x - l_s[0] if x - l_s[0] >= 0 else 0
    y_l = y - l_s[1] if y - l_s[1] >= 0 else 0
    z_l = z - l_s[2] if z - l_s[2] >= 0 else 0

    x_p_l = l_s[0] - (x - x_l)
    y_p_l = l_s[1] - (y - y_l)
    z_p_l = l_s[2] - (z - z_l)


    x_r = x + r_s[0] if x + r_s[0] <= img_s[0] else img_s[0]
    y_r = y + r_s[1] if y + r_s[1] <= img_s[1] else img_s[1]
    z_r = z + r_s[2] if z + r_s[2] <= img_s[2] else img_s[2]

    x_p_r = r_s[0] - (x_r - x)
    y_p_r = r_s[1] - (y_r - y)
    z_p_r = r_s[2] - (z_r - z)

    pad_length = np.array([[x_p_l, x_p_r], [y_p_l, y_p_r], [z_p_l, z_p_r]])
    img = pad(img, pad_length, mode = "constant")
    img_c = img[x_l:x_r + x_p_l + x_p_r, y_l:y_r + y_p_l + y_p_r, z_l:z_r + z_p_l + z_p_r]
    return img_c

class RandomCrop(object):
    def __init__(self, size = [80, 80, 80]):
        assert len(size) == 3
        self.size = size

    def __call__(self, img, gts):
        assert len(gts.shape) == 4

        seg = gts[0]
        if np.sum(seg) > 0:
            inds = np.arange(seg.size)[(seg > 0).flatten()]
        else:
            inds = np.arange(seg.size)
        center_ind = np.random.choice(inds,1)[0]

        center_coord = np.unravel_index(center_ind, seg.shape)

        #print(center_coord)

        img_ = img_crop(img, center_coord, self.size)
        gts_ = []
        for gt in gts:
            gt_ = img_crop(gt, center_coord, self.size)
            gts_.append(gt_[None])
        gts_ = np.concatenate(gts_, axis = 0)
        return img_ , gts_

class NearCenterCropToTensor(object):
    def __init__(self, size = [32, 32, 32], intersect = [3,3,3],flag = True, rate = 1.0):
        assert len(size) == 3
        self.size = size
        self.flag = flag
        self.intersect = intersect
        self.rate = rate
    
    def center_offset(self, coord, img_size):
        size = self.size
        axis = self.axis
        
        x, y = coord[:axis] + coord[axis + 1:]
        x_, y_ = img_size[:axis] + img_size[axis + 1:]
        x_s, y_s  = size[:axis] + size[axis + 1:]
        
        dilated_mask = dilated_mask_generate([x, y], [x_, y_], rectangle, [x_s, y_s])
        range_mask = dilated_mask_generate([x, y], [x_, y_], rectangle, [x_ - x_s, y_ - y_s]) 
        
        mask = np.logical_and(dilated_mask, range_mask)
        ind_sel = np.random.choice(np.arange(mask.size)[mask.flatten()], 1)[0]
        coord_offset = list(np.unravel_index(ind_sel, mask.shape))

        coord_offset.insert(axis, coord[axis])
        return coord_offset

    def __call__(self, img, gts):
        #assert len(gts.shape) == 4
        #assert gts.shape[0] == 5

        axis = int(np.random.choice(3,1)[0])
        self.axis = axis

        size = self.size
        intersect = self.intersect

        crop_size = np.array(size)
        crop_size[axis] = crop_size[axis] * 2 - intersect[axis]
        
        img_size = img.shape
        #print(crop_size, img_size)
        #range_mask = range_mask_generate(crop_size, img_size)
        #print(np.sum(range_mask), crop_size, img_size)

        #seg, end, ins, junc, centerline = gts
        ins, junc = gts

        seed_mask = ins > 0
        #mask = np.logical_and(seed_mask, range_mask)
        mask = seed_mask
        if np.sum(mask) == 0:
            mask = seed_mask
        seed_inds = np.arange(mask.size)

        if self.flag:
            junc_mask = np.logical_and(dilation(junc > 0, cube(6)), mask)
            if np.random.rand() < self.rate and np.sum(junc_mask) > 0:
                inds_ = seed_inds[junc_mask.flatten()]   
            else:
                inds_ = seed_inds[mask.flatten()]
        else:
            inds_ = seed_inds[mask.flatten()]
        
        ind_sel = np.random.choice(inds_, size = 1)[0]
        coord = list(np.unravel_index(ind_sel, img_size))

        #ins_center = ins[coord[0], coord[1], coord[2]]
        #coord_offset = self.center_offset(coord, img_size)
        #print(coord, coord_offset, self.axis)
        coord_offset = coord

        img_ = img_crop(img, coord_offset, crop_size)
        ins_ = img_crop(ins, coord_offset, crop_size)
        junc_ = img_crop(junc, coord_offset, crop_size)

        
        img_ = torch.from_numpy(img_)
        ins_ = torch.from_numpy(ins_)
        junc_ = torch.from_numpy(junc_)
        axis = self.axis

        prev_img = torch.narrow(img_, axis, 0, self.size[axis])
        cur_img = torch.narrow(img_, axis, self.size[axis] - self.intersect[axis], self.size[axis])

        prev_ins = torch.narrow(ins_, axis, 0, self.size[axis])
        cur_ins = torch.narrow(ins_, axis, self.size[axis] - self.intersect[axis], self.size[axis])

        prev_junc = torch.narrow(junc_, axis, 0, self.size[axis])
        cur_junc = torch.narrow(junc_, axis, self.size[axis] - self.intersect[axis], self.size[axis])

        imgs = torch.cat([prev_img[None], cur_img[None]], dim = 0)
        ins = torch.cat([prev_ins[None], cur_ins[None]], dim = 0)
        juncs = torch.cat([prev_junc[None], cur_junc[None]], dim = 0)
        
        gts = torch.cat([ins, juncs], dim = 0)
        
        return imgs, gts
        

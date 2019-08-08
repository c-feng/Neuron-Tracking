import numpy as np
import torch

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


import os
import numpy as np
from skimage.morphology import dilation, ball
from skimage.external import tifffile
from scipy import ndimage
import time

def direct_field3D(a):

    b, ind = ndimage.distance_transform_edt(a, return_indices=True)

    c = np.array(np.unravel_index(np.arange(a.size), shape=a.shape)).reshape(3, *a.shape)

    direction = ind - c
    direction[..., b==0] = -1
    dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)
    direction = direction / dr
    

    theta = np.arccos(direction[2, ...] / dr)
    phi = np.arctan2(direction[1, ...], direction[0, ...])

    direction[..., b==0] = 0
    
    direct_vis = (theta + 10) * 100 + (phi + 10) * 100
    direct_vis[b==0] = 0

    return direction, direct_vis

def process_a_tif(tif_path):
    tif = tifffile.imread(tif_path)
    tif_d = dilation(tif, ball(1))

    df = direct_field3D(tif_d)

    return df

tif_path = "./3450_31350_5150.tif"
s = time.time()
df, direct_vis = process_a_tif(tif_path)
print("time:", time.time()-s)


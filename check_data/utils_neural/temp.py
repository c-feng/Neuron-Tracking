import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from skimage.external import tifffile
import time

EPS = 1e-10

def direct_field2D(a):
        
    h, w = a.shape


    fig = plt.subplot()
    fig.imshow(a, cmap=plt.cm.gray)
    plt.show()

    b, ind = ndimage.distance_transform_edt(a, return_indices=True)

    # c = np.zeros((2, h, w))
    # for i in range(h):
    #     for j in range(w):
    #         c[:, i, j] = [i, j]
    c = np.array(np.unravel_index(np.arange(a.size), shape=a.shape)).reshape(2, *a.shape)

    direction = ind - c  # (2, N, N)
    # dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)

    # theta = np.zeros((h, w))
    # for i in range(h):
    #     for j in range(w):
    #         theta[i,j] = math.atan2(direction[1, i,j], direction[0, i, j])
    theta = np.arctan2(direction[1, ...], direction[0, ...])

    degree = theta / 3.14159 * 180
    degree[b==0] = -1

    degree_vis = (degree + 180) / 360
    degree_vis[b==0] = 0
    fig = plt.subplot()
    # fig.imshow(degree_vis, cmap=plt.cm.gray)
    fig.imshow(degree_vis)
    plt.show()

def direct_field3D(a):
    h, w, d = a.shape
    # fig = plt.subplot()
    # ax = Axes3D(fig)
    # X = np.arange(0, h)
    # Y = np.arange(0, w)
    # X, Y = np.meshgrid(X, Y)
    

    b, ind = ndimage.distance_transform_edt(a, return_indices=True)
    # c = np.zeros((3, h, w, d))
    # for i, j, k in product(range(h), range(w), range(d)):
    #     c[..., i, j, k] = [i, j, k]
    c = np.array(np.unravel_index(np.arange(a.size), shape=a.shape)).reshape(3, *a.shape)

    direction = ind - c
    direction[..., b==0] = -1
    dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)
    direction = direction / dr
    
    # theta = np.zeros((h, w, d))
    # for i, j, k in product(range(h), range(w), range(d)):
    #     theta[i, j, k] = math.acos(direction[2, i, j, k] / dr[i, j, k])
    # theta = np.arccos(direction[2, ...] / dr)

    # phi = np.zeros((h, w, d))
    # for i, j, k in product(range(h), range(w), range(d)):
    #     phi[i, j, k] = math.atan2(direction[1, i, j, k], direction[0, i, j, k])
    # phi = np.arctan2(direction[1, ...], direction[0, ...])

    direction[..., b==0] = 0
    # direct_vis = (theta + 10) * 100 + (phi + 10) * 100
    # direct_vis[b==0] = 0

    return direction


img_path = r"C:\Users\Administrator\Desktop\check_data\utils_neural\004988.png"
a = io.imread(img_path, as_gray=True)
# a = np.random.randint(0,2, (shape, shape))
# a = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
#               [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
#               [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
#               [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
#               [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#               [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
#               [1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
#               [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
#               [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
#               [0, 1, 1, 1, 0, 0, 1, 0, 0, 0]])
# a = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
#               [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
#               [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
#               [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
#               [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#               [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
#               [1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
#               [1, 1, 0, 1, 0, 1, 0.5, 0, 1, 0],
#               [0, 1, 0, 1, 1, 0, 0.5, 0, 0, 1],
#               [0, 1, 1, 1, 0, 0, 1, 0, 0, 0]])

s = time.time()
direct_field2D(a)
print(time.time()-s)

# a = np.zeros((10, 10, 10))
# a[3] = 1
# a[2:9, 3:5, 4:6] = 1
# s = time.time()
# field, f_vis = direct_field3D(a)
# print(time.time()-s)


# tif_path = r"H:\temp\temp\ins_gt\ins_gt_1.tif"
# a = tifffile.imread(tif_path)
# s = time.time()
# field = direct_field3D(a)
# print(time.time()-s)
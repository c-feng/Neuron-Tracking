import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from skimage.external import tifffile
from functools import partial


EPS = 1e-10
PI = math.pi

def direct_field2D(a):
        
    h, w = a.shape


    fig = plt.subplot()
    fig.imshow(a, cmap=plt.cm.gray)
    plt.show()

    b, ind = ndimage.distance_transform_edt(a, return_indices=True)

    c = np.zeros((2, h, w))
    for i in range(h):
        for j in range(w):
            c[:, i, j] = [i, j]

    direction = ind - c  # (2, N, N)
    # dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)

    theta = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            theta[i,j] = math.atan2(direction[1, i,j], direction[0, i, j])

    degree = theta / 3.14159 * 180
    degree[b==0] = -1

    degree_vis = (degree + 180) / 360
    degree_vis[b==0] = 0
    fig = plt.subplot()
    # fig.imshow(degree_vis, cmap=plt.cm.gray)
    fig.imshow(degree_vis)
    plt.show()

def direct_field2D_not(a):
        
    h, w = a.shape


    fig = plt.subplot()
    fig.imshow(a, cmap=plt.cm.gray)
    plt.show()

    nota = np.zeros_like(a)
    nota[a==0] = 1
    b, ind = ndimage.distance_transform_edt(nota, return_indices=True)

    c = np.zeros((2, h, w))
    for i in range(h):
        for j in range(w):
            c[:, i, j] = [i, j]

    direction = ind - c  # (2, N, N)
    # dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)

    theta = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            theta[i,j] = math.atan2(direction[1, i,j], direction[0, i, j])

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

    direction = ind - c  # (3, h, w, d)
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

def tif_df():
    pass

def multi_process_func(labels, tif, out_dir, name, num_proc=4):
    with Pool(processes=num_proc) as pool:
        ps = pool.map(partial(write_file, tif=tif, out_dir=out_dir, name=name), labels)
    return ps



if __name__ == "__main__":
        
    # img_path = r"C:\Users\Administrator\Desktop\check_data\utils_neural\009764.png"
    # a = io.imread(img_path, as_gray=True)
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

    a = np.zeros((100, 100))
    a[23:, 78:81] = 1
    a[13:, 23:25] = 1

    # direct_field2D(a)
    direct_field2D_not(a)

    # a = np.zeros((10, 10, 10))
    # a[3] = 1
    # a[2:9, 3:5, 4:6] = 1
    # field, f_vis = direct_field3D(a)

    # tif_path = r"H:\temp\temp\ins_gt\ins_gt_1.tif"
    # a = tifffile.imread(tif_path)
    # field, f_vis = direct_field3D(a)

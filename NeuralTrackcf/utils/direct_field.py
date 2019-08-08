import numpy as np
from scipy import ndimage
from itertools import product
from skimage.external import tifffile

def direct_field3D(a):
    """ a is a mask foreground > 0, background == 0
    """

    distance, ind = ndimage.distance_transform_edt(a, return_indices=True)

    if len(a.shape) == 3:
        c = np.array(np.unravel_index(np.arange(a.size), dims=a.shape)).reshape(3, *a.shape)
    elif len(a.shape) == 2:
        c = np.array(np.unravel_index(np.arange(a.size), dims=a.shape)).reshape(2, *a.shape)


    direction = ind - c  # (3, h, w, d)
    direction[..., distance==0] = -1
    dr = np.power(np.power(direction, 2).sum(axis=0), 0.5)
    direction = direction / dr

    if False:
        import math
        # theta = np.arccos(direction[2, ...] / dr)
        phi = np.arctan2(direction[1, ...], direction[0, ...])
        vis = ((phi + math.pi) / (2*math.pi)) * 1000
        vis[distance==0] = 0


    if len(a.shape) == 2:
        import math
        import matplotlib.pyplot as plt
        theta = np.arctan2(direction[1, ...], direction[0, ...])
        vis = (theta + math.pi) / 2*math.pi
        vis = theta
        vis[distance==0] = 0
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(a)
        axs[1].imshow(vis)
        plt.show()

    direction[..., distance==0] = 0

    return direction

def batch_direct_field3D(inputs):
    df = []
    for a in inputs:
        df.append(direct_field3D(a))
    
    return np.stack(df, axis=0)


if __name__ == "__main__":
    # from skimage import io
    # img_path = r"C:\Users\Administrator\Desktop\check_data\utils_neural\009764.png"
    # a = io.imread(img_path, as_gray=True)

    import time
    # from skimage.external import tifffile
    # tif_path = r"C:\Users\Administrator\Desktop\realData\ins_gt(1).tif"
    # a = tifffile.imread(tif_path)

    a = np.zeros((10, 10, 10))
    a[3] = 1
    a[2:9, 3:5, 4:6] = 1

    s = time.time()
    _, vis = direct_field3D(a)
    tifffile.imsave(r"C:\Users\Administrator\Desktop\realData\vis.tif", vis.astype(int))
    print("Shape:{}, time: {}s".format(a.shape, time.time()-s))

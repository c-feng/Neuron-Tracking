import numpy as np
from scipy import ndimage

def direct_field3D(a):
    """ a is a mask foreground > 0, background == 0
    """

    distance, ind = ndimage.morphology.distance_transform_edt(a, return_indices=True)

    if len(a.shape) == 3:
        c = np.array(np.unravel_index(np.arange(a.size), shape=a.shape)).reshape(3, *a.shape)
    elif len(a.shape) == 2:
        c = np.array(np.unravel_index(np.arange(a.size), shape=a.shape)).reshape(2, *a.shape)


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
        vis[a==0] = 0
        _, axs = plt.subplots(1, 3)
        axs[0].imshow(a, cmap=plt.cm.gist_heat)
        # axs[1].imshow(vis, cmap=plt.cm.CMRmap)
        axs[1].imshow(vis, cmap=plt.cm.gist_heat)
        axs[2].imshow(distance, cmap=plt.cm.spring_r)
        plt.show()

    direction[..., distance==0] = 0

    return direction

def skeletonfromDF(df, mask):
    h, w = mask.shape
    skel = np.zeros_like(mask)
    near = np.zeros((3,3))
    val = np.stack(np.where(mask), axis=0).T
    for v in val:
        sel = [slice(max(0, i), j+1) for i, j in zip(
                v - 1, v + 1)]


# sel = [slice(max(s, 0), e + 1) for s, e in zip(
#                     min_pos - np.array(cfg['fov_shape'])[:-1] // 2,
#                     max_pos + np.array(cfg['fov_shape'])[:-1] // 2)]
    return near, val



if __name__ == "__main__":
    from skimage import io
    # img_path = r"C:\Users\Administrator\Desktop\check_data\utils_neural\009764.png"
    img_path = r"H:\temp\realData\fov64_DF\5450_35600_4150_pred\5450_35600_4150_pred_81.jpg"
    a = io.imread(img_path, as_gray=True)
    a = (a[:300, :300]>0.5).astype(int)
    

    import time
    # from skimage.external import tifffile
    # tif_path = r"C:\Users\Administrator\Desktop\realData\ins_gt(1).tif"
    # a = tifffile.imread(tif_path)

    # a = np.zeros((10, 10, 10))
    # a[3] = 1
    # a[2:9, 3:5, 4:6] = 1

    # a = np.zeros((100, 100))
    # a[23:, 78:81] = 1
    # a[13:, 23:28] = 1

    distance, ind = ndimage.distance_transform_edt(a, return_indices=True)
    df = direct_field3D(a)

    gl0 = ndimage.gaussian_laplace(df[0,...], sigma=1)
    gl1 = ndimage.gaussian_laplace(df[1,...], sigma=1)
    sq = np.abs(gl0) + np.abs(gl1)

    import matplotlib.pyplot as plt
    gl0_ = (gl0 < -0.1).astype(int)
    gl1_ = (gl1 < -0.1).astype(int)
    gl_sq = (sq > 0.4).astype(int)

    # plt.imshow(np.stack([gl0_, gl1_]), cmap=plt.cm.gray)
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(gl0_)
    axs[1].imshow(gl1_)
    axs[2].imshow(gl_sq)
    plt.show()

    s = time.time()
    df, vis = direct_field3D(a)
    near, val = skeletonfromDF(df, a)
    # tifffile.imsave(r"C:\Users\Administrator\Desktop\realData\vis.tif", vis.astype(int))
    # print("Shape:{}, time: {}s".format(a.shape, time.time()-s))

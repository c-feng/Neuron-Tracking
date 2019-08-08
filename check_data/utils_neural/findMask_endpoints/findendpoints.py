import numpy as np
import os
from scipy.spatial.distance import pdist
from itertools import combinations


def findendpoints(mask):
    axis = np.stack( np.where(mask > 0), axis=0).transpose()

    axis0_min = np.min(axis[:, 0])
    axis0_max = np.max(axis[:, 0])

    axis1_min = np.min(axis[:, 1])
    axis1_max = np.max(axis[:, 1])
    if axis.shape[1] == 3:
        axis2_min = np.min(axis[:, 2])
        axis2_max = np.max(axis[:, 2])

    a = np.mean(list(filter(lambda i: axis0_min==i[0], axis.tolist())), axis=0)
    b = np.mean(list(filter(lambda i: axis0_max==i[0], axis.tolist())), axis=0)
    c = np.mean(list(filter(lambda i: axis1_min==i[1], axis.tolist())), axis=0)
    d = np.mean(list(filter(lambda i: axis1_max==i[1], axis.tolist())), axis=0)
    if axis.shape[1] == 3:
        e = np.mean(list(filter(lambda i: axis2_min==i[2], axis.tolist())), axis=0)
        f = np.mean(list(filter(lambda i: axis2_max==i[2], axis.tolist())), axis=0)

    if axis.shape[1] == 3:
        points = np.stack([a, b, c, d, e, f], axis=0)
    else:
        points = np.stack([a, b, c, d], axis=0)
    index = list(combinations(range(len(points)), 2))
    dist = pdist(points)
    idx = index[np.argmax(dist)]

    return points[idx, :], dist


if __name__ == "__main__":
        
    #mask = np.zeros((10,10))
    #mask[2:9, 5] = 1
    #
    #p, d = findendpoints(mask)
    from skimage.external import tifffile
    path = r"H:\temp\Syn_chaos_300_5_2_4_6_080_pred\Syn_chaos_300_5_2_4_6_080_pred_8.tif"
    tif = tifffile.imread(path)

    p, _ = findendpoints(tif)
    p = p.astype(int)

    tif[p[:, 0], p[:, 1], p[:, 2]] = 100

    out_dir = r"C:\Users\Administrator\Desktop\eval0\ins_mask_fixsplit_0"

    tifffile.imsave(os.path.join(out_dir, "endpoint.tif"), 100*(tif==100).astype(np.float16))
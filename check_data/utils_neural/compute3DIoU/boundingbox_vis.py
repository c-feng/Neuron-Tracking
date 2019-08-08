from skimage.external import tifffile
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import numpy as np
import os.path as osp
from skimage.measure import label as Label


FILE_PATH = "E:/6950_34600_4150/6950_34600_4150_26_pred.tif"

def bbox_vis(file_path=FILE_PATH):
    path, name = osp.split(file_path)
    tif = tifffile.imread(file_path)

    # # 转坐标
    # idx = np.where(tif > 0)
    # idx = np.stack(idx, axis=0).transpose()

    # # MeanShift 聚类
    # clustering = MeanShift(bandwidth=12).fit(idx)
    # label = np.unique(clustering.labels_)
    # n_clusters_ = len(label)

    # 连通域 完成聚类
    clustering = Label(tif)
    n_clusters_ = len(np.unique(clustering)) - 1

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    cls = {}
    cls_idx = {}
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = clustering.labels_ == k
    #     zeros = np.zeros(shape=(80, 80, 80))
    #     zeros[idx[my_members, 0], idx[my_members, 1], idx[my_members, 2]] = k + 1
    #     cls[k] = zeros  # 聚类结果
    #     cls_idx[k] = idx[my_members]  # 聚类的坐标

    #     ax.scatter(idx[my_members, 0], idx[my_members, 1], idx[my_members, 2],
    #             c=col)

    # plt.show()

    for k, col in zip(range(n_clusters_), colors):
        idx = np.where(clustering==(k+1))
        idx = np.stack(idx, axis=0).transpose()
        cls[k] = (clustering==(k+1)).astype(np.int)
        cls_idx[k] = idx

        ax.scatter(idx[:, 0], idx[:, 1], idx[:, 2], c=col)

    plt.show()

    # 找出左上右下顶点
    coords = []
    for k in range(n_clusters_):
        # 转坐标
        axis = np.stack( np.where(cls[k] > 0), axis=0).transpose()

        axis0_min = np.min(axis[:, 0])
        axis0_max = np.max(axis[:, 0])

        axis1_min = np.min(axis[:, 1])
        axis1_max = np.max(axis[:, 1])

        axis2_min = np.min(axis[:, 2])
        axis2_max = np.max(axis[:, 2])
        coords.append([axis0_min, axis1_min, axis2_min, axis0_max, axis1_max, axis2_max])

    # 用bounding box 包裹每个聚类
    bbox = {}
    for i in range(n_clusters_):

        axis0_min = coords[i][0]
        axis0_max = coords[i][3]

        axis1_min = coords[i][1]
        axis1_max = coords[i][4]

        axis2_min = coords[i][2]
        axis2_max = coords[i][5]
        
        w = axis0_max - axis0_min
        h = axis1_max - axis1_min
        l = axis2_max - axis2_min

        # cls[i][axis0_min:axis0_min+w, axis1_min, axis2_min] = i+1
        # cls[i][axis0_min, axis1_min:axis1_min+h, axis2_min] = i+1
        # cls[i][axis0_min, axis1_min, axis2_min:axis2_min+l] = i+1
        
        # cls[i][axis0_max-w:axis0_max, axis1_max, axis2_max] = i+1
        # cls[i][axis0_max, axis1_max-h:axis1_max, axis2_max] = i+1
        # cls[i][axis0_max, axis1_max, axis2_max-l:axis2_max] = i+1
        # cls_vis[i] = cls[i]
        temp = np.zeros(shape=(80,80,80))
        temp[axis0_min:axis0_min+w, axis1_min, axis2_min] = i+1
        temp[axis0_min, axis1_min:axis1_min+h, axis2_min] = i+1
        temp[axis0_min, axis1_min, axis2_min:axis2_min+l] = i+1
        
        temp[axis0_max-w:axis0_max, axis1_max, axis2_max] = i+1
        temp[axis0_max, axis1_max-h:axis1_max, axis2_max] = i+1
        temp[axis0_max, axis1_max, axis2_max-l:axis2_max] = i+1
        bbox[i] = temp

    cls_vis = {}
    for i in range(n_clusters_):
        temp = (np.array(cls[i]) > 0).astype(np.int16) + (np.array(bbox[i]) > 0) * 20
        cls_vis[i] = temp 
    
    bbox_vis = np.zeros(shape=(80,80,80))
    for i in range(n_clusters_):
        bbox_vis += cls_vis[i]
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # axis = np.stack( np.where(bbox_vis > 0), axis=0).transpose()
    # ax.scatter(axis[, 0], axis[, 1], axis[, 2])
    bbox_vis = bbox_vis * 100

    output_name = name.split('.')[0] + "_bboxvis.tif"
    tifffile.imsave(osp.join(path, output_name), bbox_vis.astype(np.int16))
    print("{} have been saved in {}".format(output_name, path))

if __name__ == "__main__":
    bbox_vis()
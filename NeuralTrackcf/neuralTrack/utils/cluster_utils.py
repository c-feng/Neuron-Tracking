import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.cluster import MeanShift
from skimage.morphology import remove_small_objects

from ..finchpy.finch import finch

def emb_meanshift(emb, mask, bandwidth = 0.6, thres = 100):
    """emb h * w * d * f
       mask h * w * d"""
    coords = np.array(np.where(mask > 0)).transpose()
    emb_ = emb[mask > 0]
    clusters_ = MeanShift(bandwidth = bandwidth).fit(emb_)
    labels_ = clusters_.labels_

    labels_array = np.array(labels_)
    coords_array = np.array(coords)

    ins_labels_array = np.unique(labels_)
    ins = np.zeros_like(mask, dtype = int)

    for i in ins_labels_array:
        #if np.sum(labels_array == i) < thres:
        coords_sel = coords_array[labels_array == i]
        ins[coords_sel[:, 0], coords_sel[:, 1], coords_sel[:, 2]] = i + 1
    return ins, ins_labels_array + 1

def emb_finch(emb, mask):
    """emb h * w * d * f
       mask h * w * d"""
    coords = np.array(np.where(mask > 0)).transpose()
    emb_ = emb[mask > 0]
    c, num_cluster = finch(emb_, [], 1)#c: n * N num_cluster N

    ins_list = []
    labels_list = []
    coords_array = np.array(coords)
    for i, num_label in enumerate(num_cluster):
        ins = np.zeros_like(mask, dtype = int)
        labels_array = c[:, i]
        for label in np.arange(num_label):
            coords_sel = coords_array[labels_array == label]
            ins[coords_sel[:, 0], coords_sel[:, 1], coords_sel[:, 2]] = label + 1
        ins_list.append(ins)
        labels_list.append(np.arange(num_label) + 1)
    return ins_list, labels_list



def PCA(data, k=2):
    # preprocess the data
    X = torch.from_numpy(data)
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k]).numpy()

def emb_visual(emb, ins, fpath):
    """emb h * w * d * f
       ins h * w * d"""
    labels = np.unique(ins[ins > 0])
    X = emb[ins > 0]
    y = ins[ins > 0]
    X_PCA = PCA(X, k = 2)
    print("x_pca, x , y",X.shape, y.shape, X_PCA.shape)
    plt.figure() 
    for label in labels:
        plt.scatter(X_PCA[y == label, 0], X_PCA[y == label, -1], label)
    plt.title('num of objects is {}'.format(len(labels)))
    plt.savefig(fpath)
    plt.close()





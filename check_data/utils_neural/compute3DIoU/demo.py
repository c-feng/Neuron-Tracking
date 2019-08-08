from skimage.external import tifffile
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import numpy as np
from skimage.measure import label as Label

tif_path = "E:/6950_34600_4150/6950_34600_4150_0_pred.tif"
tif = tifffile.imread(tif_path)

idx = np.where(tif > 0)

idx = np.stack(idx, axis=0).transpose()

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.scatter(idx[:, 0], idx[:, 1], idx[:, 2])
#ax.scatter(idx[:, 0], idx[:, 1], idx[:, 2])

clustering = MeanShift(bandwidth=7).fit(idx)
label = np.unique(clustering.labels_)
n_clusters_ = len(label)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
cls = {}
cls_idx = {}

for k, col in zip(range(n_clusters_), colors):
    my_members = clustering.labels_ == k
    zeros = np.zeros(shape=(80, 80, 80))
    zeros[idx[my_members, 0], idx[my_members, 1], idx[my_members, 2]] = k + 1
    cls[k] = zeros
    cls_idx[k] = idx[my_members]

    ax.scatter(idx[my_members, 0], idx[my_members, 1], idx[my_members, 2],
               c=col)

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

# for i in range(n_clusters_):
i = 1
ax.scatter(cls_idx[i][:, 0], cls_idx[i][:, 1], cls_idx[i][:, 2])

k = 1
axis = np.stack( np.where(cls[k] > 0), axis=0).transpose()

axis0_min = np.min(axis[:, 0])
axis0_max = np.max(axis[:, 0])

axis1_min = np.min(axis[:, 1])
axis1_max = np.max(axis[:, 1])

axis2_min = np.min(axis[:, 2])
axis2_max = np.max(axis[:, 2])

ax.scatter(axis0_min, axis1_min, axis2_min, marker='x', s=100)
ax.scatter(axis0_max, axis1_max, axis2_max, marker='x', s=100)

w = axis0_max - axis0_min
h = axis1_max - axis1_min
l = axis2_max - axis2_min

cls_vis = {}
for i in range(n_clusters_):
    cls[i][axis0_min:axis0_min+w, axis1_min, axis2_min] = i+1
    cls[i][axis0_min, axis1_min:axis1_min+h, axis2_min] = i+1
    cls[i][axis0_min, axis1_min, axis2_min:axis2_min+l] = i+1
    
    cls[i][axis0_max-w:axis0_max, axis1_max, axis2_max] = i+1
    cls[i][axis0_max, axis1_max-h:axis1_max, axis2_max] = i+1
    cls[i][axis0_max, axis1_max, axis2_max-l:axis2_max] = i+1
    cls_vis[i] = cls[i]

fig = plt.figure()
ax = fig.gca(projection='3d')

i = 2
axis_vis = np.stack( np.where(cls_vis[k] > 0), axis=0).transpose()

# for i in range(n_clusters_):
# ax.scatter(cls_idx[i][:, 0], cls_idx[i][:, 1], cls_idx[i][:, 2])
ax.scatter(axis_vis[:, 0], axis_vis[:, 1], axis_vis[:, 2])



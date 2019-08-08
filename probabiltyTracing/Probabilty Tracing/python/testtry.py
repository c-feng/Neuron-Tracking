import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from libsmop import *
import pdb

def load_swc(path):
    with open(path, 'r') as f:
        lines = csv.reader(f, delimiter=' ')
        lines = list(lines)
    lines = np.array([[float(i) for i in l] for l in lines]).astype(np.float16)
    return lines

a = np.random.randint(0, 6, (1,5))
b = np.random.randint(0, 6, (1,5))

# a = zeros(3, 10)
a = matlabarray(a)
print(a)
print(type(a))

# for i in arange(1, size(a, 2)).reshape(-1):
#     print(a[:, i])
print(a[:, 1:4])

print(a[5])

c = concat([a,b])
print(c)

path = r"D:\cf\Projects\Probabilty Tracing"

PointSWC=load_swc(os.path.join(path, '1.swc'))
PointSWC1=load_swc(os.path.join(path, '7.swc'))
PointSWC2=load_swc(os.path.join(path, '24.swc'))
PointSWC3=load_swc(os.path.join(path, '32.swc'))
PointSWC4=load_swc(os.path.join(path, '33.swc'))

Points = np.concatenate([PointSWC[:, [3, 2, 4]], PointSWC1[:, [3, 2, 4]], 
                         PointSWC2[:, [3, 2, 4]], PointSWC3[:, [3, 2, 4]],
                         PointSWC4[:, [3, 2, 4]]])

BinaryXX3 = np.zeros((512, 512, 77))
kk = 0
dataP = np.zeros([int(8e4), 4])
for point in Points:
    Idexx = range(int(np.maximum(round(point[0]-2), 0)), int(np.minimum(round(point[0]+2), 512)))
    Ideyy = range(int(np.maximum(round(point[1]-2), 0)), int(np.minimum(round(point[1]+2), 512)))
    Idezz = range(int(np.maximum(round(point[2]-2), 0)), int(np.minimum(round(point[2]+2), 77)))

    for ii in Idexx:
        for jj in Ideyy:
            for ij in Idezz:
                if BinaryXX3[ii, jj, ij] == 0:
                    dataP[kk, :] = [ii, jj, ij, 1]
                    kk += 1
                    BinaryXX3[ii, jj, ij] = 1


# axs = plt.subplot()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(Points[:, 0], Points[:, 1], Points[:, 2], marker='x', c='red')
ax.scatter(dataP[:, 0], dataP[:, 1], dataP[:, 2])
plt.show()

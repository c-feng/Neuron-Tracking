import numpy as np
import os
import csv
from skimage.external import tifffile
from libsmop import matlabarray, round

from RegiongrowModify import RegiongrowModify
from PeakdensityCenterPointMM import PeakdensityCenterPointMM
from PeakdensityClusterTrace import PeakdensityClusterTrace
from RejionConnctionsModify import RejionConnctionsModify
from ExtracConnectedCurvesPopulation import ExtracConnectedCurvesPopulation

import pdb

def load_swc(path):
    with open(path, 'r') as f:
        lines = csv.reader(f, delimiter=' ')
        lines = list(lines)
    lines = np.array([[float(i) for i in l] for l in lines]).astype(np.float16)
    return lines

path = "/media/fcheng/probabiltyTracing/Probabilty Tracing/"
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
                if BinaryXX3[ii, jj, ij]==0:
                    dataP[kk, :] = [ii, jj, ij, 1]
                    kk += 1
                    BinaryXX3[ii, jj, ij] = 1


BinaryXX3 = matlabarray(BinaryXX3)


# tif_path = r"D:\cf\Projects\Probabilty Tracing\python\ins_gt_vis.tif"
# tif = tifffile.imread(tif_path)

# datap_sel = np.random.choice(np.sum(tif>0), size=1)
# datap = matlabarray(np.stack(np.where(tif), axis=0)[:, datap_sel])
datap = matlabarray([[342], [256], [66]])
# datap = matlabarray([[12], [67], [61]])

# BinaryXX3 = matlabarray((tif>0).astype(int))
XX3 = BinaryXX3
print("1 ......")
pdb.set_trace()

# BinaryXX3二值化图, 1000迭代上限, XX3原图
Dataset, datacell, _ = RegiongrowModify(datap, BinaryXX3, 20, XX3, 1)
print("RegiongrowModify Finished")
pdb.set_trace()
Datap, _, _ = PeakdensityCenterPointMM(datacell, XX3.shape)
print("PeakdensityCenterPointMM Finished")
pdb.set_trace()
CenterSet, PartPointSet = PeakdensityClusterTrace(Datap)
print("PeakdensityClusterTrace Finished")
pdb.set_trace()
CMatrix = RejionConnctionsModify(CenterSet, PartPointSet)
print("RejionConnctionsModify Finished")
pdb.set_trace()
Addcurve, LevelCurve = ExtracConnectedCurvesPopulation(CMatrix, CenterSet)
pdb.set_trace()
print("ExtracConnectedCurvesPopulation Finished")
print(Addcurve)


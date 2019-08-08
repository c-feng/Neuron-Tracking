# Generated with SMOP  0.41
from libsmop import *
# ExtracConnectedCurvesPopulationSub.m
from Longestpath import Longestpath
from ExtracConnectedCurvesSub import ExtracConnectedCurvesSub
import numpy as np

import pdb

    
# @function
def ExtracConnectedCurvesPopulationSub(ConnecCell=None,CenterPoints=None,LevelId=None,PointId=None,*args,**kwargs):
    # varargin = ExtracConnectedCurvesPopulationSub.varargin
    # nargin = ExtracConnectedCurvesPopulationSub.nargin

    CurvesSet=cell(10,160)
# ExtracConnectedCurvesPopulationSub.m:3
    CenterPoints0=copy(CenterPoints)
# ExtracConnectedCurvesPopulationSub.m:4
    Nums=size(CenterPoints0,2)
# ExtracConnectedCurvesPopulationSub.m:5
    for i in arange(1,Nums).reshape(-1):
        CurrMatrix=CenterPoints[i].reshape([4, -1])
# ExtracConnectedCurvesPopulationSub.m:8
        LLs=zeros(1,size(CurrMatrix,2))
        # CurrMatrix = CurrMatrix.reshape([4, -1])
# ExtracConnectedCurvesPopulationSub.m:9
        # CenterPoints0[i]=concat([[CurrMatrix[arange(1,3), :]],[LLs]])
        try:
            CenterPoints0[i]=matlabarray(np.concatenate([CurrMatrix[arange(1,3), :].reshape([3, -1]), LLs]))
        except IndexError :
            pdb.set_trace()
        except ValueError:
            pdb.set_trace()
# ExtracConnectedCurvesPopulationSub.m:10
    
    Curves,MatrixLabel,CenterPointss=Longestpath(ConnecCell,CenterPoints0,LevelId,PointId,nargout=3)
# ExtracConnectedCurvesPopulationSub.m:14
    CurvesSet[1,1]=Curves
# ExtracConnectedCurvesPopulationSub.m:15
    LableSet = cell(1, 1)
    LableSet[1]=MatrixLabel
# ExtracConnectedCurvesPopulationSub.m:17
    #MatrixLabel
#MatrixLabel(1,:) the node level of tree structure 
#MatrixLabel(2,:) center point in each node
#MatrixLabel(3,:) the number of points in the cluster
    
    # #------------------
    MaxNumCurves=zeros(1,10)
# ExtracConnectedCurvesPopulationSub.m:25
    for i in arange(2,10).reshape(-1):
        DownLevelMatrixLable,Curves00,CenterPointss=ExtracConnectedCurvesSub(LableSet,ConnecCell,CenterPointss,nargout=3)
# ExtracConnectedCurvesPopulationSub.m:28
        LableSet=copy(DownLevelMatrixLable)
# ExtracConnectedCurvesPopulationSub.m:29
        if isempty(Curves00) == 0:
            Numss=size(Curves00,2)
# ExtracConnectedCurvesPopulationSub.m:31
            MaxNumCurves[i]=Numss
# ExtracConnectedCurvesPopulationSub.m:32
            for jj in arange(1,Numss).reshape(-1):
                CurvesSet[i,jj]=Curves00[jj]
# ExtracConnectedCurvesPopulationSub.m:34
        else:
            break
    
    MaxNum=max(np.max(MaxNumCurves),1)
# ExtracConnectedCurvesPopulationSub.m:40
    CurvesSets0=cell(1,100)
# ExtracConnectedCurvesPopulationSub.m:41
    kk=0
# ExtracConnectedCurvesPopulationSub.m:42
    for i in arange(1,10).reshape(-1):
        for j in arange(1,MaxNum).reshape(-1):
            if isempty(CurvesSet[i,j]) == 0:
                kk=kk + 1
# ExtracConnectedCurvesPopulationSub.m:47
                CurvesSets0[kk]=CurvesSet[i,j]
# ExtracConnectedCurvesPopulationSub.m:48
    
    if kk > 0:
        CurvesSets0=CurvesSets0[arange(1,kk)]
# ExtracConnectedCurvesPopulationSub.m:54
    else:
        CurvesSets0=[]
# ExtracConnectedCurvesPopulationSub.m:57
    
    #-----------------------------------
    return CurvesSets0, CurvesSet, CenterPointss
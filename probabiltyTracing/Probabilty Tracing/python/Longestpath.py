# Generated with SMOP  0.41
from libsmop import *
# Longestpath.m
from Longestpathsub import Longestpathsub

    
# @function
def Longestpath(ConnecCell=None,CenterPoints=None,LevelId=None,PointId=None,*args,**kwargs):
    # varargin = Longestpath.varargin
    # nargin = Longestpath.nargin

    CurrVec=CenterPoints[LevelId]
# Longestpath.m:3
    StartNode=CurrVec[:, PointId]
# Longestpath.m:4
    CurrVec[4,PointId]=1
# Longestpath.m:5
    CenterPoints[LevelId]=CurrVec
# Longestpath.m:6
    TotalLevel=size(ConnecCell,2)
# Longestpath.m:8
    Curves=zeros(3,TotalLevel)
# Longestpath.m:9
    ExtraLabel=zeros(3,TotalLevel)
# Longestpath.m:10
    kk=1
# Longestpath.m:11
    Curves[:, kk]=StartNode[arange(1,3)]
# Longestpath.m:12
    ExtraLabel[1,kk]=LevelId
# Longestpath.m:13
    ExtraLabel[2,kk]=PointId
# Longestpath.m:14
    ExtraLabel[3,kk]=0
# Longestpath.m:15
    for ij in arange(LevelId + 1,TotalLevel + 1,1).reshape(-1):
        SS=ConnecCell[ij - 1]
# Longestpath.m:17
        CurrVec=CenterPoints[ij]
# Longestpath.m:18
        PointId0=Longestpathsub(SS,PointId,2,CurrVec[4, :])
# Longestpath.m:20
        if PointId0 != 0 and CurrVec[4,PointId0] == 0:
            CurrVec=CenterPoints[ij]
# Longestpath.m:23
            kk=kk + 1
# Longestpath.m:24
            Curves[:, kk]=CurrVec[arange(1,3),PointId0]
# Longestpath.m:25
            PointId=copy(PointId0)
# Longestpath.m:26
            ExtraLabel[1,kk]=ij
# Longestpath.m:27
            ExtraLabel[2,kk]=PointId0
# Longestpath.m:28
            ExtraLabel[3,kk]=SS(2,PointId0)
# Longestpath.m:29
            CurrVec[4,PointId0]=1
# Longestpath.m:30
            CenterPoints[ij]=CurrVec
# Longestpath.m:31
        else:
            #TerminalId=0;
            break
    
    #end
    
    if kk > 1:
        Curves=Curves[:, arange(1,kk)]
# Longestpath.m:42
        ExtraLabel=ExtraLabel[:, arange(1,kk)]
# Longestpath.m:43
    else:
        Curves=matlabarray([])
# Longestpath.m:45
        ExtraLabel=matlabarray([])
# Longestpath.m:46
    
    #end
    return Curves, ExtraLabel, CenterPoints
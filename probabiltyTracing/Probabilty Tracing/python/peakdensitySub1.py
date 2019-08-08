# Generated with SMOP  0.41
from libsmop import *
# peakdensitySub1.m
import numpy as np

    
# @function
def peakdensitySub1(Image0=None,ImageLabel=None,Imagess=None,NeighborSize=None,Position=None,datalabel=None,*args,**kwargs):
    # varargin = peakdensitySub1.varargin
    # nargin = peakdensitySub1.nargin

    nx,ny,nz=size(Image0,nargout=3)
# peakdensitySub1.m:2
    Indexx=arange(max(Position[1] - NeighborSize,1),min(Position[1] + NeighborSize,nx))
# peakdensitySub1.m:3
    Indeyy=arange(max(Position[2] - NeighborSize,1),min(Position[2] + NeighborSize,ny))
# peakdensitySub1.m:4
    Indezz=arange(max(Position[3] - NeighborSize,1),min(Position[3] + NeighborSize,nz))
# peakdensitySub1.m:5
    aa=Image0[Position[1],Position[2],Position[3]]
# peakdensitySub1.m:6
    kk=0
# peakdensitySub1.m:7
    CurrVec=dot((dot(2,NeighborSize) + 1),ones(1,dot(dot(length(Indexx),length(Indeyy)),length(Indezz))))
# peakdensitySub1.m:8
    for i in arange(1,length(Indexx)).reshape(-1):
        for j in arange(1,length(Indeyy)).reshape(-1):
            for ij in arange(1,length(Indezz)).reshape(-1):
                if aa < Image0[Indexx[i],Indeyy[j],Indezz[ij]] and ImageLabel[Indexx[i],Indeyy[j],Indezz[ij]] == datalabel:
                    kk=kk + 1
# peakdensitySub1.m:13
                    CurrVec[1,kk]=np.linalg.norm(concat([Indexx[i],Indeyy[j],Indezz[ij]]) - Position, 2)
# peakdensitySub1.m:14
                    CurrVec[2,kk]=Imagess[Indexx[i],Indeyy[j],Indezz[ij]]
# peakdensitySub1.m:15
    
    if kk > 0:
        # minidist0,Indexx=min(CurrVec[1, :], nargout=2)
        minidist0 = np.min(CurrVec[1, :])
        Indexx  = np.argmin(CurrVec[1, :])
# peakdensitySub1.m:21
        ConnecIndex=CurrVec[2, Indexx]
# peakdensitySub1.m:22
        minidist0=minidist0 / (dot(2,NeighborSize) + 1)
# peakdensitySub1.m:23
    else:
        minidist0=1
# peakdensitySub1.m:25
        ConnecIndex=Imagess[Position[1],Position[2],Position[3]]
# peakdensitySub1.m:26
    return minidist0,ConnecIndex
    
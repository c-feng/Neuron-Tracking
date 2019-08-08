# Generated with SMOP  0.41
from libsmop import *
# NeighborExtract.m
import numpy as np
import pdb
    
# @function
def NeighborExtract(OrigImg=None,LabelImg=None,dataP=None,NeighborS=None,*args,**kwargs):
    # varargin = NeighborExtract.varargin
    # nargin = NeighborExtract.nargin

    Nx,Ny,Nz=size(OrigImg,nargout=3)
# NeighborExtract.m:3
    dataP=round(dataP)
# NeighborExtract.m:4
    Indexx=arange(max(dataP[1] - NeighborS,1),min(dataP[1] + NeighborS,Nx))
# NeighborExtract.m:5
    Indeyy=arange(max(dataP[2] - NeighborS,1),min(dataP[2] + NeighborS,Ny))
# NeighborExtract.m:6
    Indezz=arange(max(dataP[3] - NeighborS,1),min(dataP[3] + NeighborS,Nz))
# NeighborExtract.m:7
    Label=LabelImg[dataP[1],dataP[2],dataP[3]]
# NeighborExtract.m:9
    dataset=zeros(4,dot(dot(length(Indexx),length(Indeyy)),length(Indezz)))
# NeighborExtract.m:10
    kk=0
# NeighborExtract.m:11
    for ii in arange(1,length(Indexx)).reshape(-1):
        for jj in arange(1,length(Indeyy)).reshape(-1):
            for ij in arange(1,length(Indezz)).reshape(-1):
                if LabelImg[Indexx[ii],Indeyy[jj],Indezz[ij]] == Label:
                    kk=kk + 1
# NeighborExtract.m:17
                    dataset[arange(1,3),kk]=concat([Indexx[ii],Indeyy[jj],Indezz[ij]])[:, 0]
# NeighborExtract.m:18
                    dataset[4,kk]=OrigImg[Indexx[ii],Indeyy[jj],Indezz[ij]]
# NeighborExtract.m:19
    
    if kk > 0:
        dataset=dataset[:, arange(1,kk)].reshape([4, -1])
# NeighborExtract.m:26
        ww=zeros(1,kk)
# NeighborExtract.m:27
        for ii in arange(1,kk).reshape(-1):
            # ww[ii]=exp(- norm(dataset(arange(1,3),ii) - dataP(arange(1,3))))
            try:
                ww[0, ii]=exp(-np.linalg.norm(dataset[arange(1,3),ii] - dataP[arange(1,3)], 2))
            except IndexError:
                pdb.set_trace()

# NeighborExtract.m:29
        aver0=(dot(ww, dataset[4, :].T)) / sum(ww)
# NeighborExtract.m:32
        aver0=dot(0.2,aver0) + dot(0.8,sum(dataset[4, :])) / (dot(dot(length(Indexx),length(Indeyy)),length(Indezz)))
# NeighborExtract.m:33
    else:
        dataset=[]
# NeighborExtract.m:35
    
    return aver0, dataset
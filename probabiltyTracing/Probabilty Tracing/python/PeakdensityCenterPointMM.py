# Generated with SMOP  0.41
from libsmop import *
# PeakdensityCenterPointMM.m
from NeighborExtract import NeighborExtract
from peakdensitySub1 import peakdensitySub1

import pdb
# @function
def PeakdensityCenterPointMM(PointsSet=None,ImgSize=None,*args,**kwargs):
    # varargin = PeakdensityCenterPointMM.varargin
    # nargin = PeakdensityCenterPointMM.nargin

    Numx=length(PointsSet)
# PeakdensityCenterPointMM.m:2
    kk=0
# PeakdensityCenterPointMM.m:3
    CurrVec=zeros(5,200000)
# PeakdensityCenterPointMM.m:4
    jj=0
# PeakdensityCenterPointMM.m:5
    for i in arange(1,Numx).reshape(-1):
        if isempty(PointsSet[i]) == 0:
            kk=kk + 1
# PeakdensityCenterPointMM.m:8
            Currdata=PointsSet[i]
# PeakdensityCenterPointMM.m:9
            try:
                CurrVec[arange(1,4),arange(jj + 1,jj + size(Currdata,2))]=Currdata[0:4, :] if Currdata.shape[1]>1 else Currdata.reshape([1, 4])
            except ValueError:
                pdb.set_trace()
# PeakdensityCenterPointMM.m:10
            CurrVec[5,arange(jj + 1,jj + size(Currdata,2))]=kk
# PeakdensityCenterPointMM.m:11
            jj=jj + size(Currdata,2)
# PeakdensityCenterPointMM.m:12
        else:
            break
    
    CurrVec=CurrVec[:, arange(1,jj)]
# PeakdensityCenterPointMM.m:17
    Datap=zeros(7,jj)
# PeakdensityCenterPointMM.m:18
    Nx=ImgSize[0]
# PeakdensityCenterPointMM.m:19
    Ny=ImgSize[1]
# PeakdensityCenterPointMM.m:20
    Nz=ImgSize[2]
# PeakdensityCenterPointMM.m:21
    CurrMatrix0=zeros(Nx,Ny,Nz)
# PeakdensityCenterPointMM.m:23
    CurrMatrix1=zeros(Nx,Ny,Nz)
# PeakdensityCenterPointMM.m:24
    CurrMatrix2=zeros(Nx,Ny,Nz)
# PeakdensityCenterPointMM.m:25
    for i in arange(1,size(CurrVec,2)).reshape(-1):
        a=CurrVec[:, i]
# PeakdensityCenterPointMM.m:28
        CurrMatrix0[a[1],a[2],a[3]]=max(a[4],1)
# PeakdensityCenterPointMM.m:29
        CurrMatrix1[a[1],a[2],a[3]]=a[5]
# PeakdensityCenterPointMM.m:30
        CurrMatrix2[a[1],a[2],a[3]]=i
# PeakdensityCenterPointMM.m:31
    
    for kkmm in arange(1,5).reshape(-1):
        for ii in arange(1,size(CurrVec,2)).reshape(-1):
            a=CurrVec[:, ii]
# PeakdensityCenterPointMM.m:36
            aver0, _ =NeighborExtract(CurrMatrix0,CurrMatrix1,a,1)
# PeakdensityCenterPointMM.m:37
            CurrMatrix0[a[1],a[2],a[3]]=aver0
# PeakdensityCenterPointMM.m:38
    
    for ii in arange(1,size(Datap,2)).reshape(-1):
        Position=CurrVec[arange(1,3),ii]
# PeakdensityCenterPointMM.m:43
        datalabel=CurrVec[5,ii]
# PeakdensityCenterPointMM.m:44
        minidist0,connecIndex=peakdensitySub1(CurrMatrix0,CurrMatrix1,CurrMatrix2,5,Position,datalabel,nargout=2)
# PeakdensityCenterPointMM.m:45
        Datap[arange(1,3),ii]=Position
# PeakdensityCenterPointMM.m:47
        Datap[4,ii]=CurrMatrix0[Position[1],Position[2],Position[3]]
# PeakdensityCenterPointMM.m:48
        Datap[5,ii]=datalabel
# PeakdensityCenterPointMM.m:49
        Datap[6,ii]=minidist0
# PeakdensityCenterPointMM.m:50
        Datap[7,ii]=connecIndex
# PeakdensityCenterPointMM.m:51
    return Datap,CurrMatrix0,CurrMatrix1
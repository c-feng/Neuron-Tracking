# Generated with SMOP  0.41
from libsmop import *
# PeakdensityClusterpoint.m
from PeakdensityClusterpointsub import PeakdensityClusterpointsub
    
import pdb

# @function
def PeakdensityClusterpoint(Points=None,densityP=None,Connets=None,*args,**kwargs):
    # varargin = PeakdensityClusterpoint.varargin
    # nargin = PeakdensityClusterpoint.nargin

    #densityP=Points(4,:);
    peaksId=find(densityP > 0.9)
# PeakdensityClusterpoint.m:4
    PartPoint=cell(1,length(peaksId))
# PeakdensityClusterpoint.m:5
    try:
        Centerp=Points[arange(1,4),peaksId]
    except IndexError:
        pdb.set_trace()
# PeakdensityClusterpoint.m:6
    Num_points=size(Points,2)
# PeakdensityClusterpoint.m:7
    dataLabel=zeros(1,Num_points)
# PeakdensityClusterpoint.m:8
    for ii in arange(1,length(peaksId)).reshape(-1):
        dataLabel[peaksId[ii]]=- ii
# PeakdensityClusterpoint.m:11
    
    if length(peaksId) == 1:
        PartPoint[1]=Points
# PeakdensityClusterpoint.m:15
    
    if length(peaksId) > 1:
        for i in arange(1,Num_points).reshape(-1):
            if dataLabel[i] == 0:
                Pathdata,Idflag=PeakdensityClusterpointsub(i,Connets,dataLabel,nargout=2)
# PeakdensityClusterpoint.m:22
                dataLabel[Pathdata]=Idflag
# PeakdensityClusterpoint.m:23
        for jj in arange(1,length(peaksId)).reshape(-1):
            Idd=find(dataLabel == - jj)
# PeakdensityClusterpoint.m:28
            PartPoint[jj]=Points[:, Idd]
# PeakdensityClusterpoint.m:29
    return Centerp, PartPoint
    
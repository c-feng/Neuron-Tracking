# Generated with SMOP  0.41
from libsmop import *
# PeakdensityClusterTrace.m

    
@function
def PeakdensityClusterTrace(Datap=None,*args,**kwargs):
    varargin = PeakdensityClusterTrace.varargin
    nargin = PeakdensityClusterTrace.nargin

    #Datap(4,:)=1;
    LabelId=Datap(5,arange())
# PeakdensityClusterTrace.m:3
    Idvec=concat([0,find(diff(LabelId) > 0.5),length(LabelId)])
# PeakdensityClusterTrace.m:4
    CenterSet=cell(1,length(Idvec) - 1)
# PeakdensityClusterTrace.m:5
    PartPointSet=cell(1,length(Idvec) - 1)
# PeakdensityClusterTrace.m:6
    for i in arange(1,length(Idvec) - 1).reshape(-1):
        Indexx=arange(Idvec(i) + 1,Idvec(i + 1))
# PeakdensityClusterTrace.m:8
        Points=Datap(arange(1,4),Indexx)
# PeakdensityClusterTrace.m:9
        densityP=Datap(6,Indexx)
# PeakdensityClusterTrace.m:10
        Connets=Datap(7,Indexx) - Idvec(i)
# PeakdensityClusterTrace.m:11
        Centerp,PartPoint=PeakdensityClusterpoint(Points,densityP,Connets,nargout=2)
# PeakdensityClusterTrace.m:12
        CenterSet[i]=Centerp
# PeakdensityClusterTrace.m:15
        num0=size(PartPoint,2)
# PeakdensityClusterTrace.m:16
        PartPointSet[i]=cell(1,num0)
# PeakdensityClusterTrace.m:17
        for jj in arange(1,num0).reshape(-1):
            PartPointSet[i][jj]=PartPoint[jj]
# PeakdensityClusterTrace.m:19
    
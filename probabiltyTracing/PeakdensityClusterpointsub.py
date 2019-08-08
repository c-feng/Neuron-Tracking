# Generated with SMOP  0.41
from libsmop import *
# PeakdensityClusterpointsub.m

    
@function
def PeakdensityClusterpointsub(Index0=None,Connets=None,dataLabel=None,*args,**kwargs):
    varargin = PeakdensityClusterpointsub.varargin
    nargin = PeakdensityClusterpointsub.nargin

    Nums=length(dataLabel)
# PeakdensityClusterpointsub.m:3
    Pathdata=zeros(1,Nums)
# PeakdensityClusterpointsub.m:4
    Idflag=0
# PeakdensityClusterpointsub.m:5
    kk=1
# PeakdensityClusterpointsub.m:6
    Pathdata[kk]=Index0
# PeakdensityClusterpointsub.m:7
    while Idflag == 0 and kk < Nums + 1:

        Id=Connets(Index0)
# PeakdensityClusterpointsub.m:10
        if dataLabel(Id) == 0:
            kk=kk + 1
# PeakdensityClusterpointsub.m:12
            Pathdata[kk]=Id
# PeakdensityClusterpointsub.m:13
            Index0=copy(Id)
# PeakdensityClusterpointsub.m:14
        else:
            Idflag=dataLabel(Id)
# PeakdensityClusterpointsub.m:16

    
    Pathdata=Pathdata(arange(1,kk))
# PeakdensityClusterpointsub.m:19
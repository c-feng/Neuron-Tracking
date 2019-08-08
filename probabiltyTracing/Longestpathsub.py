# Generated with SMOP  0.41
from libsmop import *
# Longestpathsub.m

    
@function
def Longestpathsub(SS=None,PointId=None,Thre=None,Lablevec=None,*args,**kwargs):
    varargin = Longestpathsub.varargin
    nargin = Longestpathsub.nargin

    ax=0
# Longestpathsub.m:2
    Num0=size(SS,2)
# Longestpathsub.m:3
    for i in arange(1,Num0).reshape(-1):
        if SS(1,i) == PointId and SS(2,i) > ax and Lablevec(i) == 0:
            ax=SS(2,i)
# Longestpathsub.m:7
            PointId0=copy(i)
# Longestpathsub.m:8
    
    if ax < Thre:
        PointId0=0
# Longestpathsub.m:13
    
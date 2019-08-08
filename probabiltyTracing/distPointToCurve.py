# Generated with SMOP  0.41
from libsmop import *
# distPointToCurve.m

    
@function
def distPointToCurve(starP=None,curve=None,*args,**kwargs):
    varargin = distPointToCurve.varargin
    nargin = distPointToCurve.nargin

    Nums=size(curve,2)
# distPointToCurve.m:3
    disv=zeros(1,Nums)
# distPointToCurve.m:4
    for i in arange(1,Nums).reshape(-1):
        disv[i]=norm(starP - curve(arange(),i))
# distPointToCurve.m:6
    
    Minv1,Index1=min(disv,nargout=2)
# distPointToCurve.m:10
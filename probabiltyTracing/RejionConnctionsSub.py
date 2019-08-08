# Generated with SMOP  0.41
from libsmop import *
# RejionConnctionsSub.m

    
@function
def RejionConnctionsSub(ss=None,ss1=None,*args,**kwargs):
    varargin = RejionConnctionsSub.varargin
    nargin = RejionConnctionsSub.nargin

    #  Nums=size(ss1,1)
#  Numx=size(ss,1)
#  Nums=size(ss1,2);
#   Numx=size(ss,2);
    Nums=length(ss1)
# RejionConnctionsSub.m:6
    Numx=length(ss)
# RejionConnctionsSub.m:7
    Connevec=zeros(2,Nums)
# RejionConnctionsSub.m:8
    for i in arange(1,Nums).reshape(-1):
        Vess=zeros(1,Numx)
# RejionConnctionsSub.m:10
        data00=ss1[1]
# RejionConnctionsSub.m:11
        for j in arange(1,Numx).reshape(-1):
            Flag00=0
# RejionConnctionsSub.m:14
            if size(ss[j],2) > logical_and(1,size(ss1[i],2)) > 1:
                Flag00,InterSets=TwoPointsIntersection(ss[j],ss1[i],2,nargout=2)
# RejionConnctionsSub.m:16
            if Flag00 == 1:
                Vess[j]=size(InterSets,2)
# RejionConnctionsSub.m:19
        Maxv,MaxIndex=max(Vess,nargout=2)
# RejionConnctionsSub.m:22
        if Maxv > 0:
            Connevec[2,i]=Maxv
# RejionConnctionsSub.m:24
            Connevec[1,i]=MaxIndex
# RejionConnctionsSub.m:25
    
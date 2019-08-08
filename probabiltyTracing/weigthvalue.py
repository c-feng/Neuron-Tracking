# Generated with SMOP  0.41
from libsmop import *
# weigthvalue.m

    
@function
def weigthvalue(dataL1=None,L_XX3=None,*args,**kwargs):
    varargin = weigthvalue.varargin
    nargin = weigthvalue.nargin

    nxss=size(dataL1,2)
# weigthvalue.m:2
    aa_data=zeros(1,nxss)
# weigthvalue.m:3
    nx,ny,nz=size(L_XX3,nargout=3)
# weigthvalue.m:7
    for i in arange(1,nxss).reshape(-1):
        x=dataL1(1,i)
# weigthvalue.m:10
        y=dataL1(2,i)
# weigthvalue.m:11
        z=dataL1(3,i)
# weigthvalue.m:12
        dd=0
# weigthvalue.m:13
        ww=0
# weigthvalue.m:13
        idexx=max(min(round(x),nx),1)
# weigthvalue.m:14
        idexy=max(min(round(y),ny),1)
# weigthvalue.m:15
        idexz=max(min(round(z),nz),1)
# weigthvalue.m:16
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:18
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:19
        ww=ww + exp(w1)
# weigthvalue.m:20
        idexx=max(min(round(x + 1),nx),1)
# weigthvalue.m:22
        idexy=max(min(round(y),ny),1)
# weigthvalue.m:23
        idexz=max(min(round(z),nz),1)
# weigthvalue.m:24
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:25
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:26
        ww=ww + exp(w1)
# weigthvalue.m:27
        idexx=max(min(round(x - 1),nx),1)
# weigthvalue.m:29
        idexy=max(min(round(y),ny),1)
# weigthvalue.m:30
        idexz=max(min(round(z),nz),1)
# weigthvalue.m:31
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:32
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:33
        ww=ww + exp(w1)
# weigthvalue.m:34
        idexx=max(min(round(x),nx),1)
# weigthvalue.m:36
        idexy=max(min(round(y - 1),ny),1)
# weigthvalue.m:37
        idexz=max(min(round(z),nz),1)
# weigthvalue.m:38
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:39
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:40
        ww=ww + exp(w1)
# weigthvalue.m:41
        idexx=max(min(round(x),nx),1)
# weigthvalue.m:43
        idexy=max(min(round(y + 1),ny),1)
# weigthvalue.m:44
        idexz=max(min(round(z),nz),1)
# weigthvalue.m:45
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:46
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:47
        ww=ww + exp(w1)
# weigthvalue.m:48
        idexx=max(min(round(x),nx),1)
# weigthvalue.m:50
        idexy=max(min(round(y),ny),1)
# weigthvalue.m:51
        idexz=max(min(round(z) - 1,nz),1)
# weigthvalue.m:52
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:53
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:54
        ww=ww + exp(w1)
# weigthvalue.m:55
        idexx=max(min(round(x),nx),1)
# weigthvalue.m:57
        idexy=max(min(round(y),ny),1)
# weigthvalue.m:58
        idexz=max(min(round(z + 1),nz),1)
# weigthvalue.m:59
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:60
        dd=dd + dot(exp(w1),L_XX3(idexx,idexy,idexz))
# weigthvalue.m:61
        ww=ww + exp(w1)
# weigthvalue.m:62
        aa_data[i]=dd / (ww + 0.0001)
# weigthvalue.m:63
    
    
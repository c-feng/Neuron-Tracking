# Generated with SMOP  0.41
from libsmop import *
# weigthvalue.m
import pdb

    
# @function
def weigthvalue(dataL1=None,L_XX3=None,*args,**kwargs):
#     varargin = weigthvalue.varargin
#     nargin = weigthvalue.nargin

    nxss=size(dataL1,2)
# weigthvalue.m:2
    aa_data=zeros(1,nxss)
# weigthvalue.m:3
    nx,ny,nz=size(L_XX3,nargout=3)
# weigthvalue.m:7
    for i in arange(1,nxss).reshape(-1):
        x=dataL1[1,i]
# weigthvalue.m:10
        try:
            y=dataL1[2,i]
        except IndexError:
            pdb.set_trace()
# weigthvalue.m:11
        z=dataL1[3,i]
# weigthvalue.m:12
        dd=0
# weigthvalue.m:13
        ww=0
# weigthvalue.m:13
        idexx=int(max(min(round(x),nx),1))
# weigthvalue.m:14
        idexy=int(max(min(round(y),ny),1))
# weigthvalue.m:15
        idexz=int(max(min(round(z),nz),1))
# weigthvalue.m:16
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:18
        try:
            dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
        except IndexError:
            pdb.set_trace()
# weigthvalue.m:19
        ww=ww + exp(w1)
# weigthvalue.m:20
        idexx=int(max(min(round(x + 1),nx),1))
# weigthvalue.m:22
        idexy=int(max(min(round(y),ny),1))
# weigthvalue.m:23
        idexz=int(max(min(round(z),nz),1))
# weigthvalue.m:24
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:25
        dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
# weigthvalue.m:26
        ww=ww + exp(w1)
# weigthvalue.m:27
        idexx=int(max(min(round(x - 1),nx),1))
# weigthvalue.m:29
        idexy=int(max(min(round(y),ny),1))
# weigthvalue.m:30
        idexz=int(max(min(round(z),nz),1))
# weigthvalue.m:31
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:32
        dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
# weigthvalue.m:33
        ww=ww + exp(w1)
# weigthvalue.m:34
        idexx=int(max(min(round(x),nx),1))
# weigthvalue.m:36
        idexy=int(max(min(round(y - 1),ny),1))
# weigthvalue.m:37
        idexz=int(max(min(round(z),nz),1))
# weigthvalue.m:38
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:39
        dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
# weigthvalue.m:40
        ww=ww + exp(w1)
# weigthvalue.m:41
        idexx=int(max(min(round(x),nx),1))
# weigthvalue.m:43
        idexy=int(max(min(round(y + 1),ny),1))
# weigthvalue.m:44
        idexz=int(max(min(round(z),nz),1))
# weigthvalue.m:45
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:46
        dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
# weigthvalue.m:47
        ww=ww + exp(w1)
# weigthvalue.m:48
        idexx=int(max(min(round(x),nx),1))
# weigthvalue.m:50
        idexy=int(max(min(round(y),ny),1))
# weigthvalue.m:51
        idexz=int(max(min(round(z) - 1,nz),1))
# weigthvalue.m:52
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:53
        dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
# weigthvalue.m:54
        ww=ww + exp(w1)
# weigthvalue.m:55
        idexx=int(max(min(round(x),nx),1))
# weigthvalue.m:57
        idexy=int(max(min(round(y),ny),1))
# weigthvalue.m:58
        idexz=int(max(min(round(z + 1),nz),1))
# weigthvalue.m:59
        w1=dot(0.05,max(dot(- 2,((x - idexx) ** 2 + (y - idexy) ** 2 + (dot(2,z) - dot(2,idexz)) ** 2)),- 6))
# weigthvalue.m:60
        try:
            dd=dd + dot(exp(w1),L_XX3[idexx,idexy,idexz])
        except IndexError:
            pdb.set_trace()
# weigthvalue.m:61
        ww=ww + exp(w1)
# weigthvalue.m:62
        aa_data[i]=dd / (ww + 0.0001)
# weigthvalue.m:63
    return aa_data
    
# Generated with SMOP  0.41
from libsmop import *
# TwoPointsIntersection.m

    
@function
def TwoPointsIntersection(Seta=None,Setb=None,R=None,*args,**kwargs):
    varargin = TwoPointsIntersection.varargin
    nargin = TwoPointsIntersection.nargin

    MergeSet=concat([Seta,Setb])
# TwoPointsIntersection.m:2
    minV=min(MergeSet.T)
# TwoPointsIntersection.m:4
    minV=minV.T
# TwoPointsIntersection.m:5
    maxV=max(MergeSet.T)
# TwoPointsIntersection.m:6
    maxV=maxV.T
# TwoPointsIntersection.m:7
    MinCoord=max(minV - dot(2,R),0)
# TwoPointsIntersection.m:8
    MaxCoord=maxV + dot(2,R) + 1
# TwoPointsIntersection.m:9
    LabelMatrix=zeros(MaxCoord(1) - MinCoord(1),MaxCoord(2) - MinCoord(2),MaxCoord(3) - MinCoord(3))
# TwoPointsIntersection.m:10
    NumPointsSeta=size(Seta,2)
# TwoPointsIntersection.m:11
    NumPointsSetb=size(Setb,2)
# TwoPointsIntersection.m:12
    for i in arange(1,NumPointsSeta).reshape(-1):
        aax=Seta(arange(),i) - minV + 1
# TwoPointsIntersection.m:14
        LabelMatrix[aax(1),aax(2),aax(3)]=1
# TwoPointsIntersection.m:15
    
    Labelvec=zeros(1,NumPointsSetb)
# TwoPointsIntersection.m:17
    for i in arange(1,NumPointsSetb).reshape(-1):
        bbx=Setb(arange(),i) - minV + 1
# TwoPointsIntersection.m:19
        s0=max(bbx(1) - R,1)
# TwoPointsIntersection.m:20
        s1=max(bbx(2) - R,1)
# TwoPointsIntersection.m:21
        s2=max(bbx(3) - R,1)
# TwoPointsIntersection.m:22
        ss=LabelMatrix(arange(s0,s0 + dot(2,R)),arange(s1,s1 + dot(2,R)),arange(s2,s2 + dot(2,R)))
# TwoPointsIntersection.m:23
        if max(ravel(ss)) > 0:
            Labelvec[i]=1
# TwoPointsIntersection.m:27
    
    Idexx=find(Labelvec == 1)
# TwoPointsIntersection.m:31
    if isempty(Idexx) == 0:
        Flag00=1
# TwoPointsIntersection.m:33
        InterSets=Setb(arange(),Idexx)
# TwoPointsIntersection.m:34
    else:
        Flag00=0
# TwoPointsIntersection.m:36
        InterSets=[]
# TwoPointsIntersection.m:37
    
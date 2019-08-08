# Generated with SMOP  0.41
from libsmop import *
# ExtracConnectedCurvesPopulation.m
from ExtracConnectedCurvesPopulationSub import ExtracConnectedCurvesPopulationSub

    
# @function
def ExtracConnectedCurvesPopulation(ConnecCell=None,CenterPoints=None,*args,**kwargs):
    # varargin = ExtracConnectedCurvesPopulation.varargin
    # nargin = ExtracConnectedCurvesPopulation.nargin

    CurvesetConnect=cell(1,1000)
# ExtracConnectedCurvesPopulation.m:3
    Numss=size(CenterPoints[1],2)
# ExtracConnectedCurvesPopulation.m:4
    kk=0
# ExtracConnectedCurvesPopulation.m:6
    for i in arange(1,Numss).reshape(-1):
        curvesetSub, LevelCurves, _ = ExtracConnectedCurvesPopulationSub(ConnecCell,CenterPoints,1,i,nargout=2)
# ExtracConnectedCurvesPopulation.m:8
        if isempty(curvesetSub) == 0:
            Numxx=size(curvesetSub,2)
# ExtracConnectedCurvesPopulation.m:10
            for ii in arange(1,Numxx).reshape(-1):
                kk=kk + 1
# ExtracConnectedCurvesPopulation.m:12
                CurvesetConnect[kk]=curvesetSub[ii]
# ExtracConnectedCurvesPopulation.m:13
    
    if kk > 0:
        CurvesetConnect=CurvesetConnect[arange(1,kk)]
# ExtracConnectedCurvesPopulation.m:20
    else:
        CurvesetConnect=[]
# ExtracConnectedCurvesPopulation.m:22
    return CurvesetConnect, LevelCurves
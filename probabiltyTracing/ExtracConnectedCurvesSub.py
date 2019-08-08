# Generated with SMOP  0.41
from libsmop import *
# ExtracConnectedCurvesSub.m

    
@function
def ExtracConnectedCurvesSub(MatrixLabelC=None,ConnecCell=None,CenterPointss=None,*args,**kwargs):
    varargin = ExtracConnectedCurvesSub.varargin
    nargin = ExtracConnectedCurvesSub.nargin

    kk=size(MatrixLabelC,2)
# ExtracConnectedCurvesSub.m:3
    CurvesSet=cell(1,100)
# ExtracConnectedCurvesSub.m:4
    DownLevelMatrixLable=cell(1,100)
# ExtracConnectedCurvesSub.m:5
    kkk=0
# ExtracConnectedCurvesSub.m:6
    if kk > 0:
        for ii in arange(1,kk).reshape(-1):
            MatrixLabel=MatrixLabelC[ii]
# ExtracConnectedCurvesSub.m:9
            NumCurves=size(MatrixLabel,2)
# ExtracConnectedCurvesSub.m:10
            for ij in arange(1,NumCurves).reshape(-1):
                aas=MatrixLabel(arange(),ij)
# ExtracConnectedCurvesSub.m:12
                Curves,MatrixLabel0,CenterPointss=Longestpath(ConnecCell,CenterPointss,aas(1),aas(2),nargout=3)
# ExtracConnectedCurvesSub.m:13
                if isempty(Curves) == 0:
                    kkk=kkk + 1
# ExtracConnectedCurvesSub.m:15
                    CurvesSet[kkk]=Curves
# ExtracConnectedCurvesSub.m:16
                    DownLevelMatrixLable[kkk]=MatrixLabel0
# ExtracConnectedCurvesSub.m:17
    
    if kkk > 0:
        CurvesSet=CurvesSet(arange(1,kkk))
# ExtracConnectedCurvesSub.m:24
        DownLevelMatrixLable=DownLevelMatrixLable(arange(1,kkk))
# ExtracConnectedCurvesSub.m:25
    else:
        CurvesSet=[]
# ExtracConnectedCurvesSub.m:27
        DownLevelMatrixLable=[]
# ExtracConnectedCurvesSub.m:28
    
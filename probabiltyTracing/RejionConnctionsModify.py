# Generated with SMOP  0.41
from libsmop import *
# RejionConnctionsModify.m

    
@function
def RejionConnctionsModify(CenterPoints=None,ClusterPointsCell=None,*args,**kwargs):
    varargin = RejionConnctionsModify.varargin
    nargin = RejionConnctionsModify.nargin

    Numlevel=length(CenterPoints)
# RejionConnctionsModify.m:3
    # MaxCluster=1;
# for i=1:Numlevel
#     Currdata=CenterPoints{i};
#     if size(Currdata,2)>MaxCluster
#        MaxCluster=size(Currdata,2);
#     end
# end
    ConnecMatrix=cell(1,Numlevel - 1)
# RejionConnctionsModify.m:11
    # dd1=ClusterPointsCell{1};
# NumS=length(dd1);
# CurrVec=[];
# for i=1:NumS
#     CurrVec=[CurrVec,dd1{i}];
# end
# CurrVec0{1}=CurrVec;
# ClusterPointsCell{1}=CurrVec0;
    
    for i in arange(1,Numlevel - 1).reshape(-1):
        ss=ClusterPointsCell[i]
# RejionConnctionsModify.m:25
        ss1=ClusterPointsCell[i + 1]
# RejionConnctionsModify.m:26
        Connevec=RejionConnctionsSub(ss,ss1)
# RejionConnctionsModify.m:27
        ConnecMatrix[i]=Connevec
# RejionConnctionsModify.m:28
    
    
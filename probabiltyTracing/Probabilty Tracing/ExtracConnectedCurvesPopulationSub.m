function [CurvesSets0, CurvesSet,CenterPointss]=ExtracConnectedCurvesPopulationSub(ConnecCell,CenterPoints,LevelId,PointId)

CurvesSet=cell(10,160);
CenterPoints0=CenterPoints;
Nums=size(CenterPoints0,2);

for i=1:Nums
    CurrMatrix=CenterPoints{i};
    LLs=zeros(1,size(CurrMatrix,2));
    CenterPoints0{i}=[CurrMatrix(1:3,:);LLs];
end


[Curves,MatrixLabel,CenterPointss]=Longestpath(ConnecCell,CenterPoints0,LevelId,PointId);
CurvesSet{1,1}=Curves;

LableSet{1}=MatrixLabel;

%MatrixLabel
%MatrixLabel(1,:) the node level of tree structure 
%MatrixLabel(2,:) center point in each node
%MatrixLabel(3,:) the number of points in the cluster 

% %------------------µÚ¶ş¼¶
MaxNumCurves=zeros(1,10);

for i=2:10
    [DownLevelMatrixLable,Curves00,CenterPointss]=ExtracConnectedCurvesSub(LableSet,ConnecCell,CenterPointss);
    LableSet=DownLevelMatrixLable;
    if isempty(Curves00)==0
       Numss=size(Curves00,2);
       MaxNumCurves(i)=Numss;
       for jj=1:Numss
           CurvesSet{i,jj}=Curves00{jj};
       end
    else
        break
    end
end
MaxNum=max(max(MaxNumCurves),1);
CurvesSets0=cell(1,100);
kk=0;

for i=1:10
    for j=1:MaxNum
        if isempty(CurvesSet{i,j})==0
            kk=kk+1;
            CurvesSets0{kk}=CurvesSet{i,j};
        end
    end
end

if kk>0
    CurvesSets0=CurvesSets0(1:kk);
   % SpliteCurveSet=SpliteCurves(CurvesSet);
else
    CurvesSets0=[];
    %SpliteCurveSet=[];
end

%-----------------------------------

function [DownLevelMatrixLable,CurvesSet,CenterPointss]=ExtracConnectedCurvesSub(MatrixLabelC,ConnecCell,CenterPointss)

kk=size(MatrixLabelC,2);
CurvesSet=cell(1,100);
DownLevelMatrixLable=cell(1,100);
kkk=0;
if kk>0
    for ii=1:kk
        MatrixLabel=MatrixLabelC{ii};
        NumCurves=size(MatrixLabel,2);
        for ij=1:NumCurves
            aas=MatrixLabel(:,ij);
            [Curves,MatrixLabel0,CenterPointss]=Longestpath(ConnecCell,CenterPointss,aas(1),aas(2));
            if isempty(Curves)==0
                kkk=kkk+1;
                CurvesSet{kkk}=Curves;
               DownLevelMatrixLable{kkk}=MatrixLabel0;
            end
        end
    end
end

if kkk>0
    CurvesSet=CurvesSet(1:kkk);
    DownLevelMatrixLable=DownLevelMatrixLable(1:kkk);
else
    CurvesSet=[];
    DownLevelMatrixLable=[];
end
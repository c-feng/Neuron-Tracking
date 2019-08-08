function  CurvesetConnect=ExtracConnectedCurves(ConnecCell,CenterPoints)

CurvesetConnect=cell(1,1000);




kk=0;
for i=1:Numss
    curvesetSub=ExtracConnectedCurvesPopulationSub(ConnecCell,CenterPoints,1,i);
    if isempty( curvesetSub)==0
        Numxx=size(curvesetSub,2);
        for ii=1:Numxx
            kk=kk+1;
            CurvesetConnect{kk}=curvesetSub{ii};
        end
    end
end


if kk>0
    CurvesetConnect=CurvesetConnect(1:kk);
else
    CurvesetConnect=[];
end

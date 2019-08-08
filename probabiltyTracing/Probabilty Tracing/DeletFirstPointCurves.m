function CurveSet=DeletFirstPointCurves(CurveSet0)
Nums=size(CurveSet0,2);
CurveSet=cell(1,Nums);
for i=1:Nums
    Curve0=CurveSet0{i};
    Curve0=Curve0(:,2:size(Curve0,2));
    CurveSet{i}=Curve0;
end









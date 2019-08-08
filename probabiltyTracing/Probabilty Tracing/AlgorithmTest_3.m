%AlgorithmTest_3.m
[densityVec,MapOrig]=AddPointProMap(CurveSet,BinaryXX3);
CurvesetInf=DirecsAddedInCurve(CurveSet,BinaryXX3,XX3);
AdPoints=zeros(5,51);
kk=0;
for ii=1:51
    curve0=CurvesetInf{ii};
    Addcurve=AddTracingPoint(curve0,MapOrig,0);
    if Addcurve(1)~=0
        kk=kk+1;
        AdPoints(1:4,kk)=Addcurve;
        AdPoints(5,kk)=ii;
    end
end




function Addpoint=AddTracingPointstotal(CurvesetInf,MapOrig,AddPointNum)

%Addpoint=zeros(5,AddPointNum);
Nums=size(CurvesetInf,2);
kk=0;
CurrVec=zeros(5,2*Nums);

for ii=1:Nums
    curve0=CurvesetInf{ii};
    Addcurve=AddTracingPoint(curve0,MapOrig,0);
    if Addcurve(1)~=0
        kk=kk+1;
        CurrVec(1:4,kk)=Addcurve;
        CurrVec(5,kk)=ii;
    end
   
     Addcurve=AddTracingPoint(curve0,MapOrig,1);
    if Addcurve(1)~=0
        kk=kk+1;
        CurrVec(1:4,kk)=Addcurve;
        CurrVec(5,kk)=ii;
    end
end

CurrVec=CurrVec(:,1:kk);

if kk<AddPointNum
    Addpoint=CurrVec;
     [~,Index0]=DeletSimilarPoints(Addpoint(1:3,:),6);
    Addpoint=Addpoint(:,Index0);
else
    [~,Index_x]=sort(CurrVec(4,:),'descend');
    Addpoint=CurrVec(:,Index_x(1:AddPointNum));
    [~,Index0]=DeletSimilarPoints(Addpoint(1:3,:),6);
    Addpoint=Addpoint(:,Index0);
end













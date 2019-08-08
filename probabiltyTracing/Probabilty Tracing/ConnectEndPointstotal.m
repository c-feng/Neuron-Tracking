function Connectvec=ConnectEndPointstotal(CurvesetInf)

Labelvec=CurvesLabel(CurvesetInf);
Nums=size(Labelvec,2);
Connectvec=zeros(3,Nums);
Connectvec(1,:)=Labelvec(1,:);

for i=1:Nums
    currvec=Labelvec(:,i);
    Idvec=SearchConnectCurveSet(CurvesetInf,currvec(1),currvec(3),10);
    if isempty(Idvec)==0
        [maxv,maxIndex]=max(Idvec(5,:));
        Connectvec(2,i)=Idvec(1,maxIndex);
        Connectvec(3,i)=maxv;        
    end
end
 





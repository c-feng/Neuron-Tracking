function prob=ProbConnectCurves(CurvesetInf,Idcurve,flag,thre)
Idvec=SearchConnectCurveSet(CurvesetInf,Idcurve,flag,thre);
Nums=size(Idvec,2);
currvec=CurvesetInf{Idcurve};

if flag==0
    pointv=currvec(:,1);
else
    pointv=currvec(:,end);
end

for i=1:Nums
    
    
    
end





function   [Pathdata,Idflag]=PeakdensityClusterpointsub(Index0,Connets,dataLabel)

Nums=length(dataLabel);
Pathdata=zeros(1,Nums);
Idflag=0;
kk=1;
Pathdata(kk)=Index0;

while Idflag==0&&kk<Nums+1
    Id=Connets(Index0);
    if dataLabel(Id)==0
        kk=kk+1;
        Pathdata(kk)=Id;
        Index0=Id;
    else
        Idflag=dataLabel(Id);
    end
end
 Pathdata= Pathdata(1:kk);






function [Datap,CurrMatrix0,CurrVec,Coordbase]=PeakdensityCenterPointModify(PointsSet)
Numx=length(PointsSet);
kk=0;
CurrVec=zeros(5,200000);
jj=0;
for i=1:Numx
    if isempty(PointsSet{i})==0
        kk=kk+1;
        Currdata=PointsSet{i};
        CurrVec(1:4,jj+1:jj+size(Currdata,2))=Currdata(1:4,:);
        CurrVec(5,jj+1:jj+size(Currdata,2))=kk;
        jj=jj+size(Currdata,2);
    else
        break
    end
end
CurrVec=CurrVec(:,1:jj);
Datap=zeros(7,jj);


Mincx=min(CurrVec(1,:))-9;
Maxcx=max(CurrVec(1,:))+9;

Mincy=min(CurrVec(2,:))-9;
Maxcy=max(CurrVec(2,:))+9;

Mincz=min(CurrVec(3,:))-9;
Maxcz=max(CurrVec(3,:))+9;

CurrMatrix0=zeros(Maxcx-Mincx,Maxcy-Mincy,Maxcz-Mincz);
CurrMatrix1=zeros(Maxcx-Mincx,Maxcy-Mincy,Maxcz-Mincz);

size(CurrMatrix0)
CurrVec(1,:)=CurrVec(1,:)-Mincx;
CurrVec(2,:)=CurrVec(2,:)-Mincy;
CurrVec(3,:)=CurrVec(3,:)-Mincz;
Coordbase=[Mincx;Mincy;Mincz];
CurrMatrix2=zeros(size(CurrMatrix0));
for i=1:size(CurrVec,2)
    a=CurrVec(:,i);
    CurrMatrix0(a(1),a(2),a(3))=max(a(4),1);
    CurrMatrix1(a(1),a(2),a(3))=a(5);
    CurrMatrix2(a(1),a(2),a(3))=i;
end

Tem=zeros(7,7,7);
for i=1:7
    for j=1:7
        for kk=1:7
            curr1=(i-4)^2+(j-4)^2+(kk-4)^2;
            Tem(i,j,kk)=exp(-curr1/9);
        end
    end
end

Tem=Tem./sum(Tem(:));
CurrMatrix0=convn(CurrMatrix0,Tem);
CurrMatrix0=CurrMatrix0(4:size(CurrMatrix0,1)-3,...,
4:size(CurrMatrix0,2)-3,4:size(CurrMatrix0,3)-3);

for ii=1:size(Datap,2)
    Position=CurrVec(1:3,ii);
    datalabel=CurrVec(5,ii);
    [minidist0,connecIndex]=...,
    peakdensitySub1(CurrMatrix0,CurrMatrix1, CurrMatrix2,4,Position,datalabel);
    Datap(1:3,ii)=Position+Coordbase;
    Datap(4,ii)=CurrMatrix0(Position(1),Position(2),Position(3));
    Datap(5,ii)=datalabel;
    Datap(6,ii)=minidist0;
    Datap(7,ii)=connecIndex;
end

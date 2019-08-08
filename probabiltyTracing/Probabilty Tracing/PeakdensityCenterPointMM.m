function [Datap,CurrMatrix0,CurrMatrix1]=PeakdensityCenterPointMM(PointsSet,ImgSize)
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
Nx=ImgSize(1);
Ny=ImgSize(2);
Nz=ImgSize(3);

CurrMatrix0=zeros(Nx,Ny,Nz);
CurrMatrix1=zeros(Nx,Ny,Nz);
CurrMatrix2=zeros(Nx,Ny,Nz);

for i=1:size(CurrVec,2)
    a=CurrVec(:,i);
    CurrMatrix0(a(1),a(2),a(3))=max(a(4),1);
    CurrMatrix1(a(1),a(2),a(3))=a(5);
    CurrMatrix2(a(1),a(2),a(3))=i;
end

for kkmm=1:5
    for ii=1:size(CurrVec,2)
        a=CurrVec(:,ii);
        aver0=NeighborExtract(CurrMatrix0,CurrMatrix1,a,1);
        CurrMatrix0(a(1),a(2),a(3))=aver0;
    end
end

for ii=1:size(Datap,2)
    Position=CurrVec(1:3,ii);
    datalabel=CurrVec(5,ii);
    [minidist0,connecIndex]=...,
    peakdensitySub1(CurrMatrix0,CurrMatrix1, CurrMatrix2,5,Position,datalabel);
    Datap(1:3,ii)=Position;
    Datap(4,ii)=CurrMatrix0(Position(1),Position(2),Position(3));
    Datap(5,ii)=datalabel;
    Datap(6,ii)=minidist0;
    Datap(7,ii)=connecIndex;
end

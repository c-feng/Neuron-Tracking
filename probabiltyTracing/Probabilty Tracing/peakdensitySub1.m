function [minidist0,ConnecIndex]=peakdensitySub1(Image0, ImageLabel,Imagess,NeighborSize,Position,datalabel)
[nx,ny,nz]=size(Image0);
Indexx=max(Position(1)-NeighborSize,1):min(Position(1)+NeighborSize,nx);
Indeyy=max(Position(2)-NeighborSize,1):min(Position(2)+NeighborSize,ny);
Indezz=max(Position(3)-NeighborSize,1):min(Position(3)+NeighborSize,nz);
aa=Image0(Position(1),Position(2),Position(3));
kk=0;
CurrVec=(2*NeighborSize+1)*ones(1,length(Indexx)*length(Indeyy)*length(Indezz));
for i=1:length(Indexx)
    for j=1:length(Indeyy)
        for ij=1:length(Indezz)
            if aa<Image0(Indexx(i),Indeyy(j),Indezz(ij))&&ImageLabel(Indexx(i),Indeyy(j),Indezz(ij))==datalabel
                kk=kk+1;
                CurrVec(1,kk)=norm([Indexx(i);Indeyy(j);Indezz(ij)]-Position);
                CurrVec(2,kk)=Imagess(Indexx(i),Indeyy(j),Indezz(ij));
            end
        end
    end
end
if kk>0
    [minidist0,Indexx]=min(CurrVec(1,:));
    ConnecIndex=CurrVec(2,Indexx);
    minidist0=minidist0/(2*NeighborSize+1);
else
    minidist0=1;
    ConnecIndex=Imagess(Position(1),Position(2),Position(3));
end


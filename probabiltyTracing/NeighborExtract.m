function [aver0,dataset]=NeighborExtract(OrigImg,LabelImg,dataP,NeighborS)

[Nx,Ny,Nz]=size(OrigImg);
dataP=round(dataP);
Indexx=max(dataP(1)-NeighborS,1):min(dataP(1)+NeighborS,Nx);
Indeyy=max(dataP(2)-NeighborS,1):min(dataP(2)+NeighborS,Ny);
Indezz=max(dataP(3)-NeighborS,1):min(dataP(3)+NeighborS,Nz);

Label=LabelImg(dataP(1),dataP(2),dataP(3));
dataset=zeros(4,length(Indexx)*length(Indeyy)*length(Indezz));
kk=0;

for ii=1:length(Indexx)
    for jj=1:length(Indeyy)
        for ij=1:length(Indezz)
            if LabelImg(Indexx(ii),Indeyy(jj),Indezz(ij))==Label
                kk=kk+1;
                dataset(1:3,kk)=[Indexx(ii);Indeyy(jj);Indezz(ij)];
                dataset(4,kk)=OrigImg(Indexx(ii),Indeyy(jj),Indezz(ij));
            end
        end
    end
end

if kk>0
    dataset=dataset(:,1:kk);
    ww=zeros(1,kk);
    for ii=1:kk
        ww(ii)=exp(-norm(dataset(1:3,ii)-dataP(1:3)));
    end
    
    aver0=(ww*dataset(4,:)')/sum(ww);
    aver0=0.2*aver0+0.8*sum(dataset(4,:))/(length(Indexx)*length(Indeyy)*length(Indezz));
else
     dataset=[];
end



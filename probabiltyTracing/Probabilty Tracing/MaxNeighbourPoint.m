function [maxv,maxv_position]=MaxNeighbourPoint(point,Img)
[Nx,Ny,Nz]=size(Img);
pp=round(point);
Indexx=max(pp(1)-1,1):min(pp(1)+1,Nx);
Indeyy=max(pp(2)-1,1):min(pp(2)+1,Ny);
Indezz=max(pp(3)-1,1):min(pp(3)+1,Nz);
kk=0;
V=zeros(4,27);
for ii=1:length(Indexx)
    for jj=1:length(Indeyy)
        for ij=1:length(Indezz)
            dist=norm([Indexx(ii);Indeyy(jj);Indezz(ij)]-pp);
            if dist<1.01
                kk=kk+1;
                V(1:3,kk)=[Indexx(ii);Indeyy(jj);Indezz(ij)];
                V(4,kk)=Img(Indexx(ii),Indeyy(jj),Indezz(ij));
            end
        end
    end
end
[maxv,Indexx]=max(V(4,:));

maxv_position=V(1:3,Indexx);


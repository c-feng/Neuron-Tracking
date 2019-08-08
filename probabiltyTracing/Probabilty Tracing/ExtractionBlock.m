function [ClusterInfor,SegSubblock0,Subblock]=ExtractionBlock(Position,Image0,NeighborSize,Maxthre)
[nx,ny,nz]=size(Image0);
Indexx=max(Position(1)-2*NeighborSize,1):min(Position(1)+2*NeighborSize,nx);
Indeyy=max(Position(2)-2*NeighborSize,1):min(Position(2)+2*NeighborSize,ny);
Indezz=max(Position(3)-NeighborSize-2,1):min(Position(3)+NeighborSize+2,nz);
Subblock=Image0(Indexx,Indeyy,Indezz);
Subblock=min(Subblock,Maxthre);



SegSubblock=SBGCSTotal3D(Subblock,1,0.6,5);



InitialP=[Indexx(1);Indeyy(1);Indezz(1)]-1;
[Nx,Ny,Nz]=size(SegSubblock);


SegSubblock1=zeros(Nx,Ny,Nz);
kk=0;
for i=1:Nx
    for j=1:Ny
        for ij=1:Nz
            if SegSubblock(i,j,ij)==1
                indax=max(i-1,1):min(i+1,Nx);
                inday=max(j-1,1):min(j+1,Ny);
                indaz=max(ij-1,1):min(ij+1,Nz);
                currss=SegSubblock(indax,inday,indaz);
                ss0=sum(currss(:))/(length(indax)*length(inday)*length(indaz));
                if ss0>0.8
                    kk=kk+1;
                   % Segdataset(:,kk)=[i;j;ij];
                    SegSubblock1(i,j,ij)=1;
                end
            end
        end
    end
end
Position1=Position-InitialP;
Segdataset=Regiongrow(Position1,SegSubblock1,60,1);
%SegSubblock=SegSubblock1;
Tempp=zeros(3,3,3);
for ii=1:3
    for jj=1:3
        for ij=1:3
            Tempp(ii,jj,ij)=exp(-(ii-2)^2-(jj-2)^2-(ij-2)^2);
        end
    end
end


%  ConvImag0=convn(SegSubblock1,Tempp);
%  ConvImag0=convn(ConvImag0,Tempp);
%  ConvImag0=convn(ConvImag0,Tempp);
 
%[Sphere_XX,DiffSphere_XX]=SphereCoorDinate(SegSubblock1,r0,Theta,Phi)
SegSubblock0=zeros(Nx,Ny,Nz);
for i=1:size(Segdataset,2)
    curr=Segdataset(1:3,i);
    SegSubblock0(curr(1),curr(2),curr(3))=1;
end

ax=ProjectionOnedimen(SegSubblock0,1);
ax=MeanfilterS(ax,1);
ay=ProjectionOnedimen(SegSubblock0,2);
ay=MeanfilterS(ay,1);
az=ProjectionOnedimen(SegSubblock0,3);
az=MeanfilterS(az,1);
[~,Index]=max(ax);
[~,Indey]=max(ay);
[~,Indez]=max(az);
centerp=[Index;Indey;Indez]
InitialU=SphereCoorDinate(SegSubblock0,centerp,0:1:25,40,80);
figure()
imagesc(InitialU)
somas=zeros(3,50000);
kk=0;
for i=1:Nx
    for j=1:Ny
        for ij=1:Nz
            if SegSubblock0(i,j,ij)==1
                 flag=JudgePointInRecshape(centerp,InitialU,[i;j;ij]);
                 if flag==1
                     kk=kk+1;
                     somas(:,kk)=[i;j;ij]+InitialP;
                 end
            end
        end
    end
end
if kk>0
    ClusterInfor=somas(:,1:kk);
else
    ClusterInfor=[];
end




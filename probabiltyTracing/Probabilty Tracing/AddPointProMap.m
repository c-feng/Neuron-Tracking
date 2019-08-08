function [densityVec,MapOrig]=AddPointProMap(CurveSet,BinaryImg)

MapOrig=BinaryImg;
NumCurves=size(CurveSet,2);
[Nx,Ny,Nz]=size(BinaryImg);

for i=1:NumCurves
    curve0=CurveSet{i};
    for ij=1:size(curve0,2)
        Point=round(curve0(1:3,ij));
        Indexx=max(Point(1)-2,1):min(Point(1)+2,Nx);
        Indeyy=max(Point(2)-2,1):min(Point(2)+2,Ny);
        Indezz=max(Point(3)-1,1):min(Point(3)+1,Nz);
        MapOrig(Indexx,Indeyy,Indezz)=0;
    end
end

Dataset=Extract3Dpoints(MapOrig);
densityVec=kernelDensity(Dataset,ones(1,size(Dataset,2)),2,4);
MapOrig=zeros(Nx,Ny,Nz);
for i=1:size(densityVec,2)
    currv=densityVec(:,i);
    MapOrig(currv(1),currv(2),currv(3))=currv(4);
end






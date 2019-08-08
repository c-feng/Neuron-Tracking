function [Npoints,IndexVec]=NeigbourPointsModify(singlePoint,HashSet,PointSet,Thre)

[Nxx,Nyy]=size(HashSet);
px=round(mod(singlePoint(1),Nxx));
if px==0
    px=Nxx;
end
py=round(mod(singlePoint(2),Nyy));
if py==0
    py=Nyy;
end
Thre0=round(Thre+0.5);
Indexx=max(px-Thre0,1):min(px+Thre0,Nxx);
Indeyy=max(py-Thre0,1):min(py+Thre0,Nyy);
IndexLable=[];
for ii=1:length(Indexx)
    for jj=1:length(Indeyy)
        IndexLable=[IndexLable,HashSet{Indexx(ii),Indeyy(jj)}];
    end
end
if isempty(IndexLable)==0
    partPoints=PointSet(:, IndexLable);
    distVec=zeros(3,size( partPoints,2));
    IndexVec=zeros(1,size( partPoints,2));
    kk=0;
    for i=1:size(partPoints,2)
        if norm(partPoints(:,i)-singlePoint)<Thre
            kk=kk+1;
            distVec(:,kk)=partPoints(:,i);
            IndexVec(kk)=IndexLable(i);
        end
    end
    if kk>0
        Npoints=distVec(:,1:kk);
        IndexVec=IndexVec(1:kk);
    else
        Npoints=[];
         IndexVec=[];
    end
else
    Npoints=[];
     IndexVec=[];
end


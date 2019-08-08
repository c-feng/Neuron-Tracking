function Npoints=NeigbourPoints(singlePoint,PointSet,Thre)
HashSet=cell(100,100);
Nums=size(PointSet,2);
for i=1:Nums
    px=round(mod(PointSet(1,i),100));
    if px==0
        px=100;
    end
    py=round(mod(PointSet(2,i),100));
    if py==0
        py=100;
    end
    HashSet{px,py}=[HashSet{px,py},i];
end

px=round(mod(singlePoint(1),100));
if px==0
    px=100;
end
py=round(mod(singlePoint(2),100));
if py==0
    py=100;
end
Thre0=round(Thre+0.5);
Indexx=max(px-Thre0,1):min(px+Thre0,100);
Indeyy=max(py-Thre0,1):min(py+Thre0,100);
IndexLable=[];
for ii=1:length(Indexx)
    for jj=1:length(Indeyy)
        IndexLable=[IndexLable,HashSet{Indexx(ii),Indeyy(jj)}];
    end
end
if isempty(IndexLable)==0
    partPoints=PointSet(:, IndexLable);
    distVec=zeros(3,size( partPoints,2));
    kk=0;
    for i=1:size(partPoints,2)
        if norm(partPoints(:,i)-singlePoint)<Thre
            kk=kk+1;
            distVec(:,kk)=partPoints(:,i);
        end
    end
    if kk>0
        Npoints=distVec(:,1:kk);
    else
        Npoints=[];
    end
else
    Npoints=[];
end












function densityVec=kernelDensity(PointS,Weights,Sigma,Thre)
Num_x=size(PointS,2);
densityVec=zeros(4,Num_x);
%--------------HashMatrix
HashSet=cell(100,100);
Nums=size(PointS,2);
for i=1:Nums
    px=round(mod(PointS(1,i),100));
    if px==0
        px=100;
    end
    py=round(mod(PointS(2,i),100));
    if py==0
        py=100;
    end
    HashSet{px,py}=[HashSet{px,py},i];
end
%------------------
for ii=1:Num_x
    singleP=PointS(:,ii);
    [Npoints,Indexv]=NeigbourPointsModify(singleP,HashSet,PointS,Thre);
    dist=0;
    for jj=1:size(Npoints,2)
        dist=dist+Weights(Indexv(jj))*exp(-norm(singleP-Npoints(:,jj))^2/(2*Sigma^2));
    end
    densityVec(:,ii)=[singleP;dist];
end

densityVec(4,:)=densityVec(4,:)./max(densityVec(4,:));



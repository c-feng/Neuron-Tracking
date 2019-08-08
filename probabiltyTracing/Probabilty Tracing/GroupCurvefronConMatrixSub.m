function IdSet=GroupCurvefronConMatrixSub(ConnecMatrix,Idvec,thre)
Nums=length(Idvec);
IdSet=zeros(1,2*Nums);
kk=0;
for i=1:Nums
    currv=ConnecMatrix(Idvec(i),:);
    if currv(3)>thre
        kk=kk+1;
        IdSet(kk)=currv(2);
    end
     if currv(5)>thre
        kk=kk+1;
        IdSet(kk)=currv(4);
    end
end
if kk>0
    IdSet=IdSet(1:kk);
else
    IdSet=[];
end





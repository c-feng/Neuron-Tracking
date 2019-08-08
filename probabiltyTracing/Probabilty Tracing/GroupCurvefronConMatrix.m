function GroupCurves=GroupCurvefronConMatrix(Connectvec,Id,thre)

Nums=size(Connectvec,2);
Labelv=zeros(1,Nums);
ComMatrix=zeros(Connectvec(1,end),5);
GroupCurves=[];
for i=1:Nums
    currv=Connectvec(:,i);
    if ComMatrix(currv(1),1)==0
       ComMatrix(currv(1),1:3)=currv';
    else
        if currv(2)~=0
        ComMatrix(currv(1),4:5)=currv(2:3)';
        else
            ComMatrix(currv(1),4:5)=[-1,-1];
        end
    end
end

Labelv(Id)=1;

for ii=1:50
    IdSet=GroupCurvefronConMatrixSub(ComMatrix,Id,thre);
    if isempty(IdSet)==0
        Currv=zeros(1,size(IdSet,2));
        for jj=1:size(IdSet,2)
            if Labelv(IdSet(jj))==0
                Currv(jj)=1;
                Labelv(IdSet(jj))=1;
            end
        end
        Id0=find(Currv==1);
        if isempty(Id0)==1
            break
        else
            Id=IdSet(Id0);
        end
    else
        break
    end
end

GroupCurves=find(Labelv==1);





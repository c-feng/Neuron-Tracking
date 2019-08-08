function   Connevec=RejionConnctionsSub(ss,ss1);
%  Nums=size(ss1,1)
%  Numx=size(ss,1)
%  Nums=size(ss1,2);
%   Numx=size(ss,2);
Nums=length(ss1);
Numx=length(ss);
Connevec=zeros(2,Nums);
for i=1:Nums
    Vess=zeros(1,Numx);
    data00=ss1{1};
    
    for j=1:Numx
        Flag00=0;
        if size(ss{j},2)>1&size(ss1{i},2)>1
            [Flag00,InterSets]=TwoPointsIntersection(ss{j},ss1{i},2);
        end
        if Flag00==1
            Vess(j)=size(InterSets,2);
        end
    end
    [Maxv,MaxIndex]=max(Vess);
    if Maxv>0
        Connevec(2,i)=Maxv;
        Connevec(1,i)=MaxIndex;
    end
end



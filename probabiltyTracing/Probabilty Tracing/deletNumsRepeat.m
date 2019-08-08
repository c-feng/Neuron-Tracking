function  [Indexx,vect]=deletNumsRepeat(vecs)
Num0=size(vecs,2);
Labelvec=zeros(1,Num0);
for i=1:Num0
    if Labelvec(i)==0
        for j=i+1:Num0
            if vecs(i)==vecs(j)
                Labelvec(j)=1;
            end
        end
    end
end

Indexx=find(Labelvec==0);
vect=vecs(Indexx);


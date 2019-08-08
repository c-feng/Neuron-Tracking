function [repPoints,Index0]=DeletSimilarPoints(Points,Thre)

Num0=size(Points,2);
Currvec=zeros(1,Num0);


for i=1:Num0
    for j=i+1:Num0
        if norm(Points(1:3,i)-Points(1:3,j))<Thre&&Currvec(i)==0
            Currvec(j)=1;
        end
    end
end


Index0=find(Currvec==0);
repPoints=Points(:,Index0);




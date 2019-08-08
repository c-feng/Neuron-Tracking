function [Dist,P]=ProbSkeleton(dataP,T)

Nums=size(dataP,2);
P=zeros(Nums-2,1);
Dist=zeros(Nums-2,1);
for i=1:Nums-2
    Dist(i)=1/2*norm(dataP(1:3,i)+dataP(1:3,i+2)-2*dataP(1:3,i+1));
    P(i)=exp(-Dist(i)/T);
end





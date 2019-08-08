function [Minv1,Index1]=distPointToCurve(starP,curve)

Nums=size(curve,2);
disv=zeros(1,Nums);
for i=1:Nums
    disv(i)=norm(starP-curve(:,i));
end


[Minv1,Index1]=min(disv);




function [avec,b,A]=LSSVR(data,y,sigma,c)
Nums=length(y);
KernelMat=zeros(Nums,Nums);
for i=1:Nums
    for j=1:Nums
        a=exp(-norm(data(:,i)-data(:,j))^2/(2*sigma^2));
        KernelMat(i,j)=a;
    end
end
A=zeros(Nums+1,Nums+1);
A(1,2:Nums+1)=ones(1,Nums);
A(2:Nums+1,1)=ones(Nums,1);
A(2:Nums+1,2:Nums+1)=KernelMat+1/c*eye(Nums);
bb=zeros(Nums+1,1);
bb(2:Nums+1)=y;
x=inv(A)*bb;
b=x(1);
avec=x(2:Nums+1);
%--------------------------
yy=zeros(Nums,1); %  model value

for i=1:Nums
    ss=0;
    for j=1:Nums
        ss=ss+avec(j)*exp(-norm(data(i)-data(j))^2/(2*sigma^2));
    end
    yy(i)=ss+b;
end

figure()
plot(yy)
hold on
plot(y,'r')

%--------------------------



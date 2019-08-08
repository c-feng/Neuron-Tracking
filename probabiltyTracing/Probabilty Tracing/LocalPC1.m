function [P,g,H,sigmaH,mdata3]=LocalPC1(data1,data3,data2,W)
% m 幅度
% data1:数据点
% data2: 协方差矩阵
% data3:新的数据点
nx=size(data1,2);
X=data3;
nyy=size(data1,1);
C=zeros(1,nx);
U=zeros(nyy,nx);
g=zeros(nyy,1);
HH=zeros(nyy,nyy);
MMs=HH;
MMs1=zeros(nyy,1);

for i=1:nx
     dd=X-data1(:,i);
     dd(3)=2*dd(3);
     if norm(dd)<3.5%3.5%4.5%3.5
     C(i)=W(i)*exp(-(.5)*dd'*data2*dd);
     
     else
      C(i)=0;
     end
     U(:,i)=data2*dd;
     g=g+C(i)*U(:,i);
     HH=HH+C(i)*(U(:,i)*U(:,i)'-data2);
     MMs=MMs+C(i)*data2;
      %MMs1=MMs1+C(i)*data2(:,:,i)*data1(:,i);
     MMs1=MMs1+C(i)*data2*dd;
end

% sM=nx;
% P=sum(C)./sM;
% g=-g./sM;
% H=HH./sM;

P=sum(C);
if P>0
g=-g;
H=HH;
sigmaH=-H./P+(P^(-2))*g*g';
else
    sigmaH=0;
    H=HH;
end

if abs(det(MMs))>0.1
mdata3=inv(MMs)*MMs1;
else
    mdata3=MMs1;
end
data=normrnd(0,1,2,1000);
data1=data;
data2=data;
p1=[1;1]./sqrt(2);
p2=[1;-1]./sqrt(2);
p3=[1;-1]./sqrt(2);
p4=[1;1]./sqrt(2);
A=[p1,p2]*[3,0;0,1];
A1=[p3,p4]*[3,0;0,1];
yy=A*data;
yy1=A*data1;
yy2=A1*data2;
yy1(1,:)=yy1(1,:)+16;
yy1(2,:)=yy1(2,:)+7;
yy2(1,:)=yy2(1,:)+7;
yy2(2,:)=yy2(2,:)+6;
plot(yy(1,:),yy(2,:),'ro')
hold on;
plot(yy1(1,:),yy1(2,:),'*')
hold on;
plot(yy2(1,:),yy2(2,:),'.')
E=[yy,yy1,yy2];
R=zeros(1,3000);
R(1:1:1000)=1;
R(1001:1:2000)=1;
R(2001:1:3000)=2;
Q=[E;R];
%------------------------
pp=[Q(:,1),Q(:,200),Q(:,2300)];
A=LearnVec(Q,0.005,pp,15)
%---------------------
%plott(A,3,0.5)
plot(C(1,:),C(2,:),'r.')
data=normrnd(0,1,2,1000);
data1=data;
data1(1,:)=data1(1,:);
%data1(2,:)=data1(1,:)+3;
p1=[1;1]./sqrt(2);
p2=[1;-1]./sqrt(2);
%[p1;p2]*
A=[p1,-p2]*[3,0;0,1];

yy=A*data;
yy1=A*data1;
yy1(1,:)=yy1(1,:)+8;
%yy1(2,:)=yy1(2,:)+9;
plot(yy(1,:),yy(2,:),'ro')
hold on
plot(yy1(1,:),yy1(2,:),'ko')



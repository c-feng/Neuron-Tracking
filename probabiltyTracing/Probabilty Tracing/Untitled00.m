data=normrnd(0,1,2,1000);
data1=data;
data2=data;
p1=[1;1]./sqrt(2);
p2=[1;-1]./sqrt(2);
p3=[1;-1]./sqrt(2);
p4=[1;1]./sqrt(2);
X=[p1,p2]*[3,0;0,1];
A1=[p3,p4]*[3,0;0,1];
yy=X*data;
yy1=X*data1;
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
R(1001:1:2000)=2;
R(2001:1:3000)=3;
Q=[E;R];
X=zeros(3,4);
C=zeros(2,4);
B=ceil(rand(1,4)*3000);
D=zeros(3000,1);
F=zeros(3000,4);
for b=1:1:4
    X(:,b)=Q(:,B(b));
    C(:,b)=E(:,B(b));
end
A=LearnVec(Q,0.001,X,3)
figure()
plot(C(1,:),C(2,:),'r.')

D=zeros(3000,4);
for j=1:1:3000
    D(j,1)=norm(E(:,j)-A(1:2,1));
    D(j,2)=norm(E(:,j)-A(1:2,2));
    D(j,3)=norm(E(:,j)-A(1:2,3));
    D(j,4)=norm(E(:,j)-A(1:2,4));
end
MinVec=zeros(1,3000);
for j=1:3000
    [~,Index]=min(D(j,:));
    MinVec(j)=Index;
end
Index1=find(MinVec==1);
Index2=find(MinVec==2);
Index3=find(MinVec==3);
Index4=find(MinVec==4);
C1=E(:,Index1);
C2=E(:,Index2);
C3=E(:,Index3);
C4=E(:,Index4);
figure()
plot(C1(1,:),C1(2,:),'r.')
hold on
plot(C2(1,:),C2(2,:),'k.')
plot(C3(1,:),C3(2,:),'g.')
plot(C4(1,:),C4(2,:),'m.')
    
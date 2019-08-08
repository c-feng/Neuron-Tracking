%成与图9.5的类似的数据
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
%然后用均值法及学习向量化进行聚类分析
%均值法，已知k=3

%B=[A(1,:);A(2,:)];


C=zeros(3000,1);
D=zeros(3000,3);
E=[yy,yy1,yy2];
a=floor(1+(3000-1)*rand(1,3));
B=E(:,a);
% C1=zeros(2,3000);
% C2=zeros(2,3000);
% C3=zeros(2,3000);
for kkk=1:50
   % k1=0;k2=0;k3=0;
    for j=1:1:3000
        D(j,1)=norm(E(:,j)-B(:,1));
        D(j,2)=norm(E(:,j)-B(:,2));
        D(j,3)=norm(E(:,j)-B(:,3));
    end
    MinVec=zeros(1,3000);
    for j=1:3000
        [~,Index]=min(D(j,:));
        MinVec(j)=Index;
    end
    Index1=find(MinVec==1);%
    Index2=find(MinVec==2);
    Index3=find(MinVec==3);
    u1=mean(E(:,Index1),2);
    u2=mean(E(:,Index2),2);
    u3=mean(E(:,Index3),2);
    B1=[u1,u2,u3];
    if norm(B1-B)<0.01
        break
    else
        B=B1;
    end
end
%---------------- the  clusters
C1=E(:,Index1);
C2=E(:,Index2);
C3=E(:,Index3);
figure()
plot(C1(1,:),C1(2,:),'r.')
hold on
plot(C2(1,:),C2(2,:),'k.')
plot(C3(1,:),C3(2,:),'g.')
%-----------------


%     for j=1:1:3000
%         D(j,1)=norm(E(:,j)-B(:,1));
%         D(j,2)=norm(E(:,j)-B(:,2));
%         D(j,3)=norm(E(:,j)-B(:,3));
%         [~,unindex]=min(D);
%         if unindex==1
%             k1=k1+1;
%             C1(:,k1)= E(:,j);
%         end
%         if unindex==2
%             k2=k2+1;
%             C2(:,k2)=E(:,j);
%         end
%         if unindex==3
%             k3=k3+1;
%             C3(:,k3)=E(:,j);
%         end
%     end
    
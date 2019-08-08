%zuoye2.m

Y=[ 0.460
    0.376
    0.264
    0.318
    0.215
    0.237
    0.149
    0.211
    0.091
    0.267
    0.057
    0.099
    0.161
    0.198
    0.370
    0.042
    0.103];
X=[0.697 
   0.774 
   0.634 
   0.608 
   0.556
   0.403 
   0.481
   0.437
   0.666 
   0.243
   0.245
   0.343 
   0.639
   0.657 
   0.360 
   0.593
   0.719 ];

%I=ones(17,17);
I=eye(17);
A=zeros(17,17);
r=0.1;
c=0.1;
for m=1:1:17
    for n=1:1:17
       A(n,m)=exp(-sum((X(n,:)-X(m,:)).^2)/(2*r^2)); 
    end
end
B=zeros(17,17);
B=A+(1/c)*I;
C=zeros(18,18);
C(1,1)=0;
for d=2:1:18
    C(1,d)=1;
end
for e=2:1:18
    C(e,1)=1;
end
for f=2:1:18
    for g=2:1:18
        C(f,g)=B(f-1,g-1);
    end
end
F=[0;Y];
E=inv(C)*F;
b0=E(1);
avec0=E(2:end)

[avec,b,AA]=LSSVR(X',Y,.1,.1);






% 
% 
% XX=load('xigua.txt');
% x=XX(:,1:2);
% y=XX(:,3);
% y(9:17)=-1;
% z=zeros(17,17);
% c=10;
% sigma=1;
% I=ones(17,1);
% l=length(y);
% for i=1:length(y)
%     for j=1:length(y)
%         z(i,j)=y(i)*y(j)*exp(-norm(x(i,:)-x(j,:))^2/(2*sigma^2));
%     end
% end
% 
% d=z+(1/c)*eye(l);%矩阵第四项
% m=[0,-y';y,d];%.为横向拼接；为纵向
% ans=inv(m)*[0;I];
% b=ans(1);
% a=ans(2:l+1);
% %-----解出矩阵---------------------
% %------将离散点画出来--------------------
% plot(x(1:6,1),x(1:6,2),'o');
% hold on
% plot(x(7:17,1),x(7:17,2),'.');
% %------将超平面画出来--------------------
% x1=0.01:0.005:1;
% x2=0.01:0.0025:0.5+0.0025;
% yj=zeros(1,length(x1))
% for i=1:length(x2)    
%     for j=1:length(x1)
%         xt=[x1(j) x2(i)];
%         s=0;
%         for k=1:l
%             s=s+a(k)*y(k)*exp(-(norm(x(k,:)-xt))^2/(2*sigma^2));
%         end
%         yj(j)=s+b;
%     end
%     [~,yindex]=min(abs(yj));
%     ymin(i,1:2)=[x1(yindex(1)) x2(i)];
% end
% plot(ymin(:,1),ymin(:,2),'-');

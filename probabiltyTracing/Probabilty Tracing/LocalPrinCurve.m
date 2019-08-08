function [sigmaH,Shiftvec]=LocalPrinCurve(Points,SinglePoint,MatrixCov,W)

Nums=size(Points,2);
X=SinglePoint;
nyy=size(Points,1);
C=zeros(1,Nums);
U=zeros(nyy,Nums);
g=zeros(nyy,1);
HH=zeros(nyy,nyy);
MMs=HH;
MMs1=zeros(nyy,1);
CovM=MatrixCov;
for i=1:Nums
     dd=X-Points(:,i);
     C(i)=W(i)*exp(-(.5)*dd'*CovM*dd);
     U(:,i)=CovM*dd;
     g=g+C(i)*U(:,i);
     HH=HH+C(i)*(U(:,i)*U(:,i)'-CovM);
     MMs=MMs+C(i)*CovM;
     MMs1=MMs1+C(i)*CovM*dd;
end
% figure()
% plot(sqrt(sum(U.*U)))
% hold on
% plot(C,'r')


P=sum(C);
if P>0
g=-g;
H=HH;
sigmaH=-H./P+(P^(-2))*g*g';
else
    sigmaH=eye(ny);
end

if abs(det(MMs))>0.1
Shiftvec=inv(MMs)*MMs1;
else
    Shiftvec=MMs1;
end
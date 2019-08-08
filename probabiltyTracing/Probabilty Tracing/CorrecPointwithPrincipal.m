function [corretdata,pdirc,Ratio]=CorrecPointwithPrincipal(datap,BinaryXX3,OrigImg)
datatt=SphereCoorDinateExtr(BinaryXX3,datap,0:0.3:7,15,30);
ss=datap;
 W=weigthvalue(datatt,OrigImg);
 W=W./max(max(W),0.01);
for i=1:100
    [sigmaH,Shiftvec]=LocalPrinCurve(datatt,ss,0.3*eye(3),W);
    ss1=ss+0.15*sqrt(1/i)*normrnd(0,3,3,1);
   
    [~,Shiftvec1]=LocalPrinCurve(datatt,ss1,0.3*eye(3),W);
    if norm(Shiftvec1)<norm(Shiftvec)
        ss=ss1;
    end
end

corretdata=ss;

H=inv(sigmaH+0.01*eye(3));
[EigVec,Eigvalue]=eig(H);

[maxv,Index]=max(diag(abs(Eigvalue)));
pdirc=EigVec(:,Index);
Ratio=maxv/sum(diag(abs(Eigvalue)));


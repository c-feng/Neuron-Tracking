function x1=gproPCAA(x0,sigmaH,T1,T2,at1)
%º∆À„g,A,H,u

a=0.5;
g=sigmaH*x0;
A=[x0,-T1,-T2];
H=zeros(3,3);
H(1,1)=x0'*x0-1;
H(2,2)=-T1'*x0+at1;
H(3,3)=-T2'*x0+at1;
ss=inv(A'*A-H);
B=ss*A';
U=B*g;
P=eye(3)-A*ss*A';
if sum(abs(P*g))<0.001&min(U)>-0.0001
    x1=x0;
else
    v=zeros(2,1);
    for i=1:3
    if U(i)<-0.0001
        v(i)=U(i);
    else
        v(i)=-H(i,i);
    end
    end
    S=P*g+B'*v;
    
    p1=(1-a)*g'*S/(abs(U'*ones(3,1))+1);
   
    d=S-p1*B'*ones(3,1);
    d=d./max(abs(d));
    for i=1:20
        x1=x0+0.5^i*d;
        x1=x1./norm(x1);
        Ls=x1'*sigmaH*x1-x0'*sigmaH*x0;
        dkkss1=-T1'*x1./norm(x1)+at1+0.001;
        dkkss3=-T2'*x1./norm(x1)+at1+0.001;
        dkkss2=0.5^(i+1)*g'*d;
        if Ls>dkkss2&norm(x1)<1.0001&dkkss1<0&norm(x1)>0.999&dkkss3<0
%              x1=x0+0.5^i*d;
            break
        end
    end
    if i==20
        x1=x0;
    end
end
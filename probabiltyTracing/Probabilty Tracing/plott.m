function plott(A,q,h)
A=zeros(3,q);
C=zeros(2,q);
B=ceil(rand(1,q)*3000);
D=zeros(3000,1);
F=zeros(3000,q);
for b=1:1:q
    A(:,b)=Q(:,B(b));
    C(:,b)=E(:,B(b));
end
for j=1:1:3000;
    for i=1:1:q;
    F(j,i)=norm(E(:,j)-C(:,i));
    D(j,1)=min(d(j,:));
    if Q(3,j)==A(3,i)
        C(:,i)=C(:,i)+h*(E(:,j)-C(:,i));
    else
         C(:,i)=C(:,i)-h*(E(:,j)-C(:,i));
    end
    end
end
function [x1,x2,x3,sigmaH]=gproPCAtotalFF(sigmaH,T1,T2,at1)
%data:n行3列样本数据。
%T1 约束数据的第一组成分向量。

x0=T1;
for i=1:15
    x1=gproPCAA(x0,sigmaH,T1,T2,at1);
    x0=x1;
    mms=x0'*T1;
    if mms<0.7
        
        break
    end
end


%第二主成分
x2=x1;
[idxv,idexx]=sort(abs(x1));
cdd=(idxv(1)^2+idxv(2)^2)/idxv(3);
x2(idexx(3))=-sign(x1(idexx(3)))*cdd;

if cdd~=0
x2=x2./norm(x2);
else
    x2(idexx(2))=1;
end

%第三主成分
x3=zeros(3,1);
x3(idexx(1))=-1;
AA=[x1';x2'];
x3(idexx(2))=det(AA(:,[idexx(1),idexx(3)]))/det(AA(:,[idexx(2),idexx(3)]));
x3(idexx(3))=det(AA(:,[idexx(2),idexx(1)]))/det(AA(:,[idexx(2),idexx(3)]));
x3=x3./norm(x3);
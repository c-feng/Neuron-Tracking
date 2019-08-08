function A=LearnVec(Q,h,A,Num0)
q=size(A,2);
dim0=size(A,1)-1;
NumSample=size(Q,2);
for jjk=1:Num0
    for j=1:1:NumSample
        dd=zeros(1,q);
        for i=1:1:q
            dd(i)=norm(Q(1:dim0,j)-A(1:dim0,i));
        end
        [~,Index]=min(dd);
        pp=A(:,Index);
        if pp(dim0+1)==Q(dim0+1,j)
            pp(1:dim0)=pp(1:dim0)+h*(Q(1:dim0,j)-pp(1:dim0));
        else
            pp(1:dim0)=pp(1:dim0)-h*(Q(1:dim0,j)-pp(1:dim0));
        end
        A(:,Index)=pp;
    end
end
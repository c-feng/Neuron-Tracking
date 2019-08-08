function A=plottt(Q,h,A,Num0)


q=size(A,2);
for jjk=1:Num0
    for j=1:1:3000
        dd=zeros(1,3);
        for i=1:1:q
            dd(i)=norm(Q(1:2,j)-A(1:2,i));
        end
        [~,Index]=min(dd);
        pp=A(:,Index);
        if pp(3)==Q(3,j)
            pp(1:2)=pp(1:2)+h*(Q(1:2,j)-pp(1:2));
        else
            pp(1:2)=pp(1:2)-h*(Q(1:2,j)-pp(1:2));
        end
        A(:,Index)=pp;
    end
end
function Prob=ProbConnectTotal(Line1,LineSet,T0)

num0=size(LineSet,2);
Prob=zeros(num0+1,1);
Currv=zeros(num0,1);
for i=1:num0
    Line2=LineSet{i};
    [~,Pver]=ProbConnect(Line1,Line2);
    Currv(i)=exp(-Pver/T0);
end

Z=sum(Currv)+1;
Prob(num0+1)=1/Z;
Prob(1:num0)=Currv./Z;












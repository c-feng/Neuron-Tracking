%AlgorithmTest.m

Curve1=zeros(3,30);
n1=ones(3,1)./sqrt(3);
for i=1:29
    Curve1(:,i+1)=i*n1;
end

Curve2=zeros(3,29);
n2=[-1;-1.5;-1]./sqrt(2+1.5^2);

for i=1:29
    Curve2(:,i)=i*n2+0.2*(rand(3,1)-0.5);
end

n3=[0;0;1];
Curve3=zeros(3,29);

for i=1:29
    Curve3(:,i)=i*n3+0.2*(rand(3,1)-0.5);
end

Point=zeros(3,1);
SegSet{1}=Curve1;
SegSet{2}=Curve2;
SegSet{3}=Curve3;
CurvesLabel=zeros(1,3);
CurvesLabel(1)=1;
Thre=2;
[CurveIds,CurvesLabel,PPoints]=...
    SearchConnectSegmentsSub(Point,SegSet,CurvesLabel,Thre);

Line1=Curve1(:,1:2);
Line2=Curve2(:,1:2);
Line3=Curve3(:,1:2);
LineSet=cell(1,2);
LineSet{1}=Line2;
LineSet{2}=Line3;
Prob=ProbConnectTotal(Line1,LineSet,1);

ss1=InnerEnergeComput(Line1,Line2);
ss2=InnerEnergeComput(Line1,Line3);
ss3=InnerEnergeComput(Line1,[]);
T=.1;
R=exp(-(ss1-ss2)/T)*Prob(2)/Prob(1)
R=exp(-(ss3-ss2)/T)*Prob(3)/Prob(1)





% plot3(Curve3(1,:),Curve3(2,:),Curve3(3,:),'ro-')
% hold on
% plot3(Curve1(1,:),Curve1(2,:),Curve1(1,:),'o-')






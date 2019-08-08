function [P,Pver]=ProbConnect(Line1,Line2)
AverageP=...
0.25*(Line1(1:3,1)+Line1(1:3,2)+Line2(1:3,1)+Line2(1:3,2));
P=norm(Line1(1:3,1)-AverageP)+norm(Line2(1:3,1)-AverageP);

n0=(Line1(1:3,1)+Line1(1:3,2))-Line2(1:3,1)-Line2(1:3,2);
nn0=n0./max(norm(n0),0.01);

kk1=Line1(1:3,1)-AverageP;
kk2=Line2(1:3,1)-AverageP;
ss1=sqrt(kk1'*kk1-(kk1'*nn0)^2);
ss2=sqrt(kk2'*kk2-(kk2'*nn0)^2);
Pver=ss1+ss2;












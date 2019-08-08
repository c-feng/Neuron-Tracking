function flag=JudgePointInRecshape(Centerp,ShapeInf,Point)
[Nx,Ny]=size(ShapeInf);
Currp=Point-Centerp;
Point_Sphere=PointToSphereCoordinate(Currp);
deltaAzi = (2*3.1415926)/Nx;
deltaAlt = (3.1415926)/Ny;
Alt = max(floor(Point_Sphere(2) / deltaAlt),1);
Azi = max(floor(Point_Sphere(3) / deltaAzi),1);
r0=ShapeInf(Azi,Alt);
rr=norm(Point-Centerp);
if rr<r0+0.5
    flag=1;
else
    flag=0;
end





% P1=SpherepointTransform(r0,Alt*deltaAlt,Azi*deltaAzi);
% 
% 
% 
% if Alt-1<1
%     Alt0=Ny;
% else
%     Alt0=Alt-1;
% end
% if Alt+1>Ny
%     Alt1=1;
% else
%     Alt1=Alt+1;
% end
% 
% if Azi-1<1
%     Azi0=Nx;
% else
%     Azi0=Azi-1;
% end
% 
% if Azi+1>Nx
%     Azi1=1;
% else
%     Azi1=Azi+1;
% end
% CurrMat=zeros(2,5);
% CurrMat(:,1)=[Alt0;Azi];CurrMat(:,2)=[Alt1;Azi];
% CurrMat(:,3)=[Alt;Azi];CurrMat(:,4)=[Alt;Azi0];
% CurrMat(:,5)=[Alt;Azi1];
% pairvec=zeros(2,4);
% pairvec(:,1)=[1;2];pairvec(:,2)=[2;4];
% pairvec(:,3)=[4;5];pairvec(:,4)=[5;1];
% 
% r0=ShapeInf(CurrMat(2,3),CurrMat(1,3));
% P0=SpherepointTransform(r0,CurrMat(1,3)*deltaAlt,CurrMat(2,3)*deltaAzi);
% %P0=Centerp+P0;
% Flag=zeros(1,4);
% Point=Point-Centerp;
% for i=4
%     currv=pairvec(:,i);
%     aa1=CurrMat(:,currv(1));
%     aa2=CurrMat(:,currv(2));
%     r1=ShapeInf(aa1(2),aa1(1));
%     r2=ShapeInf(aa2(2),aa2(1));
%     P1=SpherepointTransform(r1,aa1(1)*deltaAlt,aa1(2)*deltaAzi)
%     %P1=Centerp+P1;
%     P2=SpherepointTransform(r2,aa2(1)*deltaAlt,aa2(2)*deltaAzi)
%     %P2=Centerp+P2;
%    
%     flag=JudgeApointInPyramid(Centerp,P0,P1,P2,Point)
%     PP=[P0,P1,P2];
%     Flag(i)=flag;
% end
% if sum(Flag)==4;
%     flag=1;
% else
%     flag=0;
% end
% 

function PointSphere=PointToSphereCoordinate(Point)
%x=R*sin(Alti)*cos(Azi)
%y=R*sin(Alti)*sin(Azi)
%z=R*cos(Alti)

R=norm(Point);
PointSphere=zeros(3,1);
if R>0
    r=sqrt(Point(1)^2+Point(2)^2);
    Alti=atan2(r,abs(Point(3)));
    if (Point(3)<0)
        Alti=3.1415926-Alti;
    end
    dx=Point(1)/abs(R*sin(Alti)+0.001);
    dy=Point(2)/abs(R*sin(Alti)+0.001);
    if dy<0
        acos_dx=2*3.1415926-acos(dx);
    else
        acos_dx=acos(dx);
    end
    PointSphere(1)=R;
    PointSphere(2)=Alti;
    PointSphere(3)=acos_dx;
end



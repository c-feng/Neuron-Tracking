function [CurvesetInf,MapOrig]=UpdatedCurvesMap(AddPoints,CurvesetInf0,MapOrig0)
MapOrig=MapOrig0;
CurvesetInf=CurvesetInf0;
Nums=size(AddPoints,2);
[Nx,Ny,Nz]=size(MapOrig);

for i=1:Nums
     Point=AddPoints(1:3,i);
     Indexx=max(Point(1)-3,1):min(Point(1)+3,Nx);
     Indeyy=max(Point(2)-3,1):min(Point(2)+3,Ny);
     Indezz=max(Point(3)-2,1):min(Point(3)+2,Nz);
     MapOrig(Indexx,Indeyy,Indezz)=0;
end

for i=1:Nums
     s=AddPoints(:,i);
     Curve0=CurvesetInf{s(5)};
     Numss=size(Curve0,2);
     if Numss==1
         Curve00=zeros(7,2);
         s1=(s(1:3)-Curve0(1:3));
         s1=s1./max(norm(s1),0.01);
         if s1'*Curve0(4:6)>0
             Curve00(:,1)=Curve0;
             Curve00(1:3,2)=s(1:3);
             Curve00(4:6,2)=(s1+Curve0(4:6))/norm(s1+Curve0(4:6));
             Curve00(7,2)=0.5;
         else
             Curve00(:,2)=Curve0;
             Curve00(1:3,1)=s(1:3);
             Curve00(4:6,1)=(s1+Curve0(4:6))/norm(s1+Curve0(4:6));
             Curve00(7,1)=0.5;
         end
         CurvesetInf{s(5)}=Curve00;
     end
      if Numss>1
         Curve00=zeros(7,Numss+1);
         s0=norm(s(1:3)-Curve0(1:3,1));
         s1=norm(s(1:3)-Curve0(1:3,end));
         if s0<s1
             Curve00(1:3,1)=s(1:3);
             Curve00(7,1)=0.5;
             Curve00(4:6,1)=(Curve0(1:3,1)-s(1:3))./norm(Curve0(1:3,1)-s(1:3));
             Curve00(:,2:end)=Curve0;
         else
             Curve00(1:3,end)=s(1:3);
             Curve00(4:6,end)=(s(1:3)-Curve0(1:3,1))./norm(Curve0(1:3,1)-s(1:3));
             Curve00(7,end)=0.5;
             Curve00(:,1:end-1)=Curve0;
         end
      end
      CurvesetInf{s(5)}=Curve00;
end




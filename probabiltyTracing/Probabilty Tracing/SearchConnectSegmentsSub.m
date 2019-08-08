function [CurveIds,CurvesLabel,PPoints]=SearchConnectSegmentsSub(Point,SegSet,CurvesLabel,Thre)

CurveIds=zeros(4,1000);
%  the 1st row represents Point Id;
%  the  2rnd row represents the curve Id, 
% whose terminal point matchs with the point in Point;
% if this terminal point is starpoint, the element in 3th row is 1, else
% the element in 4th is zero. 
Numss=size(Point,2);
CurvesNum=size(CurvesLabel,2);
for ii=1:Numss
    pp=Point(1:3,ii);
    kk=0;
    flag=0;
    for jj=1:CurvesNum
        if CurvesLabel(1,jj)==0
            Currcurve=SegSet{jj};
            P0=Currcurve(1:3,1);
            P1=Currcurve(1:3,size(Currcurve,2));
            dist0=norm(pp-P0);
            dist1=norm(pp-P1);
            if dist0<dist1&&dist0<Thre
                kk=kk+1;
                flag=1;
                CurveIds(:,kk)=[ii;jj;1;0];
                CurvesLabel(jj)=1;
                
            end
            if dist0>dist1&&dist1<Thre
                kk=kk+1;
                flag=1;
                CurveIds(:,kk)=[ii;jj;0;1];
                CurvesLabel(jj)=1;
            end
        end
    end
    if flag==0
        kk=kk+1;
        CurveIds(1,kk)=ii;
    end
end


CurveIds=CurveIds(:,1:kk);
PPoints=zeros(3,kk);
dd=0;

for ii=1:kk
    if CurveIds(2,ii)~=0
        CurrCurve=SegSet{CurveIds(2,ii)};
        if CurveIds(3,ii)==1
            dd=dd+1;
            PPoints(:,dd)=CurrCurve(1:3,1);
        else
            dd=dd+1;
            PPoints(:,dd)=CurrCurve(1:3,size(CurrCurve,2));
        end
    end
end

if dd==0
    PPoints=[];
else
    PPoints=PPoints(:,1:dd);
end





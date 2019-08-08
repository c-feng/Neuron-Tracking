function Indexvec=SearchNeighbor(Point,Labelcurve,CurveSet,Thre)

Numxx=size(CurveSet,2);
Indexvec=zeros(4,100);
kk=0;
for i=1:Numxx
    if i~=Labelcurve
        Curve=CurveSet{i};
        [Minv1,Index1]=distPointToCurve(Point,Curve);
        if Minv1<Thre
            kk=kk+1;
            flag0=1;
            if Index1==1
                flag0=0;
            end
            if Index1==size(Curve,2)
                flag0=2;
            end
            Indexvec(:,kk)=[Labelcurve;i;Minv1;Index1;flag0];
        end
    end
end
if kk>0
    Indexvec=Indexvec(:,1:kk);
else
    Indexvec=[];
end






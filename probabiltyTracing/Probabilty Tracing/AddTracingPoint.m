function Addcurve=AddTracingPoint(curve0,MapOrig,flag)

if size(curve0,2)==1
    Point=curve0(1:3);
    Dirc=curve0(4:6);
    for ii=1:300
        Dirc1=Dirc+0.1*normrnd(0,1,3,1);
        Dirc1=Dirc1./max(norm(Dirc1),0.01);
        point0=Point+(2+rand())*sign(rand()-0.5)*Dirc1;
        [w,point0]=MaxNeighbourPoint(point0,MapOrig);
        if w>0.5
            break
        end
    end
    if ii<300
        Addcurve=[point0;w];
    else
        Addcurve=zeros(4,1);
    end
else
    if flag==0
        Point=curve0(1:3,1);
        Dirc=curve0(4:6,1);
        for ii=1:300
            Dirc1=Dirc+0.1*normrnd(0,1,3,1);
            Dirc1=Dirc1./max(norm(Dirc1),0.01);
            point0=Point-(1.5+1.5*rand())*Dirc1;
            [w,point0]=MaxNeighbourPoint(point0,MapOrig);
            dircc=(point0-Point)/max(norm(point0-Point),0.01);
            if w>0.6&&dircc'*Dirc<-0.8
                break
            end
        end
        if ii<300
            Addcurve=[point0;w];
        else
            Addcurve=zeros(4,1);
        end
    end
    
    if flag==1
        Point=curve0(1:3,end);
        Dirc=curve0(4:6,end);
        for ii=1:300
            Dirc1=Dirc+0.1*normrnd(0,1,3,1);
            Dirc1=Dirc1./max(norm(Dirc1),0.01);
            point0=Point+(1.5+1.5*rand())*Dirc1;
            [w,point0]=MaxNeighbourPoint(point0,MapOrig);
            
            if w>0.6
                break
            end
        end
        if ii<300
            Addcurve(1:4,1)=[point0;w];
        else
            Addcurve=zeros(4,1);
        end
    end
end


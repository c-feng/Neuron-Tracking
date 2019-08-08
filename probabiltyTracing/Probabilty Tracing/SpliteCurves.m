function  CurveSet=SpliteCurves(CurveSet0)
Levels=size(CurveSet0,1);
CurveSet=cell(1,1000);
kk=0;
for i=1:Levels-1
    Numss=size(CurveSet0(i,:),2);
    for jj=1:Numss
        curve=CurveSet0{i,jj};
        if isempty(curve)==0
            [splitcurves,CurrSet]=SpliteCurvesSub(curve,CurveSet0(i+1,:));
            CurveSet0(i+1,:)=CurrSet;
            for ss=1:size(splitcurves,2)
                kk=kk+1;
                CurveSet{kk}=splitcurves{ss};
            end
        end
    end
end
if kk>0
    CurveSet=CurveSet(1:kk);
    SinglePoints=zeros(3,kk);
    kk0=0;
    for ii=1:kk
        if size(CurveSet{ii},2)==1
            kk0=kk0+1;
            SinglePoints(:,kk0)=CurveSet{ii};
        end
    end
    
    for ii=1:kk
        if size(CurveSet{ii},2)>1
            CurrCurve=CurveSet{ii};
            Starp=CurrCurve(:,1);
            Endp=CurrCurve(:,end);
            minv0=distPointToCurve(Starp,SinglePoints)
            minv1=distPointToCurve(Endp,SinglePoints)
            if minv0<0.5
                CurrCurve=CurrCurve(:,2:end);
            end
            if minv1<0.5
                if size(CurrCurve,2)>1
                    CurrCurve=CurrCurve(:,1:end-1);
                else
                    CurrCurve=[]
                end
            end
            if isempty(CurrCurve)==0
                CurveSet{ii}=CurrCurve;
            end
        end
    end
end





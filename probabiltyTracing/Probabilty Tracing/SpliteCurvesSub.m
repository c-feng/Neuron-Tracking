function [SplitCurves,CurveSet00]=SpliteCurvesSub(curve,CurveSet0)

Numc=size(curve,2);
Label0=zeros(1,Numc);
Num1=size(CurveSet0,2);
CurveSet00=cell(1,Num1);

for i=1:Num1
    curve11=CurveSet0{i};
    if isempty(curve11)==0
        starP=curve11(1:3,1);
        endP=curve11(1:3,size(curve11,2));
        [Minv1,Index1]=distPointToCurve(starP,curve);
        [Minv2,Index2]=distPointToCurve(endP,curve);
        if Minv1<1
            Label0(Index1)=1;
            curve11=curve11(:,2:end);
        end
        if Minv2<1
            Label0(Index2)=1;
            curve11=curve11(:,1:end-1);
        end
        CurveSet00{i}=curve11;
    end
end

IndexSet=findLabelsInVec(Label0,0);
Indexs=find(Label0==1);
Num2=size(IndexSet,2);


if Num2>0
    SplitCurves=cell(1,Num2+length(Indexs));
    for ii=1:Num2
        SplitCurves{ii}=curve(:,IndexSet{ii});
    end
    for ii=Num2+1:Num2+length(Indexs)
        SplitCurves{ii}=curve(:,Indexs(ii-Num2));
    end
else
    SplitCurves=curve;
end


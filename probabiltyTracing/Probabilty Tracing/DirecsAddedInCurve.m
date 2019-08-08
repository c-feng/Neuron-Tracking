function CurvesetInf=DirecsAddedInCurve(CurveSet,BinaryImg,OrigImg)

Numss=size(CurveSet,2);
 CurvesetInf=cell(1,Numss);
for i=1:Numss
    CurrCurve=CurveSet{i};
    if size(CurrCurve,2)==1
        datap=CurrCurve;
         [corretdata,pdirc,Ratio]=CorrecPointwithPrincipal(datap,BinaryImg,OrigImg);
         CurvesetInf{i}=[corretdata;pdirc;Ratio];
    else
        Currves=zeros(7,size(CurrCurve,2));
        Currves(7,:)=0.5;
        for ii=1:size(CurrCurve,2)-1
            dirc=CurrCurve(:,ii+1)-CurrCurve(:,ii);
            dirc=dirc./max(norm(dirc),0.001);
            Currves(1:3,ii)=CurrCurve(:,ii);
            Currves(4:6,ii)=dirc;
        end
          Currves(1:3,end)=CurrCurve(:,end);
          Currves(4:6,end)=Currves(4:6,end-1);
          CurvesetInf{i}=Currves;
    end
end


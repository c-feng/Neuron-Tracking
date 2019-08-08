function [Idvec1,probcon]=SearchConnectCurveSet(CurveSet,Idcurve,flag,thre)

curve0=CurveSet{Idcurve};
if flag==0
    data0=curve0(1:3,1);
    dirc0=curve0(4:6,1);
else
    data0=curve0(1:3,end);
    dirc0=curve0(4:6,end);
end

Nums=size(CurveSet,2);
kk=0;
Idvec=zeros(4,50);

for i=1:Nums
    Currdata=CurveSet{i};
    if i~=Idcurve
        s1=norm(Currdata(1:3,1)-data0);
        s2=norm(Currdata(1:3,end)-data0);
        
        if s1<thre&s1<=s2
        kk=kk+1;
        Idvec(1,kk)=i;
        Idvec(2,kk)=0;
        tts=0.5*(Currdata(1:3,1)-data0);
        dirc1=Currdata(4:6,end);
        a1=sqrt(norm(tts)^2-(abs(tts'*dirc1))^2);
        a2=sqrt((norm(tts))^2-(abs(tts'*dirc0))^2);
         Idvec(3,kk)=0.2*s1+0.5*(a1+a2);
        
        Idvec(4,kk)=Currdata(4:6,1)'*dirc0;
        end
        if s2<thre&s2<=s1
        kk=kk+1;
        Idvec(1,kk)=i;
        Idvec(2,kk)=1;
        tts=0.5*(Currdata(1:3,end)-data0);
        dirc1=Currdata(4:6,end);
        a1=sqrt(norm(tts)^2-(abs(tts'*dirc1))^2);
        a2=sqrt((norm(tts))^2-(abs(tts'*dirc0))^2);
        Idvec(3,kk)=0.1*s2+0.5*(a1+a2);
        Idvec(4,kk)=Currdata(4:6,end)'*dirc0;
        end
    end
end

if kk>0
    Idvec=Idvec(:,1:kk);
    Indexx=deletNumsRepeat(Idvec(1,:));
    Idvec1=Idvec(:,Indexx);
    probcon=zeros(1,size(Idvec1,2));
    for ii=1:size(Idvec1,2)
        xx=probdist_corr(Idvec1(3,ii),Idvec1(4,ii));
        probcon(ii)=xx;
    end
   aax=0.1*sum(probcon)+0.001;
    %probcon=probcon./(sum(probcon)+aax)
    %probcon=probcon-aax/(sum(probcon)+aax);
    probcon=probcon-0.1;
    Idvec1=[Idvec1;probcon];
else
    Idvec1=[];
end











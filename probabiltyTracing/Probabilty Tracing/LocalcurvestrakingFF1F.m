function [MMk,vmmk,idexx,x1,x2,x3,radiuss,rav]=LocalcurvestrakingFF1F(dataL1,T1,T2,T3,ttk_Label,threv,XXv);
MMk=[];
vmmk=[];
idexx=1;
x1=T1;
x2=T2;
x3=T3;

radiuss=0;
rav=0;

%aaww=zeros(3,15);

for i=1:1


% [dataS,W,data11,x10,x11]=localpointsextracFsskk33(dataL1,XXv,threv,x1,stz); 

   [dataS,W,data11,x10,x11]=localpointsextracFsskk333(dataL1,XXv,threv,x1,ttk_Label);
   nxxs=size(dataS,2);
   
   W1=sort(W,'descend');
   dsw=min(20,length(W1));
   if dsw>10
   thrdk=mean(W1(1:dsw));
   
   else
       thrdk=0;
   end

 if  nxxs>max(.1*threv,5)&thrdk>max([.3*threv,4.5*sqrt(threv)]);
% if  nxxs>max(.1*threv,5)&thrdk>max([.25*threv,3*sqrt(threv)]);
    [P,g,H,sigmaH,kk]=LocalPC1(dataS,data11,.15*eye(3),W);
   
    if P>0
    %[s,v,d]=svd(sigmaH);
    
    if abs(det(sigmaH)/(det(inv(sigmaH))+0.0001))<10       
     % [x1,x2,x3]=gproPCAtotalFF(sign(det(sigmaH))*inv(sigmaH),T1./norm(T1),x11,0.9);
      [x1,x2,x3]=gproPCAtotalFF(sign(det(sigmaH))*inv(sigmaH),T1./norm(T1),x11,0.9);
    end
    else
        idexx=0;
    end
    
    
 %[s,v,d]=svd(sigmaH);


    
KK1=[x3,x2,x1];
%[T1,x11,x1]
MMk=-(KK1(:,1)*KK1(:,1)'+KK1(:,2)*KK1(:,2)')*kk+data11;
dataL1=MMk;
vmmk=KK1(:,3);
radiuss=0;
rav=0;
else
    idexx=0;
    break
end
end




















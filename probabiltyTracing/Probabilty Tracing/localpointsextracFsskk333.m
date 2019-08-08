function [dataS0,W,data11,x10,x11,AAj,AAj2]=localpointsextracFsskk333(data1,XXv,threv,x1,ttk_Label)

[nx,ny,nz]=size(XXv);

idexxs=max(round(data1(1)-7),1):min(round(data1(1)+7),nx);
idexys=max(round(data1(2)-7),1):min(round(data1(2)+7),ny);
idexzs=max(round(data1(3)-6),1):min(round(data1(3)+6),nz);
mxx1=data1(1)-idexxs(1)+1;
myy1=data1(2)-idexys(1)+1;
mzz1=data1(3)-idexzs(1)+1;
ML1=[mxx1,myy1,mzz1]';
XXV1=XXv(idexxs,idexys,1:nz);
Lssx=min(size(XXV1));

if ttk_Label==0;
aaa=-0.9;
bbb=0.85;
else
    aaa=-0.95;
    bbb=0.95;
end




if Lssx>3&nz>3
AAj=zeros(4,216*4);
k=0;

for i=0:1/18*pi:35/18*pi
    for j=0:1/18*pi:(17/18)*pi
         ds1=[sin(j)*cos(i),sin(j)*sin(i),cos(j)]';
         jd=ds1'*x1;
         if jd<aaa
         mms=[ML1+ds1,ML1+2*ds1,ML1+3*ds1,ML1+4*ds1,ML1+5*ds1];
         akmmv=weigthvalue(mms,XXV1);
         arra=ray_burstsampling1(akmmv,threv+3.5*sqrt(threv));
         k=k+1;
         AAj([1:3],k)=ds1';
         AAj(4,k)=mean(akmmv(1:3))+0*arra;
        
         end
    end
end

if k>1
[idexv1,idexx]=max(AAj(4,1:k));
xss=AAj(:,idexx);
xss(1:3)=xss(1:3)./norm(xss(1:3));
x10=xss(1:3);
else
    x10=-x1;
end
else
    x10=-x1;
end
 AAsss1=zeros(36,18);

if Lssx>3&nz>3
AAj2=zeros(4,216*4);
k2=0;
for i=0:1/18*pi:35/18*pi
    for j=0:1/18*pi:(17/18)*pi
        ds1=[sin(j)*cos(i),sin(j)*sin(i),cos(j)]';
        jd=ds1'*x1;
      if jd>bbb
       for tt=1:1 
          mms=[ML1+ds1,ML1+2*ds1,ML1+3*ds1,ML1+4*ds1,ML1+5*ds1,ML1+6*ds1];
          akmmv=weigthvalue(mms,XXV1);
          arra=ray_burstsampling1(akmmv,threv+6*sqrt(threv));
         AAsss1(round(i*18+0.5),round(j*18+0.5))=arra;
        end
        k2=k2+1;
        AAj2([1:3],k2)=ds1';
        AAj2(4,k2)=mean(akmmv(1:3))+0*arra;
       end
    end 
end
if k2>1
[idexv1,idexx]=max(abs(AAj2(4,1:k2)));

xss=AAj2(:,idexx);
xss(1:3)=xss(1:3)./norm(xss(1:3));
x11=xss(1:3);
else
    x11=x1;
end
else
    x11=x1;
end

 
 dataSS=zeros(8*125,4);
 kk0=0;
 data10=data1;
 %data10(3)=(1+nz)/2;
 
 for i=0:2
       mms=(data10+i*x10);
       idet1=max(min(round(mms(1)-2),nx),1):min(max(round(mms(1)+2),1),nx);
       idet2=max(min(round(mms(2)-2),ny),1):min(max(round(mms(2)+2),1),ny);
       idet3=max(min(round(mms(3)-2),nz),1):min(max(round(mms(3)+2),1),nz);
       nt1=length(idet1);
       nt2=length(idet2);
       nt3=length(idet3);
       for it1=1:nt1
           for it2=1:nt2
               for it3=1:nt3
                   kk0=kk0+1;
                   dataSS(kk0,:)=[idet1(it1),idet2(it2),idet3(it3),max(XXv(idet1(it1),idet2(it2),idet3(it3))-threv-0*sqrt(threv),0)]; 
               end
           end
       end     
 end

 
 data10=data1;

 for i=1:2
       mms=(data10+i*x11);
       idet1=max(min(round(mms(1)-2),nx),1):min(max(round(mms(1)+2),1),nx);
       idet2=max(min(round(mms(2)-2),ny),1):min(max(round(mms(2)+2),1),ny);
       idet3=max(min(round(mms(3)-2),nz),1):min(max(round(mms(3)+2),1),nz);
       nt1=length(idet1);
       nt2=length(idet2);
       nt3=length(idet3);
       for it1=1:nt1
           for it2=1:nt2
               for it3=1:nt3
                   kk0=kk0+1;
                   dataSS(kk0,:)=[idet1(it1),idet2(it2),idet3(it3),max(XXv(idet1(it1),idet2(it2),idet3(it3))-threv-0*sqrt(threv),0)]; 
               end
           end
       end     
 end
 
 if kk0>0
dataS=(dataSS(1:kk0,:));

dataS=deletrepeteddataT(dataS');

dataS(3,:)=dataS(3,:);%+stz-1;
W=dataS(4,:);
dataS0=dataS(1:3,:);
data11=data1;
 else
     dataS0=[];
     W=[];
     data11=data1;
 end
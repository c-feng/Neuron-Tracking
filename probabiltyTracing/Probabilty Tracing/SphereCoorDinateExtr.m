function [ExtractPoints,InitialU,Sphere_XX]=SphereCoorDinateExtr(L_XX3,L_data,r0,Theta,Phi)

PI = 3.141592654;
Sphere_XX = zeros(length(r0),Theta, Phi);
[nx,ny,nz]=size(L_XX3);
a = 360/Theta;
b = 180/Phi;
stepp=r0(2)-r0(1);
 for k=1:length(r0)
     for i = 1 : Theta
         for j = 1 : Phi
            x = ( r0(k)*sin( b*j*PI/180 )*cos( a*i*PI/180 ) + L_data(1) );        
            y = ( r0(k)*sin( b*j*PI/180 )*sin( a*i*PI/180 ) + L_data(2) );
            z = ( r0(k)*cos( b*j*PI/180 ) +  L_data(3) );   
            dd=0; ww=0;
            flag=(x<0||x>nx+1)||(y<0||y>ny)||(z<0||z>nz);
            if flag==0
            idexx=max(min(round(x),nx),1);
            idexy=max(min(round(y),ny),1);
            idexz=max(min(round(z),nz),1);
      
            w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
            dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
            ww=ww+exp(w1);
        
          idexx=max(min(round(x+1),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z),nz),1);
          w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
          dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
          idexx=max(min(round(x-1),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z),nz),1);
          w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
          dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
           
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y-1),ny),1);
          idexz=max(min(round(z),nz),1);
            w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y+1),ny),1);
          idexz=max(min(round(z),nz),1);
          w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
          dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z)-1,nz),1);
          w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
          dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z+1),nz),1);
          w1=max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
          dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          Sphere_XX (k,i,j) = dd/(ww+0.0001);
            else
                Sphere_XX (k,i,j) =0;
            end
        end
    end
 end

 
 InitialU=zeros(Theta,Phi);
 for i=1:Theta
     for j=1:Phi
         aa0=Sphere_XX(:,i,j);
         Indexx=find(aa0>0.75);
         if isempty(Indexx)==0
             InitialU(i,j)=Indexx(length(Indexx));
         else
             InitialU(i,j)=0;
         end
     end
 end
 
 Bsize=round(r0(end)+1);
 Indexx=max(1, L_data(1) -Bsize):min(nx, L_data(1) +Bsize);
 Indeyy=max(1, L_data(2) -Bsize):min(ny, L_data(2) +Bsize);
 Indezz=max(1, L_data(3) -Bsize):min(nz, L_data(3) +Bsize);
 ExtractPoints=zeros(3,5000);
 kk=0;
 InitialU=round(stepp*InitialU);
 for ii=1:length(Indexx)
     for jj=1:length(Indeyy)
         for ij=1:length(Indezz)
             if L_XX3(Indexx(ii),Indeyy(jj),Indezz(ij))>0
                 currv=[Indexx(ii);Indeyy(jj);Indezz(ij)];
                  flag=JudgePointInRecshape(L_data,InitialU,currv);
                 if flag==1
                     kk=kk+1;
                      ExtractPoints(:,kk)=currv;
                 end
             end
         end
     end
 end
 if kk>0
     ExtractPoints=ExtractPoints(:,1:kk);
 else
     ExtractPoints=[];
 end
 
 
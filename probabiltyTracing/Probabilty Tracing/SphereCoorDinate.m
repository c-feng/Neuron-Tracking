function [InitialU,Sphere_XX,DiffSphere_XX,SegmentS]=SphereCoorDinate(L_XX3,L_data,r0,Theta,Phi)

PI = 3.141592654;
Sphere_XX = zeros(length(r0),Theta, Phi);
[nx,ny,nz]=size(L_XX3);
a = 360/Theta;
b = 180/Phi;

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

 [Nx,Ny,Nz]=size(Sphere_XX);
 DiffSphere_XX=zeros(Nx,Ny,Nz); 
 for i=1:Ny
     for j=1:Nz
         for ij=1:Nx-1
             aa=Sphere_XX(ij,i,j);
             aa1=Sphere_XX(ij+1,i,j);
             DiffSphere_XX(ij,i,j)=max(aa-aa1,0);
         end
     end
 end
 
 InitialU=zeros(Theta,Phi);
 for i=1:Theta
     for j=1:Phi
         aa0=DiffSphere_XX(:,i,j);
         %aa1=Sphere_XX(:,i,j);
         Indexx=find(aa0>0.35);
         if isempty(Indexx)==0
             InitialU(i,j)=Indexx(1);
         else
             InitialU(i,j)=0.5*Nx;
         end
     end
 end
 
 
 for i=1:Theta
     for j=1:Phi
         aa=DiffSphere_XX(:,i,j);
         aa=MeanfilterS(aa,10);
         DiffSphere_XX(:,i,j)=aa./max(max(aa),0.01);
     end
 end
 
 
 
 Ux = 5*zeros(Theta+2,Phi+2);
 Ux(2:Theta+1,2:Phi+1)=InitialU;
 Ux(:,1)=Ux(:,2);
 Ux(:,Phi+2)=Ux(:,Phi+1);
 
 Ux(1,:)=Ux(2,:);
 Ux(Theta+2,:)=Ux(Theta+1,:);
 
idexxS=1:length(r0);
for JJ=1:30
    for i=2:Theta+1
        for j=2:Phi+1
            pp0=Ux(i,j);
            TT=DiffSphere_XX(:,i-1,j-1);
            JJt=TT'.*exp(-0.05*(idexxS-pp0).^2);
            Ux(i,j)=.6*(sum(idexxS.*JJt)/sum(JJt))+0.4*(1/4)*(Ux(i-1,j)+Ux(i,j-1)+Ux(i+1,j)+Ux(i,j+1));
        end
        Ux=max(Ux,0);
        Ux(1,:)=Ux(3,:);
        Ux(Theta+2,:)=Ux(Theta,:);
        Ux(:,1)=Ux(:,3);
        Ux(:,Phi+2)=Ux(:,Phi);
    end
end
 SegmentS=Ux(2:Theta+1,2:Phi+1);
 
 
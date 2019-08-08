function aa_data=weigthvalue(dataL1,L_XX3)
            nxss=size(dataL1,2);
            aa_data=zeros(1,nxss);
           
            

            [nx,ny,nz]=size(L_XX3);
            for i=1:nxss

            x=dataL1(1,i);
            y=dataL1(2,i);
            z=dataL1(3,i);
            dd=0; ww=0;
            idexx=max(min(round(x),nx),1);
            idexy=max(min(round(y),ny),1);
            idexz=max(min(round(z),nz),1);
      
            w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
            dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
            ww=ww+exp(w1);
            
            idexx=max(min(round(x+1),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z),nz),1);
                 w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
          idexx=max(min(round(x-1),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z),nz),1);
           w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
           ww=ww+exp(w1);
           
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y-1),ny),1);
          idexz=max(min(round(z),nz),1);
           w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
           idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y+1),ny),1);
          idexz=max(min(round(z),nz),1);
            w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
          ww=ww+exp(w1);
          
           idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z)-1,nz),1);
            w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
           ww=ww+exp(w1);
          
          idexx=max(min(round(x),nx),1);
          idexy=max(min(round(y),ny),1);
          idexz=max(min(round(z+1),nz),1);
           w1=.05*max(-2*((x-idexx)^2+(y-idexy)^2+(2*z-2*idexz)^2),-6);
           dd=dd+exp(w1)*L_XX3(idexx,idexy,idexz);
           ww=ww+exp(w1);
          aa_data(i)= dd/(ww+0.0001);
            end
  
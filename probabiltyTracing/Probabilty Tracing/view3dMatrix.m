function SS=view3dMatrix(XMatrix,Indexx,flag0)

[Nx,Ny,Nz]=size(XMatrix);

if flag0==1
    SS=zeros(Ny,Nz);
    for ii=1:Ny
        for jj=1:Nz
            SS(ii,jj)=XMatrix(Indexx,ii,jj);
        end
    end
end

if flag0==2
    SS=zeros(Nx,Nz);
    for ii=1:Nx
        for jj=1:Nz
            SS(ii,jj)=XMatrix(ii,Indexx,jj);
        end
    end
end

if flag0==3
    SS=zeros(Nx,Ny);
    for ii=1:Nx
        for jj=1:Ny
            SS(ii,jj)=XMatrix(ii,jj,Indexx);
        end
    end
end




function Dataset=Extract3Dpoints(XXBinary)

[Nx,Ny,Nz]=size(XXBinary);
Dataset=zeros(3,round(Nx*Ny*Nz));
kk=0;
for i=1:Nx
    for j=1:Ny
        for ij=1:Nz
            if XXBinary(i,j,ij)==1
                kk=kk+1;
                Dataset(:,kk)=[i;j;ij];
            end
        end
    end
end

if kk>0
    Dataset=Dataset(:,1:kk);
else
    Dataset=[];
end








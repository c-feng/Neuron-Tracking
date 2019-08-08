function [DataP,ImgBinary]=RegiongrowsubModify(datap,ImgBinary,Labeflag,Thresize)

Nums=size(datap,2);
DataP=zeros(3,1000*Nums);
kk=0;
[Nx,Ny,Nz]=size(ImgBinary);

for iii=1:2
    for i=1:Nums
        datapp=round(datap(:,i));
        Indexx=max(datapp(1)-2*Thresize,1):min(datapp(1)+2*Thresize,Nx);
        Indeyy=max(datapp(2)-2*Thresize,1):min(datapp(2)+2*Thresize,Ny);
        Indezz=max(datapp(3)-Thresize,1):min(datapp(3)+Thresize,Nz);
        for ii=1:length(Indexx)
            for jj=1:length(Indeyy)
                for ij=1:length(Indezz)
                    if ImgBinary(Indexx(ii),Indeyy(jj),Indezz(ij))==Labeflag
                        kk=kk+1;
                        DataP(:,kk)=[Indexx(ii);Indeyy(jj);Indezz(ij)];
                        ImgBinary(Indexx(ii),Indeyy(jj),Indezz(ij))=0;
                    end
                end
            end
        end
    end
    if kk>0
        datap=DataP(:,1:kk);
        Nums=kk;
    else
        break
    end
   
end

 if kk>0
        DataP=DataP(:,1:kk);
    else
        DataP=[];
 end





function [Dataset,datacell,ImgBinary0]=RegiongrowModify(datap,ImgBinary,Iterative,MatrixLabel,OriImg,Labelflag)
%[Nx,Ny,Nz]=size(ImgBinary);

Dataset=zeros(3,5e5);
kk=0;
Numpp=size(datap,2);
datacell=cell(1,Iterative);
for i=1:Numpp
    pp=datap(1:3,i);
    ImgBinary(pp(1),pp(2),pp(3))=0;
end
for i=1:Iterative
      [Currdata,ImgBinary]=RegiongrowsubModify(datap,ImgBinary,Labelflag,1);
      
      if isempty(Currdata)==0
          Dataset(:,kk+1:kk+size(Currdata,2))=Currdata;
          kk=kk+size(Currdata,2);
          datap=Currdata;
          Label0=FindPointsLabel(Currdata,MatrixLabel);
          aa_data=weigthvalue(Currdata,OriImg);
          datacell{i}=[Currdata;aa_data;Label0];
          Indexxx=i;
      else
          break
      end
end
if kk>0
  Dataset=Dataset(:,1:kk);
  datacell= datacell(1:Indexxx);
else
    Dataset=[];
    datacell=[];
end
ImgBinary0=ImgBinary;
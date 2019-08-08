function [Dataset,datacell,ImgBinary0]=RegiongrowModify(datap,ImgBinary,Iterative,OriImg,Labelflag)
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
          aa_data=weigthvalue(Currdata,OriImg);
          datacell{i}=[Currdata;aa_data];
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
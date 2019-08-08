function [CenterSet,PartPointSet]=PeakdensityClusterTrace(Datap)
%Datap(4,:)=1;
LabelId=Datap(5,:);
Idvec=[0,find(diff(LabelId)>0.5),length(LabelId)];
CenterSet=cell(1,length(Idvec)-1);
PartPointSet=cell(1,length(Idvec)-1);
for i=1:length(Idvec)-1
    Indexx=Idvec(i)+1:Idvec(i+1);
    Points=Datap(1:4,Indexx);
    densityP=Datap(6,Indexx);
    Connets=Datap(7,Indexx)-Idvec(i);
    [Centerp,PartPoint]=PeakdensityClusterpoint(Points,densityP,Connets);
   

    CenterSet{i}=Centerp;
    num0=size(PartPoint,2);
    PartPointSet{i}=cell(1,num0);
    for jj=1:num0
         PartPointSet{i}{jj}=PartPoint{jj};
    end
end



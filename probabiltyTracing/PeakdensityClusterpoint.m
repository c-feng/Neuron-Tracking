function [Centerp,PartPoint]=PeakdensityClusterpoint(Points,densityP,Connets)

%densityP=Points(4,:);
peaksId=find(densityP>0.9);
PartPoint=cell(1,length(peaksId));
Centerp=Points(1:4,peaksId);
Num_points=size(Points,2);
dataLabel=zeros(1,Num_points);

for ii=1:length(peaksId)
     dataLabel(peaksId(ii))=-ii;
end

if length(peaksId)==1
    PartPoint{1}=Points;
end

if length(peaksId)>1
    
    for i=1:Num_points
        if dataLabel(i)==0
            [Pathdata,Idflag]=PeakdensityClusterpointsub(i,Connets,dataLabel);
            dataLabel(Pathdata)=Idflag;
        end
    end
    
    for jj=1:length(peaksId)
        Idd=find(dataLabel==-jj);
        PartPoint{jj}=Points(:,Idd);
    end
end









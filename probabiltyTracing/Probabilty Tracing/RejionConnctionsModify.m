function ConnecMatrix=RejionConnctionsModify(CenterPoints,ClusterPointsCell)

Numlevel=length(CenterPoints);
% MaxCluster=1;
% for i=1:Numlevel
%     Currdata=CenterPoints{i};
%     if size(Currdata,2)>MaxCluster
%        MaxCluster=size(Currdata,2);
%     end
% end
ConnecMatrix=cell(1,Numlevel-1);

% dd1=ClusterPointsCell{1};
% NumS=length(dd1);
% CurrVec=[];
% for i=1:NumS
%     CurrVec=[CurrVec,dd1{i}];
% end
% CurrVec0{1}=CurrVec;
% ClusterPointsCell{1}=CurrVec0;



for i=1:Numlevel-1
    ss=ClusterPointsCell{i};
    ss1=ClusterPointsCell{i+1};
    Connevec=RejionConnctionsSub(ss,ss1);
    ConnecMatrix{i}=Connevec;
end
    

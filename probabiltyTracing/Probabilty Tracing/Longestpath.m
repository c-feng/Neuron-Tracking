function [Curves,ExtraLabel,CenterPoints]=Longestpath(ConnecCell,CenterPoints,LevelId,PointId)

CurrVec=CenterPoints{LevelId};
StartNode=CurrVec(:,PointId);
CurrVec(4,PointId)=1;
CenterPoints{LevelId}=CurrVec;

TotalLevel=size(ConnecCell,2);
Curves=zeros(3,TotalLevel);
ExtraLabel=zeros(2,TotalLevel);
kk=1;
Curves(:,kk)=StartNode(1:3);
ExtraLabel(1,kk)=LevelId;
ExtraLabel(2,kk)=PointId;
ExtraLabel(3,kk)=0;
for ij=LevelId+1:1:TotalLevel+1
    SS=ConnecCell{ij-1};
    CurrVec=CenterPoints{ij};

    PointId0=Longestpathsub(SS,PointId,2,CurrVec(4,:));
  
    if PointId0~=0&&CurrVec(4,PointId0)==0
        CurrVec=CenterPoints{ij};
        kk=kk+1;
        Curves(:,kk)=CurrVec(1:3,PointId0);
        PointId=PointId0;
        ExtraLabel(1,kk)=ij;
        ExtraLabel(2,kk)=PointId0;
        ExtraLabel(3,kk)=SS(2,PointId0);
        CurrVec(4,PointId0)=1;
        CenterPoints{ij}=CurrVec;
    else
        %TerminalId=0;
        break
    end
end

%end


if kk>1
    Curves=Curves(:,1:kk);
    ExtraLabel=ExtraLabel(:,1:kk);
else
    Curves=[];
    ExtraLabel=[];
end
%end

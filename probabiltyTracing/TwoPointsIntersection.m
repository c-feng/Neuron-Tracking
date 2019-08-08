function [Flag00,InterSets]=TwoPointsIntersection(Seta,Setb,R)
MergeSet=[Seta,Setb];

minV=min(MergeSet');
minV=minV';
maxV=max(MergeSet');
maxV=maxV';
MinCoord=max(minV-2*R,0);
MaxCoord=maxV+2*R+1;
LabelMatrix=zeros(MaxCoord(1)-MinCoord(1),MaxCoord(2)-MinCoord(2),MaxCoord(3)-MinCoord(3));
NumPointsSeta=size(Seta,2);
NumPointsSetb=size(Setb,2);
for i=1:NumPointsSeta
    aax=Seta(:,i)-minV+1;
    LabelMatrix(aax(1),aax(2),aax(3))=1;
end
Labelvec=zeros(1,NumPointsSetb);
for i=1:NumPointsSetb
    bbx=Setb(:,i)-minV+1;
    s0=max(bbx(1)-R,1);
    s1=max(bbx(2)-R,1);
    s2=max(bbx(3)-R,1);
    ss=LabelMatrix(s0:s0+2*R,s1:s1+2*R,s2:s2+2*R);
    
    if max(ss(:))>0
        
        Labelvec(i)=1;
    end
end

Idexx=find(Labelvec==1);
if isempty(Idexx)==0
    Flag00=1;
    InterSets=Setb(:,Idexx);
else
    Flag00=0;
    InterSets=[];
end

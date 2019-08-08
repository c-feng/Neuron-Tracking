function PointId0=Longestpathsub(SS,PointId,Thre,Lablevec)
ax=0;
Num0=size(SS,2);
for i=1:Num0
    
    if SS(1,i)==PointId && SS(2,i)>ax && Lablevec(i)==0
        ax=SS(2,i);
        PointId0=i;
    end
end

if ax<Thre
    PointId0=0;
end
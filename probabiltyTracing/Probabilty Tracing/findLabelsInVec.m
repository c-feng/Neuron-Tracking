function  IndexSet=findLabelsInVec(Vec,Label0)

Indexx=find(Vec==Label0);
Indexdiff=[0,find(diff(Indexx)>1),length(Indexx)];
IndexSet=cell(1,length(Indexdiff)-1);
for i=1:length(Indexdiff)-1
    IndexSet{i}=Indexx(Indexdiff(i)+1):Indexx(Indexdiff(i+1));
end

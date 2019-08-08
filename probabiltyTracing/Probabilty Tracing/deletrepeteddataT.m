function dataSS=deletrepeteddataT(dataS)
ndd=size(dataS,1);
dataSS=[];
[dataS1,idexx1]=deletrepeteddata(dataS,ones(ndd,1));
nx=size(dataS,2);
nxx=size(idexx1,2);
idexx1=[idexx1,nx+1];
for i=1:nxx
    dataS11=deletrepeteddataL(dataS1(:,idexx1(i):idexx1(i+1)-1));
    dataSS=[dataSS,dataS11];
end




function y1=viewcelldata(BB,idexx,Label);
y1=0;
nx=size(idexx,2);
for i=1:nx
    kks=BB{idexx(i)};
    hold on
 if size(kks,2)>1%&sum(sum(AA))~=0;
     idtxx=[1:1:size(kks,2)];
     if Label==1
     plot3(kks(1,idtxx),kks(2,idtxx),kks(3,idtxx),'ro-','Markersize',3);
     else
      plot3(kks(1,idtxx),kks(2,idtxx),kks(3,idtxx),'ko-','Markersize',3);
     end
     y1=y1+1;
     grid on
 end
 if size(kks,2)==1
     kks
     plot3(kks(1),kks(2),kks(3),'go-','Markersize',5);
 end
end

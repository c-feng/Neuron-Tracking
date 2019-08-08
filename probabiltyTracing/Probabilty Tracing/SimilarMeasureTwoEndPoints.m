function [dist,Corrscore]=SimilarMeasureTwoEndPoints(curve0,curve1,flagvec)

if flagvec(1)==0&&flagvec(2)==0
    dist=norm(curve0(1:3,1)-curve1(1:3,1));
    Corrscore=curve0(4:6,1)'*curve1(4:6,1);
end
if flagvec(1)==0&&flagvec(2)==1
    dist=norm(curve0(1:3,1)-curve1(1:3,end));
    Corrscore=curve0(4:6,1)'*curve1(4:6,end);
    
end

if flagvec(1)==1&&flagvec(2)==0
    dist=norm(curve0(1:3,end)-curve1(1:3,1));
    Corrscore=curve0(4:6,end)'*curve1(4:6,1);
end

if flagvec(1)==1&&flagvec(2)==1
    dist=norm(curve0(1:3,end)-curve1(1:3,end));
    Corrscore=curve0(4:6,end)'*curve1(4:6,end);
end






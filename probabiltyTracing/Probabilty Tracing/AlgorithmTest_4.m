%AlgorithmTest_4.m
% [densityVec,MapOrig]=AddPointProMap(CurveSet,BinaryXX3);
% CurvesetInf=DirecsAddedInCurve(CurveSet,BinaryXX3,XX3);
% Addpoint=AddTracingPointstotal(CurvesetInf,MapOrig,AddPointNum);
[CurvesetInf0,MapOrig]=UpdatedCurvesMap(Addpoint,CurvesetInf,MapOrig);
%Connectvec=ConnectEndPointstotal(CurvesetInf);
Curve0=CurvesetInf{35};
a=AddTracingPoint(Curve0,MapOrig,0);
 
%  Idvec=SearchConnectCurveSet(CurvesetInf,1,1,10);
%  Labelvec=zeros(3,2*size(CurvesetInf,2));
%  kk=0;
%  for i=1:size(CurvesetInf,2)
%      if size(CurvesetInf{i},2)==1
%          kk=kk+1;
%          Labelvec(1,kk)=i;
%          Labelvec(3,kk)=-1;
%      else
%          Labelvec(1,kk+1:kk+2)=i;
%          Labelvec(3,kk+2)=1;
%          kk=kk+2;
%      end
% %  end
%  Labelvec= Labelvec(:,1:kk);
 



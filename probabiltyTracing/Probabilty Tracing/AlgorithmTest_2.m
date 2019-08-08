%AlgorithmTest_2.m

PointSWC=load('1.swc');
PointSWC1=load('7.swc');
PointSWC2=load('24.swc');
PointSWC3=load('32.swc');
PointSWC4=load('33.swc');
Points=[PointSWC(:,[4,3,5])',PointSWC1(:,[4,3,5])',PointSWC2(:,[4,3,5])'];
Points=[Points,PointSWC3(:,[4,3,5])',PointSWC4(:,[4,3,5])'];
XX3=zeros(512,512,77);
for i=1:77
    XX3(:,:,i)=double(imread('01.tif',i));
end
dataP=zeros(4,5e5);
kk=0;
BinaryXX3=zeros(512,512,77);
for i=1:size(Points,2)
    pp=Points(:,i);
    Idexx=max(round(pp(1)-2),1):min(round(pp(1)+2),512);
    Ideyy=max(round(pp(2)-2),1):min(round(pp(2)+2),512);
    Idezz=max(round(pp(3)-1),1):min(round(pp(3)+1),77);
    for ii=1:length(Idexx)
        for jj=1:length(Ideyy)
            for ij=1:length(Idezz)
                if BinaryXX3(Idexx(ii),Ideyy(jj),Idezz(ij))==0
                    kk=kk+1;
                    dataP(:,kk)=[Idexx(ii);Ideyy(jj);Idezz(ij);XX3(Idexx(ii),Ideyy(jj),Idezz(ij))];
                    BinaryXX3(Idexx(ii),Ideyy(jj),Idezz(ij))=1;
                end
            end
        end
    end
end

datap=dataP(1:3,3005);
[Dataset,datacell]=RegiongrowModify(datap,BinaryXX3,1000,XX3,1);
%Datap=PeakdensityCenterPointModify(datacell);
%Datap=PeakdensityCenterPointM(datacell);
Datap=PeakdensityCenterPointMM(datacell,size(XX3));
[CenterSet,PartPointSet]=PeakdensityClusterTrace(Datap);
CMatrix=RejionConnctionsModify(CenterSet,PartPointSet);
[Addcurve,LevelCurve]=ExtracConnectedCurvesPopulation(CMatrix,CenterSet);
CurveSet=SpliteCurves(LevelCurve);
viewcelldata(Addcurve,1:size(Addcurve,2),0)
plot3(dataP(1,:),dataP(2,:),dataP(3,:),'g.')

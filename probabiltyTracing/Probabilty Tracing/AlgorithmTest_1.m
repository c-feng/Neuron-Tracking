% %AlgorithmTest_1.m
% 
% PointSWC=load('1.swc');
% Points=PointSWC(:,[4,3,5])';
% XX3=zeros(512,512,77);
% 
% for i=1:77
%     XX3(:,:,i)=double(imread('01.tif',i));
% end
% 
% dataP=zeros(4,5e5);
% kk=0;
% BinaryXX3=zeros(512,512,77);
% for i=1:size(Points,2)
%     pp=Points(:,i);
%     Idexx=max(round(pp(1)-2),1):min(round(pp(1)+2),512);
%     Ideyy=max(round(pp(2)-2),1):min(round(pp(2)+2),512);
%     Idezz=max(round(pp(3)-1),1):min(round(pp(3)+1),77);
%     for ii=1:length(Idexx)
%         for jj=1:length(Ideyy)
%             for ij=1:length(Idezz)
%                 if BinaryXX3(Idexx(ii),Ideyy(jj),Idezz(ij))==0
%                     kk=kk+1;
%                     dataP(:,kk)=[Idexx(ii);Ideyy(jj);Idezz(ij);XX3(Idexx(ii),Ideyy(jj),Idezz(ij))];
%                     BinaryXX3(Idexx(ii),Ideyy(jj),Idezz(ij))=1;
%                 end
%             end
%         end
%     end
% end
% dataP=dataP(:,1:kk);
% 
 [datatt,InitialU]=SphereCoorDinateExtr(BinaryXX3,dataP(1:3,14054),0:0.3:7,15,30);
ss=datatt(:,7);
ss0=ss;
for i=1:200
    [sigmaH,Shiftvec]=LocalPrinCurve(datatt,ss,0.3*eye(3),ones(1,size(datatt,2)));
    ss1=ss+0.2*sqrt(1/i)*normrnd(0,3,3,1);
    [sigmaH,Shiftvec1]=LocalPrinCurve(datatt,ss1,0.3*eye(3),ones(1,size(datatt,2)));
    if norm(Shiftvec1)<norm(Shiftvec)
        ss=ss1;
    end
end


%[ss0,ss,mean(datatt')']

% figure()
% plot3(dataP(1,:),dataP(2,:),dataP(3,:),'.')
% hold on
% plot3(datatt(1,:),datatt(2,:),datatt(3,:),'ro')



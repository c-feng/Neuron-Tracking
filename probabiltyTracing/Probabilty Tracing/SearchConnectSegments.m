function CellSegs=SearchConnectSegments(ObjectSeg,SegSet,PointId,CurveId,Thre)
Point=ObjectSeg(1:3,PointId);
CurvesLabel=zeros(3,size(SegSet,2));
CurvesLabel(1:3,CurveId)=1;
CellSegs=cell(1,10);
[CurveIds,flagVec]=SearchConnectSegmentsSub(Point,SegSet,CurvesLabel,Thre);












# Generated with SMOP  0.41
from libsmop import *
# RegiongrowModify.m

    
@function
def RegiongrowModify(datap=None,ImgBinary=None,Iterative=None,OriImg=None,Labelflag=None,*args,**kwargs):
    varargin = RegiongrowModify.varargin
    nargin = RegiongrowModify.nargin

    #[Nx,Ny,Nz]=size(ImgBinary);
    Dataset=zeros(3,500000.0)
# RegiongrowModify.m:3
    kk=0
# RegiongrowModify.m:4
    Numpp=size(datap,2)
# RegiongrowModify.m:5
    datacell=cell(1,Iterative)
# RegiongrowModify.m:6
    for i in arange(1,Numpp).reshape(-1):
        pp=datap(arange(1,3),i)
# RegiongrowModify.m:8
        ImgBinary[pp(1),pp(2),pp(3)]=0
# RegiongrowModify.m:9
    
    for i in arange(1,Iterative).reshape(-1):
        Currdata,ImgBinary=RegiongrowsubModify(datap,ImgBinary,Labelflag,1,nargout=2)
# RegiongrowModify.m:12
        if isempty(Currdata) == 0:
            Dataset[arange(),arange(kk + 1,kk + size(Currdata,2))]=Currdata
# RegiongrowModify.m:15
            kk=kk + size(Currdata,2)
# RegiongrowModify.m:16
            datap=copy(Currdata)
# RegiongrowModify.m:17
            aa_data=weigthvalue(Currdata,OriImg)
# RegiongrowModify.m:18
            datacell[i]=concat([[Currdata],[aa_data]])
# RegiongrowModify.m:19
            Indexxx=copy(i)
# RegiongrowModify.m:20
        else:
            break
    
    if kk > 0:
        Dataset=Dataset(arange(),arange(1,kk))
# RegiongrowModify.m:26
        datacell=datacell(arange(1,Indexxx))
# RegiongrowModify.m:27
    else:
        Dataset=[]
# RegiongrowModify.m:29
        datacell=[]
# RegiongrowModify.m:30
    
    ImgBinary0=copy(ImgBinary)
# RegiongrowModify.m:32
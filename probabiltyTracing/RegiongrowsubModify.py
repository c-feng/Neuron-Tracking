# Generated with SMOP  0.41
from libsmop import *
# RegiongrowsubModify.m

    
@function
def RegiongrowsubModify(datap=None,ImgBinary=None,Labeflag=None,Thresize=None,*args,**kwargs):
    varargin = RegiongrowsubModify.varargin
    nargin = RegiongrowsubModify.nargin

    Nums=size(datap,2)
# RegiongrowsubModify.m:3
    DataP=zeros(3,dot(1000,Nums))
# RegiongrowsubModify.m:4
    kk=0
# RegiongrowsubModify.m:5
    Nx,Ny,Nz=size(ImgBinary,nargout=3)
# RegiongrowsubModify.m:6
    for iii in arange(1,2).reshape(-1):
        for i in arange(1,Nums).reshape(-1):
            datapp=round(datap(arange(),i))
# RegiongrowsubModify.m:10
            Indexx=arange(max(datapp(1) - dot(2,Thresize),1),min(datapp(1) + dot(2,Thresize),Nx))
# RegiongrowsubModify.m:11
            Indeyy=arange(max(datapp(2) - dot(2,Thresize),1),min(datapp(2) + dot(2,Thresize),Ny))
# RegiongrowsubModify.m:12
            Indezz=arange(max(datapp(3) - Thresize,1),min(datapp(3) + Thresize,Nz))
# RegiongrowsubModify.m:13
            for ii in arange(1,length(Indexx)).reshape(-1):
                for jj in arange(1,length(Indeyy)).reshape(-1):
                    for ij in arange(1,length(Indezz)).reshape(-1):
                        if ImgBinary(Indexx(ii),Indeyy(jj),Indezz(ij)) == Labeflag:
                            kk=kk + 1
# RegiongrowsubModify.m:18
                            DataP[arange(),kk]=concat([[Indexx(ii)],[Indeyy(jj)],[Indezz(ij)]])
# RegiongrowsubModify.m:19
                            ImgBinary[Indexx(ii),Indeyy(jj),Indezz(ij)]=0
# RegiongrowsubModify.m:20
        if kk > 0:
            datap=DataP(arange(),arange(1,kk))
# RegiongrowsubModify.m:27
            Nums=copy(kk)
# RegiongrowsubModify.m:28
        else:
            break
    
    if kk > 0:
        DataP=DataP(arange(),arange(1,kk))
# RegiongrowsubModify.m:36
    else:
        DataP=[]
# RegiongrowsubModify.m:38
    
    
@function
def RegiongrowModify(datap=None,ImgBinary=None,Iterative=None,MatrixLabel=None,OriImg=None,Labelflag=None,*args,**kwargs):
    varargin = RegiongrowModify.varargin
    nargin = RegiongrowModify.nargin

    #[Nx,Ny,Nz]=size(ImgBinary);
    
    Dataset=zeros(3,500000.0)
# RegiongrowsubModify.m:48
    kk=0
# RegiongrowsubModify.m:49
    Numpp=size(datap,2)
# RegiongrowsubModify.m:50
    datacell=cell(1,Iterative)
# RegiongrowsubModify.m:51
    for i in arange(1,Numpp).reshape(-1):
        pp=datap(arange(1,3),i)
# RegiongrowsubModify.m:53
        ImgBinary[pp(1),pp(2),pp(3)]=0
# RegiongrowsubModify.m:54
    
    for i in arange(1,Iterative).reshape(-1):
        Currdata,ImgBinary=RegiongrowsubModify(datap,ImgBinary,Labelflag,1,nargout=2)
# RegiongrowsubModify.m:57
        if isempty(Currdata) == 0:
            Dataset[arange(),arange(kk + 1,kk + size(Currdata,2))]=Currdata
# RegiongrowsubModify.m:60
            kk=kk + size(Currdata,2)
# RegiongrowsubModify.m:61
            datap=copy(Currdata)
# RegiongrowsubModify.m:62
            Label0=FindPointsLabel(Currdata,MatrixLabel)
# RegiongrowsubModify.m:63
            aa_data=weigthvalue(Currdata,OriImg)
# RegiongrowsubModify.m:64
            datacell[i]=concat([[Currdata],[aa_data],[Label0]])
# RegiongrowsubModify.m:65
            Indexxx=copy(i)
# RegiongrowsubModify.m:66
        else:
            break
    
    if kk > 0:
        Dataset=Dataset(arange(),arange(1,kk))
# RegiongrowsubModify.m:72
        datacell=datacell(arange(1,Indexxx))
# RegiongrowsubModify.m:73
    else:
        Dataset=[]
# RegiongrowsubModify.m:75
        datacell=[]
# RegiongrowsubModify.m:76
    
    ImgBinary0=copy(ImgBinary)
# RegiongrowsubModify.m:78
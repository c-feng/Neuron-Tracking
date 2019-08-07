""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz=800
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True

def showpoints(xyz,c_gt=None, c_pred = None ,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
    cv2.namedWindow('show3d', cv2.WINDOW_NORMAL)
    cv2.moveWindow('show3d',0,0)
    cv2.setMouseCallback('show3d',onmouse)

    dll=np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')

    global showsz,mousex,mousey,zoom,changed
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*2.2)/showsz
    if c_gt is None:
        c0=np.zeros((len(xyz),),dtype='float32')+255
        c1=np.zeros((len(xyz),),dtype='float32')+255
        c2=np.zeros((len(xyz),),dtype='float32')+255
    else:
        c0=c_gt[:,0]
        c1=c_gt[:,1]
        c2=c_gt[:,2]


    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0


    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('show3d',show)
        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256
        if cmd==ord('q'):
            break
        elif cmd==ord('Q'):
            sys.exit(0)

        if cmd==ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            else:
                if c_pred is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True



        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
        if waittime!=0:
            break
    return 


########################
########################
import colorsys
import random

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return np.array(rgb_colors)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

if __name__=='__main__':
    # np.random.seed(100)
    # showpoints(np.random.randn(2500,3))

    #######################################
    data_root = "/home/fcheng/Neuron/pointnet/data/real_data_9600/train"
    # data_root = "/home/fcheng/Neuron/pointnet/data/real_data/test"

    npy_files = os.listdir(data_root)
    
    # for path in npy_files:
        # npy = np.load(os.path.join(data_root, path))
    #     print(path)
    #     print(npy.shape)
    npy = np.load(os.path.join(data_root, npy_files[10]))
    print(npy_files[10])
    labels = np.unique(npy[:, -1])
    print(labels)
    colors = ncolors(len(labels))

    npy_labels = npy[:, -1]
    npy_labels_order = np.zeros(len(npy), dtype=int)
    for cnt, l in enumerate(labels):
        npy_labels_order[npy_labels==l] = cnt

    # coords = pc_normalize(npy[:, :3])
    showpoints(npy[:, :3], c_gt=colors[npy_labels_order], normalizecolor=False, ballradius=2)

    #########################################################
    # from skimage.external import tifffile
    # data_root = "/home/jjx/Biology/data/data_synthesize/ins/"
    # tif_files = os.listdir(data_root)

    # tif = tifffile.imread(os.path.join(data_root, tif_files[0]))
    # coords = np.stack(np.where(tif>0), axis=0).T
    # c_labels = tif[coords[:, 0], coords[:, 1], coords[:, 2]]
    # # coords_l = np.concatenate([coords, labels[:, None]], axis=1)

    # labels = np.unique(c_labels)
    # print(labels)
    # colors = ncolors(len(labels))

    # c_labels_order = np.zeros(len(coords), dtype=int)
    # for cnt, l in enumerate(labels):
    #     c_labels_order[c_labels==l] = cnt
    # showpoints(coords, c_gt=colors[c_labels_order], normalizecolor=False, ballradius=2)
    


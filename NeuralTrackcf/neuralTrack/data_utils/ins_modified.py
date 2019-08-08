import os
import numpy as np
import os.path as osp
import sys

from glob import glob
from skimage.external import tifffile
from skimage.morphology import dilation,ball
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
from treelib import Node, Tree

from ..utils.mask_utils import dilated_mask_generate
from ..utils.osutils import mkdir_if_missing
from ..utils.coords_utils import endpoint_cal
from ..utils.logging import Logger, TFLogger
from ..utils.metrics import calc_bd, soft_metric

def name_to_int(name):
    # name="6950_34600_4150_068.swc"
    # return 68
    return int( name.split(".")[0].split("_")[-1] )

def str_to_float(s):
    """字符串转换为float"""
    if s is None:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def psc(root_dir, file_name, suffix='.swc'):
    ''' process_single_
    '''
    # root_dir = r'C:\Users\Administrator\Desktop\check_swc\Modified_Selected_Dataset\swcs'
    # file_name = '3450_29100_4650_011'

    file_path = os.path.join(root_dir, file_name + suffix)

    # 读取一个.swc文件
    with open(file_path) as f:
        lines = f.readlines()

    # 将一个.swc文件转化为n行的list
    swc = []
    for line in lines:
        line = line.rstrip('\n')
        swc.append(line.split(' '))

    # 将文件读取的字符串转换为数值
    swc_num = []
    for i in swc:
        i = [str_to_float(j) for j in i]
        swc_num.append(i)
    
    return swc_num

def analyze_single_swc(swcs_dir, swc_name):
    # 计算一个.swc文件中两点之前的平均距离
    num_list = np.array(psc(swcs_dir, swc_name))
    nums = len(num_list)
    coords = num_list[:, 2:5]
    coords = coords / [0.2, 0.2, 1.0]  # 比例缩放
    coords = coords[:, ::-1]

    dist = cdist(XA=coords, XB=coords, metric='euclidean')
    dist = [dist[i, i+1] for i in range(nums-1)]
    d_mean = np.mean(dist)
    return d_mean

def branch_connect(ins, end, endpoint_coords, thres = 2):
    labels = []
    
    endpoint_coords_filter = []
    branch_coords_filter = []

    for endpoint_coord in endpoint_coords:
        x,y,z = endpoint_coord 
        endpoint_label = ins[x,y,z]
        range_mask = dilated_mask_generate(endpoint_coord, ins.shape, ball, [thres])
        
        #print("range_mask {} {}".format(np.sum(range_mask), thres))
        nearby_labels = ins[range_mask]
        nearby_end_labels = end[range_mask]

        labels_ = np.unique(nearby_labels).tolist()
        end_labels_ = np.unique(nearby_end_labels).tolist()

        if 0 in labels_:
            labels_.remove(0)

        if len(labels_) >= 2:
            mask = np.zeros_like(ins)
            #print(fname,labels_)
            label_ = min(labels_)
            for l in labels_:
                mask[ins == l] = 1    

            ins[mask > 0] = label_

            labels.extend(labels_)

            branch_coords_filter.append(endpoint_coord)
            if 1 in end_labels_:
                end[range_mask] = 0
                print("{} {} {} now is modified to trunk".format(x,y,z))
            else:
                end[x,y,z] = 2
        else:
            end[x,y,z] = 1

    labels = np.unique(labels).tolist()
    #return ins,labels,endpoint_coords_filter,branch_coords_filter
    return ins, end, labels

def ins_sel(ins, labels):
    ins_ = np.zeros_like(ins)
    for i in labels:
        mask = ins == i
        ins_[mask] = i
    return ins_

def single_run(ins_p,swcs_p,target_dir = None,flag = True):
    fname = osp.basename(ins_p)
    ins = tifffile.imread(ins_p)
    swcs_p.sort()
    infos = []
    endpoint_coords_filter = []
    branch_coords_filter = []
    
    ins_modified_dir = osp.join(target_dir,"ins_modified")
    ends_dir = osp.join(target_dir,"ends")
    
    mkdir_if_missing(ins_modified_dir)
    mkdir_if_missing(ends_dir)

    swcs_p_length = [[x, len(open(x).readlines()) ] for x in swcs_p]
    swcs_p_length.sort(key = lambda x : x[-1], reverse=True)

    end = np.zeros_like(ins)
    for swc_p,length in swcs_p_length:
        fname_swc = osp.basename(swc_p)
        ins_label = name_to_int(fname_swc) + 1
        #density = analyze_single_swc(osp.dirname(swc_p), osp.splitext(fname_swc)[0])

        f = osp.basename(swc_p)
        endpoint_coords, branch_coords, coords, ftree = endpoint_cal(swc_p,1,ins.shape)
        #print("ends: {} branches: {}".format(len(endpoint_coords), len(branch_coords)))

        if len(branch_coords) > 0:
            end[branch_coords[:,0],branch_coords[:,1],branch_coords[:,2]] = 2

        ins, end, labels_= branch_connect(ins, end, endpoint_coords)
        #print(np.sum(range_mask))

        if len(labels_) > 0:
            print(f,labels_)
            infos.append("{} {}".format(f,labels_))
        else:
            ins_ = np.zeros_like(ins)
            ins_r = np.zeros_like(ins)

            ins_[ins == ins_label] = ins_label
            ins_r[np.logical_and(ins > 0, ins != ins_label)] = \
                    ins[np.logical_and(ins > 0, ins != ins_label)]
            
            metric_ = soft_metric(ins_ > 0, ins_r > 0)
            dice_ , prec_, recall_ = metric_[:3]
            print("label {} dice {} prec {} recall {} rate {}".\
                    format(ins_label, *metric_))
            if dice_ > 0.1 or prec_ > 0.1 or recall_ > 0.1:
                '''mask_ = dilation(ins_ > 0, 3)
                matchs_ = np.unique(ins[mask_]).to_list()
                if 0 in matchs_:matchs_ = matchs_.remove(0)
                matchs_r = matchs_.remove(ins_label)
                if len(matchs_r) == 1:
                    label_ = min(matchs_)
                    mask = np.logical_or(ins == match_[0], ins == match_[1])
                    ins[mask] = label_
                else:'''
                mask = ins_ > 0
                ins[mask] = 0
                print("{} is removed".format(ins_label))
                infos.append("{} removed".format(ins_label))




            '''metrics_, matchs_ = calc_bd(ins_, ins_r, 2)
            metric_ = metrics_[0]
            match_ = matchs_[0]

            dice_ , prec_, recall_ = metric_[:3]
            
            label_ =  min(match_)
            print("label {} {} dice {} prec {} recall {} rate {}".\
                    format(*match_, *metric_))
            if dice_ > 0.3 or prec_ > 0.3 or recall_ > 0.3:
                mask = np.logical_or(ins == match_[0], ins == match_[1])
                ins[mask] = label_'''




    target_p = osp.join(ins_modified_dir,fname)
    tifffile.imsave(target_p,ins)
    
    end_p = osp.join(ends_dir,fname)
    tifffile.imsave(end_p,end.astype(np.uint8))

    return "\n".join(infos)

if __name__ == "__main__":
    
    source_dir = "/home/jjx/Biology/data_modified"
    #sys.stdout = Logger(osp.join(data_dir, "merge_log.txt"))
    ins_p = glob(osp.join(source_dir,"ins/*"))

    swcs_p = [glob( osp.join("/home/jjx/Biology/Modified_Selected_Dataset", "swcs/{}*.swc".format(\
                osp.splitext(osp.basename(x))[0]))) for x in ins_p] 

    target_dir = "/home/jjx/Biology/data_modified"
    mkdir_if_missing(target_dir)

    
    with Pool(8) as pool:
        infos = pool.starmap(partial(single_run,target_dir = target_dir),\
                zip(ins_p,swcs_p))

    with open(osp.join(target_dir, "merge_log.txt"),"w") as g:
        g.write("\n".join(infos))


import numpy as np
from skimage.external import tifffile
import math
import os
from skimage.morphology import dilation, ball

def Connect_2_points(p1=[34,50,178], p2=[100,100,106], size=[300,300,300]):
    """ Connect two points in the spaces.
        Filling the gap between two points
    """
    EPS = 1e-7
    mask = np.zeros(shape=size)
    gap_xyz = np.array( [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]] )
    r = np.power(np.power(gap_xyz, 2).sum() , 0.5)
    r_ceil = np.ceil(r)
    theta = math.acos(gap_xyz[2] / (r+EPS))
    phi = math.atan(gap_xyz[1] / (gap_xyz[0]+EPS) )
    # direction = direction_v / ( np.power(np.power(direction_v, 2).sum(), 0.5) )
    
    for delta in range(int(r_ceil)):
        coord_x = p1[0] + delta*math.sin(theta)*math.cos(phi)
        coord_y = p1[1] + delta*math.sin(theta)*math.sin(phi)
        coord_z = p1[2] + delta*math.cos(theta)
        mask[int(coord_x), int(coord_y), int(coord_z)] = 1  # 200

    # debug
    # mask[int(p1[0]), int(p1[1]), int(p1[2])] = 1000  # 2000
    # mask[int(p2[0]), int(p2[1]), int(p2[2])] = 1000  # 2000
    # tifffile.imsave("test.tif", mask.astype(np.int16))
    
    return mask

def test():
    EPS = 1e-7
    p1 = [34,50,178]
    p2 = [100,100,106]
    size=[300,300,300]
    mask = np.zeros(shape=size)
    gap_xyz = np.array( [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]] )
    r = np.power(np.power(gap_xyz, 2).sum() , 0.5)
    r_ceil = np.ceil(r)
    theta = math.acos(gap_xyz[2] / (r+EPS))
    phi = math.atan(gap_xyz[1] / (gap_xyz[0]+EPS) )
    # direction = direction_v / ( np.power(np.power(direction_v, 2).sum(), 0.5) )

    for delta in range(int(r_ceil)):
        coord_x = p1[0] + delta*math.sin(theta)*math.cos(phi)
        coord_y = p1[1] + delta*math.sin(theta)*math.sin(phi)
        coord_z = p1[2] + delta*math.cos(theta)
        mask[int(coord_x), int(coord_y), int(coord_z)] = 200

    mask[int(p1[0]), int(p1[1]), int(p1[2])] = 2000
    mask[int(p2[0]), int(p2[1]), int(p2[2])] = 2000
    tifffile.imsave("test.tif", mask.astype(np.int16))


def connect_2_points_(p1=[34,50,178], p2=[100,100,106], size=[300,300,300]):
    """ Connect two points in the spaces.
        Filling the gap between two points
    """

    mask = np.zeros(shape=size)
    p1_int = np.round(p1).astype(np.int)
    p2_int = np.round(p2).astype(np.int)
    direction_v = np.array( [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]] )
    direction = direction_v / ( np.power(np.power(direction_v, 2).sum(), 0.5) )
    length = int(np.power(np.power(direction_v, 2).sum(), 0.5)) 

    mask[p2_int[0],p2_int[1],p2_int[2]] = 1 
    #print(length)
    
    for i in range(length):
        x_new = np.round(direction[0] * i + p1[0]).astype(np.int)
        y_new = np.round(direction[1] * i + p1[1]).astype(np.int)
        z_new = np.round(direction[2] * i + p1[2]).astype(np.int)
        # if length > 2:
        #     print(x_new,y_new,z_new)
        mask[x_new, y_new, z_new] = 1  # 200

    # debug
    # mask[int(p1[0]), int(p1[1]), int(p1[2])] = 2000
    # mask[int(p2[0]), int(p2[1]), int(p2[2])] = 2000
    # tifffile.imsave("test.tif", mask.astype(np.int16))
    return mask

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
        i = [[str_to_float(j) for j in i]]
        swc_num += i
    
    return swc_num

def str_to_float(s):
    """字符串转换为float"""
    if s is None:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def find_cross_end(swc_array):
    swc_c_e = []
    for i, l in enumerate(swc_array):
        if l[6] == -1:
            state = -1
        else:
            if i == len(swc_array)-1:
                state = -1
            else:
                if l[0] - l[6] == 1:
                    if l[0] == swc_array[i+1, 6]:
                        state = 0
                    else: state = -1
                else:
                    state = -2
        temp = l.tolist() + [state]
        swc_c_e.append(temp)
    return np.array(swc_c_e)

def sparse2dense_Loop(swc_array, shape, label):
    """ 将稀疏的gt_mask点, 连接成连续的线
        input: .swc 处理为list形式之后
        output: mask
    """
    mask = np.zeros(shape)
    c_e = find_cross_end(swc_array)[:, -1]
    for i, s in enumerate(c_e):
        if s == 0:
            line = connect_2_points_(p1=swc_array[i, 2:5][::-1], p2=swc_array[i+1, 2:5][::-1], size=shape)
        else:
            if s == -1:
                if swc_array[i, 6] == -1:
                    line = connect_2_points_(p1=swc_array[i, 2:5][::-1], p2=swc_array[i+1, 2:5][::-1], size=shape)
                else:
                    continue
            elif s == -2:
                line = connect_2_points_(p1=swc_array[int(swc_array[i, -1]), 2:5][::-1], p2=swc_array[i, 2:5][::-1], size=shape)
                mask[line>0] = label
                line = connect_2_points_(p1=swc_array[i, 2:5][::-1], p2=swc_array[i+1, 2:5][::-1], size=shape)
            else:
                print("The line {} have error. s: {}".format(i, s))
                break

        mask[line>0] = label
    mask = dilation(mask, ball(1))
    return mask


if __name__ == "__main__":
    # test()
#    m = Connect_2_points(p1=[173, 140, 10], p2=[173, 139, 9])
#    m = Connect_2_points(p1=[172, 136, 3], p2=[173, 138, 4])
    # m = Connect_2_points(p1=[0, 0, 3], p2=[0, 0, 8])
    # print(m.sum())
    # connect_2_points_()

    swc_path = r"C:\Users\Administrator\Desktop\Syn_chaos_300_5_2_4_6_000\swcs\Syn_chaos_300_5_2_4_6_000_4.swc"
    path, name = os.path.split(swc_path)
    label = str_to_float(name.split('.')[0].split('_')[-1])
    swc_num = np.array(psc(path, name.split('.')[0]))
    c_e = find_cross_end(swc_num)

    mask = sparse2dense_Loop(swc_num, shape=[300, 300, 300], label=label)
    
    tifffile.imsave(os.path.join(path, name.split('.')[0]+'.tif'), (mask>0).astype(np.float32))
    print('')
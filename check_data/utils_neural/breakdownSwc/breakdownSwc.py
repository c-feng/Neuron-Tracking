import numpy as np
import os

# /media/jjx/Biology/source_data/Modified_Selected_Dataset/swcs/

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

def str_to_float(s):
    """字符串转换为float"""
    if s is None:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def breakdownSingle(swc_num):
    # 将单个swc文件中的信号, 在分叉点位置拆散
    num = 0
    swcs = {}
    last = swc_num[0]

    swcs[num] = []
    swcs[num].append(last)
    for i, line in enumerate(swc_num[1:]):
        if last[-1] + 1 == line[-1]:
            swcs[num].append(line)
        else:
            if last[-1] >= line[-1]:
                num += 1
                swcs[num] = []
                swcs[num].append(line)
            else:
                swcs[num].append(line)
        last = line

    # 从1 开始标记
    for i, line in swcs.items():
        idx0 = line[0][0] - 1
        # idx_1 = line[-1]
        for j, l in enumerate(line):
            if j == 0:
                l[-1] = -1
            else:
                l[-1] = l[-1] - idx0
            l[0] = l[0] - idx0

    return swcs

def swcslist2txt(swcs_list, file_dir, name):
    for i, swc_list in enumerate(swcs_list, 1):
        swc_str_list = [str(l).strip('[').strip(']').replace(', ', ' ') for l in swcs_list[swc_list]]

        file_name = name.split('.')[0] + "_{:d}.swc".format(i)
        file = open(os.path.join(file_dir, file_name), 'w')
        for j, swc_str in enumerate(swc_str_list):
            file.write(swc_str + '\n')
        file.close()


# file_path = r"C:\Users\Administrator\Desktop\swcs\5200_34350_3900_083.swc"
# file_name = os.path.basename(file_path)
# root_dir = os.path.dirname(file_path)
# # root_dir = r"C:\Users\Administrator\Desktop\swcs"
# # name = "3450_31350_5150_011"
# swc_num = psc(root_dir, file_name.split(".")[0])
# swcs_ = breakdownSingle(swc_num)

root_dir = r"C:\Users\Administrator\Desktop\swcs"
swcs_name = os.listdir(root_dir)

output_dir = r"C:\Users\Administrator\Desktop\swcs_break"

for swc_name in swcs_name:
    swc_num = psc(root_dir, swc_name.split('.')[0])
    swcs = breakdownSingle(swc_num)

    swcslist2txt(swcs, output_dir, swc_name)



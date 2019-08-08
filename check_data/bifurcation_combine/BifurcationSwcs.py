import os
import json
import numpy as np

"""
    4450_27600_4650_002.swc [2, 3] 
    4450_27600_4650_003.swc [2, 4]
    4450_27600_4650_007.swc [7, 8]
    4450_27600_4650_009.swc [9, 10]
    4450_27600_4650_010.swc [9, 11]
    4450_27600_4650_013.swc [13, 14]
    4450_27600_4650_014.swc [13, 15]
    4450_27600_4650_020.swc [20, 21]
    4450_27600_4650_021.swc [20, 22]
    4450_27600_4650_022.swc [20, 23]
    4450_27600_4650_025.swc [24, 26]
    4450_27600_4650_028.swc [28, 29]
    4450_27600_4650_031.swc [31, 32]
    4450_27600_4650_032.swc [31, 33]
    4450_27600_4650_033.swc [31, 34]
    4450_27600_4650_034.swc [31, 35]
    4450_27600_4650_036.swc [36, 37]
"""

def find_name(name, l_names):
    """ find all the specify name in the names
        return the index of the position in the list
    """
    l = []
    for i, v in enumerate(l_names):
        if v == name:
            l.append(i)
    
    return l

def merge_lists_(label_lists):
    """ 将标签对融合在一起,
        存在一个list中
    """
    l_np = np.array(label_lists)
    unique_l = np.unique(l_np[:, 0])
    labels_all = []
    for l in unique_l:
        ind = np.where(l_np[:, 0] == l)[0]
        labels = [l, ]
        for i in ind:
            labels.append(l_np[i, 1])
        labels_all.append(labels)
    return labels_all

def merge_lists(label_lists):
    """ 将标签对融合在一起,
        存在一个list中
    """
    a = []
    for i in label_lists:
        a.append(i[0])
    unique_l = list(set(a))
    unique_l.sort()
    labels_all = []
    for l in unique_l:
        labels = [l, ]
        for label in label_lists:
            if label[0] == l:
                labels += label[1:]
        labels_all.append(labels)
    return labels_all

def find_removed(lines, names_list):
    removeds = []
    for i, name in enumerate(names_list):
        if name == '':
            k = i - 1
            for j in range(i-1, -1,-1):
                if names_list[j] != '':
                    k = j
                    break
            last_line = lines[k]
            remove_line = lines[i]
            id_ = '_'.join(last_line.split(' ')[0].split('.')[0].split('_')[:-1])
            removeds.append('_'.join([id_, remove_line.split(' ')[0]]))
    return removeds

file_path = "merge_log.txt"

output_file = open("output.txt", 'w+')

def main():
    with open(file_path) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    names_list = ['_'.join( line.split(' ')[0].split('.')[0].split('_')[:-1]) for line in lines]

    removed = find_removed(lines, names_list)

    names = list(set(names_list))

    for name in names:
        # debug
        # name = "22750_29500_600"
    #    name = "4450_27600_4650"
        # print(name)
        if name == '': continue
        l = find_name(name, names_list)
        a = []
        for i in l:
            a.append( json.loads(' '.join(lines[i].split(' ')[1:])) )
        merge_a = merge_lists(a)

        for i in merge_a:
            print(name, '', i, file=output_file)
        print('', file=output_file)

    print("removed:", file=output_file)
    for r in removed:
        print(r, file=output_file)

    output_file.close()

main()


# 6200_32350_3650  [3, 5, 6, 4]

import os
import re

def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if line]
    
    return lines[:-2]

metric = {"Dice": 0,
          "Prec": 1,
          "Recall": 2}
pattern_id = re.compile(r'[0-9]+[_][0-9]+[_][0-9]+[_][0-9]+')
pattern_v = re.compile(r'[0-9]+[.][0-9]+')


def filterbythreshold(file_path, threshold=0.8, m='Prec'):
    f = read_txt(file_path)
    
    ids = [pattern_id.findall(l) for i,l in enumerate(f) if i%2==0]
    ids = [id_[0] for id_ in ids]
    values = [pattern_v.findall(l) for i,l in enumerate(f) if i%2==1]
    values = [[float(i)  for i in v] for v in values]
    
    retain = []
    for v, i in zip(values, ids):
        if v[metric[m]] >= threshold*100:
            retain.append(i)
    return retain

if __name__ == "__main__":
    file_path = r"C:\Users\Administrator\Desktop\log.txt"
    # f = read_txt(file_path)

    a = filterbythreshold(file_path)

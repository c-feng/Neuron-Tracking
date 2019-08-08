import os
import glob
import numpy as np
from skimage.external import tifffile

dataset_dir = "/home/jjx/Biology/data_modified/"

tiff_paths = glob.glob(os.path.join(dataset_dir, 'ins_modified', "*.tif"))

def mean_std():

    max_v = 0
    min_v = 0
    mean = []
    std = []
    i = 0 
    tifs = np.zeros(shape=[301,301,301,1])
    for tiff_path in tiff_paths:
        # print(tiff_path)
        # if i > 1:
        #     break
        # i += 1
        tif = tifffile.imread(tiff_path)
        mean.append(np.mean(tif))
        std.append(np.std(tif))
        max_v = max(np.max(tif), max_v)
        min_v = min(np.min(tif), min_v)
        tif = tif[:, :, :, np.newaxis]
        print(os.path.split(tiff_path)[1],tif.shape)
        if tif.shape[0] != 301:
            print('----------')
            continue
        tifs = np.concatenate([tifs, tif], axis=3)
        # print(np.unique(tif))
        # print(mean, std, max_v, min_v)

    print(tifs.shape)
    print(np.mean(tifs), np.std(tifs))


num_pos = 0
num_neg = 0
for tiff_path in tiff_paths:
    tif = tifffile.imread(tiff_path)
    num_pos += np.sum(tif>0)
    num_neg += np.sum(tif==0)

print("num_pos:{}\nnum_neg:{}".format(num_pos, num_neg))
# num_pos:1466270
# num_neg:3453540226

import os
import sys
import time
import numpy as np
import glob
from skimage.external import tifffile
import matplotlib.pyplot as plt

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

path = r"H:\temp\test"
file_paths = glob.glob(os.path.join(path, "*.tif"))

for file_path in file_paths:
    root = os.path.dirname(file_path)
    name = os.path.basename(file_path).split('.')[0]

    tif = tifffile.imread(file_path)

    out_dir = os.path.join(root, name)
    mkdir_if_not_exist(out_dir)
    print("Processing {} ...".format(name))
    for i in np.unique(tif)[1:]:
        i = int(i)
        ins = (tif==i).astype(int)

        tifffile.imsave(os.path.join(out_dir, name+'_'+str(i)+'.tif'), ins.astype(np.float16))
        
        ins_z = (np.sum(ins,axis = 0)>0).astype(int)
        plt.imsave(os.path.join(out_dir, name+'_'+str(i)+'.jpg'), ins_z, cmap='gray')
        print("\r{}_{} have been saved.".format(name, i), end='')
        # sys.stdout.flush()
    print("\n")

# for i in range(5):
#     print("\rcount",i, end='')
#     # sys.stdout.write("\rcount:{}".format(i))
#     sys.stdout.flush()
#     time.sleep(1)
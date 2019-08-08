import os
import numpy as np
import glob
from skimage.external import tifffile

def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

out_path = r"C:\Users\Administrator\Desktop\dfs_data\dfs"
mkdir_if_not_exist(out_path)

paths = r"C:\Users\Administrator\Desktop\dfs_data\dfs"
tif_paths = glob.glob(os.path.join(paths, "*.tif"))
print(tif_paths)
for tif_path in tif_paths:
    name = tif_path.split('\\')[-1]
    print("Processing: {}".format(name))

    tif_df = tifffile.imread(tif_path)
    direct_vis = ~np.all(tif_df.transpose(1,2,3,0)==[0,0,0], axis=3)

    tifffile.imsave(os.path.join(out_path, name+"_direct_vis.tif"), (100*direct_vis).astype(np.float16))

    print("")
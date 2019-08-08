import os
import numpy as np
import re
from skimage.external import tifffile

def fromSeed2Data(seed_coord, full_data, patch_shape):
    """ seed为中心点
    """
    if isinstance(full_data, (np.ndarray, list, tuple)):
        shape = np.array(full_data).shape
    elif isinstance(full_data, torch.Tensor):
        shape = list(full_data.shape)
    patch = np.zeros(patch_shape)
    
    sel = [slice(max(s, 0), e+1) for s, e in zip(
                np.array(seed_coord)+1-np.array(patch_shape)//2-np.array(patch_shape)%2,
                np.array(seed_coord)+np.array(patch_shape)//2)]
    if len(sel) == 2:
        patch = full_data[..., sel[0], sel[1]].copy()
    elif len(sel) == 3:
        patch = full_data[..., sel[0], sel[1], sel[2]].copy()
    else:
        print("the data have shape of {}".format(shape))
    return patch

def write_file(datas, out_path, infos=None):
    num = len(datas)
    if infos == None:
        infos = np.arange(num)
    for data, info in zip(datas, infos):
        file_name = "_".join(info)
        tifffile.imsave(os.path.join(out_path, file_name+'.tif'), data.astype(np.float16))

PATCH_SHAPE = [72, 72, 72]
FOV_SHAPE = [41, 41, 41]

file_path = r"C:\Users\Administrator\Desktop\odd_error.txt"
with open(file_path) as f:
    lines = f.readlines()
lines = [line.rstrip('\n') for line in lines]

pattern = re.compile("loss")
end = pattern.search(lines[0]).span()[0]-1

tif_path = r"H:\dataset\Neural\data_modified\ins_modified"
out_path = r"C:\Users\Administrator\Desktop\odd_error"
for line in lines:

    a = eval(line[:end])
    for a_i in a:
        tif = tifffile.imread(os.path.join(tif_path, a_i[0]+'.tif'))

        whd = PATCH_SHAPE
        coords = a_i[1]
        patch =  tif[coords[0]:coords[0]+whd[0], coords[1]:coords[1]+whd[1],
                  coords[2]:coords[2]+whd[2]]
        fov = fromSeed2Data(a_i[2], patch, FOV_SHAPE)
    
        write_file([patch, fov], out_path=out_path, infos=[[a_i[0], "patch", str(coords[0]), str(coords[1]), str(coords[2])], 
                                                           [a_i[0], str(coords[0]), str(coords[1]), str(coords[2]), "fov",\
                                                                    str(a_i[2][0]), str(a_i[2][1]), str(a_i[2][2])]])


from skimage.external import tifffile
import os.path as osp
import os
import glob
import numpy as np
import argparse


#FILE_PATH = "C:/Users/cf__e/Desktop/21250_28250_350/centers/21250_28250_350_0.tif"
FILE_PATH = r"H:\skeleton_test_data\visual_results\5950_32100_3650_0\5950_32100_3650_0.tif"

def get_arguments():
    parser = argparse.ArgumentParser(description="Full Convolution Nerwork")
    parser.add_argument("-file-path", type=str, default='',
                        help="The path of *.tif file.")

    return parser.parse_args()

def main():
    args = get_arguments()
    
    if not osp.exists(args.file_path):
        tif_path = FILE_PATH
    else:
        tif_path = args.file_path

    
    path, name = osp.split(tif_path)
    
    
    tif = tifffile.imread(tif_path)
    
    tif_vis = (tif > 0) * 30
    
    #output_name = "check_1x14SD.tif"
    output_name = name.split('.')[0] + "_vis.tif"
    tifffile.imsave(osp.join(path, output_name), tif_vis.astype(np.int16))
    print("{} have been saved in {}".format(output_name, path))

def func(tif_path):
    
    path, name = osp.split(tif_path)
    
    
    tif = tifffile.imread(tif_path)
    
    tif_vis = (tif > 0) * 30
    
    #output_name = "check_1x14SD.tif"
    output_name = name.split('.')[0] + "_vis.tif"
    tifffile.imsave(osp.join(path, output_name), tif_vis.astype(np.int16))
    print("{} have been saved in {}".format(output_name, path))


if __name__ == "__main__":
    # main()

    root_path = r"H:\skeleton_test_data\visual_results"
    names = os.listdir(root_path)
    for name in names:
        func(os.path.join(root_path, name, name+".tif"))
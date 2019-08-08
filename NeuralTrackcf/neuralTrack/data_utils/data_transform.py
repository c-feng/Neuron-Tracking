import numpy as np
import os.path as osp 
import sys
import argparse
import lmdb

from glob import glob
from functools import partial
from skimage.external import tifffile

from ..utils.dataset import data_prepare, imgs_read
from ..utils.lmdb_utils import img_to_lmdb, lmdb_to_img
from ..utils.coo_utils import array_to_coo

working_dir = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Transfrom imgs to lmdb format")
parser.add_argument('--lmdb_dir', type=str, metavar='PATH')
parser.add_argument('--data_dir', type=str, metavar='PATH')

args = parser.parse_args()

env = lmdb.open(args.lmdb_dir, max_dbs=7, map_size=int(1e12))

imgs_data = env.open_db("imgs".encode())
imgs_size_data = env.open_db("imgs_size".encode())

segs_data = env.open_db("segs".encode())
ends_data = env.open_db("ends".encode())
ins_data = env.open_db("ins".encode())
juncs_data = env.open_db("juncs".encode())
centerlines_data = env.open_db("centerlines".encode())

imgs_info = data_prepare(args.data_dir)
with env.begin(write=True) as txn:
    for fnames in imgs_info:
        fname = osp.splitext(osp.basename(fnames[0]))[0]
        print(fname)
        
        img_fname = fnames[0]
        imgs = imgs_read(fnames)

        img_str, img_size_str = img_to_lmdb(imgs[0])
        txn.put(fname.encode(), img_str, db = imgs_data)
        txn.put(fname.encode(), img_size_str, db = imgs_size_data)

        seg_str, _ = array_to_coo(imgs[1])
        txn.put(fname.encode(), seg_str.encode(), db = segs_data)

        end_str, _ = array_to_coo(imgs[2])
        txn.put(fname.encode(), end_str.encode(), db = ends_data)

        ins_str, _ = array_to_coo(imgs[3])
        txn.put(fname.encode(), ins_str.encode(), db = ins_data)

        junc_str, _ = array_to_coo(imgs[4])
        txn.put(fname.encode(), junc_str.encode(), db = juncs_data)

        centerline_str, _ = array_to_coo(imgs[5])
        txn.put(fname.encode(), centerline_str.encode(), db = centerlines_data)

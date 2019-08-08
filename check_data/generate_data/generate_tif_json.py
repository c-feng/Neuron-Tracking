import json
import os.path as osp
import os
from generate_data import *
from skimage.external import tifffile
import numpy as np
from copy import deepcopy

colors = np.array([0,2000],np.uint16)


def write_json(obj,fpath):
    dir = osp.dirname(fpath)
    #osutils.mkdir_if_missing(dir)
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4)


def generate_tif(filetif_dir, tif_name, ins_num=3):

    ins_num = ins_num
    tif_shape = [300, 300, 300]
    tif = np.zeros(shape=tif_shape, dtype=np.int32)
    
    #debug
    #tif_vis = np.zeros(shape=tif_shape, dtype=np.int32)

    coords_info = dict()
    
    i_ins = 1
    for n in range(ins_num):
        gl_all = gene_n_degree_branch3D(num_degree=[1, 10])

        # 抖动
        for gl in gl_all:
            for i in range(gl.num_points):
                mu = 0
                s = 1
                deltax = 0.45 * random.gauss(mu=mu, sigma=s)
                deltay = 0.45 * random.gauss(mu=mu, sigma=s)
                deltaz = 0.45 * random.gauss(mu=mu, sigma=s)
                
                gl.x_values[i] = gl.x_values[i] + deltax
                gl.y_values[i] = gl.y_values[i] + deltay
                gl.z_values[i] = gl.z_values[i] + deltaz

        gl_all, data = pre_process(gl_all)

        tif_narr = np.zeros(shape=tif_shape, dtype=np.int32)
        tif_narr_vis = np.zeros(shape=tif_shape, dtype=np.int32)
        
        for i, d in zip(range(len(data)), data):
            for j in range(len(d)):
                tif_narr[tuple(d[j].astype(np.int))] = i+1
                tif_narr_vis[tuple(d[j].astype(np.int))] = 1
                
                coords_info[i_ins] = [int(i) for i in d[0]]#list(d[0].astype(np.int))
                tif[tuple(d[j].astype(np.int))] = i_ins
                #debug
                #tif_vis[tuple(d[j].astype(np.int32))] = 1
            i_ins += 1

    tifffile.imsave(osp.join(filetif_dir, tif_name+".tif"), tif.astype(np.int16))
    #debug
    #tifffile.imsave(osp.join(filetif_dir, tif_name+"_vis.tif"), colors[tif_vis])
    print("{} have been saved.\n".format(tif_name+".tif"))

    json_path = osp.join(filetif_dir, tif_name+".json")
    write_json(coords_info, json_path)

def generate_tif_v1(filetif_dir, tif_name, ins_num=3):

    ins_num = ins_num
    tif_shape = [300, 300, 300]
    tif = np.zeros(shape=tif_shape, dtype=np.int32)
    
    #debug
    tif_vis = np.zeros(shape=tif_shape, dtype=np.int32)

    coords_info = dict()
    
    i_ins = 1
    for n in range(ins_num):
        degree_2th_num = 3  #int(random.gauss(10, 3))
        gl_all = gene_n_degree_branch3D(num_degree=[1, degree_2th_num])

        # 抖动
        for gl in gl_all:
            for i in range(gl.num_points):
                mu = 0
                s = 1
                deltax = 0.9 * random.gauss(mu=mu, sigma=s)
                deltay = 0.9 * random.gauss(mu=mu, sigma=s)
                deltaz = 0.9 * random.gauss(mu=mu, sigma=s)
                
                gl.x_values[i] = gl.x_values[i] + deltax
                gl.y_values[i] = gl.y_values[i] + deltay
                gl.z_values[i] = gl.z_values[i] + deltaz

        # 毛刺
        size_num = int(random.gauss(100, 10))
        ind = list(np.random.choice(gl_all[0].num_points, size=size_num, replace=False))
        init_coords = [np.array(gl_all[0].x_values)[ind], np.array(gl_all[0].y_values)[ind], np.array(gl_all[0].z_values)[ind]]
        init_thetas = np.array(gl_all[0].theta_)[ind]
        init_phis = np.array(gl_all[0].phi_)[ind]
        for k in range(size_num):
            ini_coord = [init_coords[0][k], init_coords[1][k], init_coords[2][k]]
            gl_i = GenerateLine_2Direct_3D(init_coords=ini_coord,
                                    init_theta=init_thetas[k]+random.uniform(0, PI/2),
                                    init_phi=init_phis[k]+random.uniform(0, PI/2),
                                    num_points=int(random.gauss(mu=30, sigma=3)))
            gl_i.fill_points()
            gl_all[0].x_values += gl_i.x_values
            gl_all[0].y_values += gl_i.y_values
            gl_all[0].z_values += gl_i.z_values
            gl_all[0].phi_ += gl_i.phi_
            gl_all[0].theta_ += gl_i.theta_


        gl_all, data = pre_process(gl_all)

        tif_narr = np.zeros(shape=tif_shape, dtype=np.int32)
        tif_narr_vis = np.zeros(shape=tif_shape, dtype=np.int32)
        
        for i, d in zip(range(len(data)), data):
            for j in range(len(d)):
                tif_narr[tuple(d[j].astype(np.int))] = i+1
                tif_narr_vis[tuple(d[j].astype(np.int))] = 1
                
                coords_info[i_ins] = [int(i) for i in d[0]]#list(d[0].astype(np.int))
                tif[tuple(d[j].astype(np.int))] = i_ins
                #debug
                tif_vis[tuple(d[j].astype(np.int32))] = 1
            i_ins += 1

    tifffile.imsave(osp.join(filetif_dir, tif_name+".tif"), tif.astype(np.int16))
    #debug
    tifffile.imsave(osp.join(filetif_dir, tif_name+"_vis.tif"), colors[tif_vis])
    print("{} have been saved.\n".format(tif_name+".tif"))

    json_path = osp.join(filetif_dir, tif_name+".json")
    write_json(coords_info, json_path)


def generate_tif_v2(filetif_dir, tif_name, ins_num=3):
    # 添加背景噪声

    ins_num = ins_num
    tif_shape = [300, 300, 300]
    tif = np.zeros(shape=tif_shape, dtype=np.int32)
    
    #debug
    tif_vis = np.zeros(shape=tif_shape, dtype=np.int32)

    coords_info = dict()
    
    i_ins = 1
    for n in range(ins_num):
        degree_2th_num = 3  #int(random.gauss(10, 3))
        gl_all = gene_n_degree_branch3D(num_degree=[1, degree_2th_num])

        # 抖动
        for gl in gl_all:
            for i in range(gl.num_points):
                mu = 0
                s = 1
                deltax = 0.9 * random.gauss(mu=mu, sigma=s)
                deltay = 0.9 * random.gauss(mu=mu, sigma=s)
                deltaz = 0.9 * random.gauss(mu=mu, sigma=s)
                
                gl.x_values[i] = gl.x_values[i] + deltax
                gl.y_values[i] = gl.y_values[i] + deltay
                gl.z_values[i] = gl.z_values[i] + deltaz

        # 毛刺
        size_num = int(random.gauss(100, 10))
        ind = list(np.random.choice(gl_all[0].num_points, size=size_num, replace=False))
        init_coords = [np.array(gl_all[0].x_values)[ind], np.array(gl_all[0].y_values)[ind], np.array(gl_all[0].z_values)[ind]]
        init_thetas = np.array(gl_all[0].theta_)[ind]
        init_phis = np.array(gl_all[0].phi_)[ind]
        for k in range(size_num):
            ini_coord = [init_coords[0][k], init_coords[1][k], init_coords[2][k]]
            gl_i = GenerateLine_2Direct_3D(init_coords=ini_coord,
                                    init_theta=init_thetas[k]+random.uniform(0, PI/2),
                                    init_phi=init_phis[k]+random.uniform(0, PI/2),
                                    num_points=int(random.gauss(mu=30, sigma=3)))
            gl_i.fill_points()
            gl_all[0].x_values += gl_i.x_values
            gl_all[0].y_values += gl_i.y_values
            gl_all[0].z_values += gl_i.z_values
            gl_all[0].phi_ += gl_i.phi_
            gl_all[0].theta_ += gl_i.theta_


        gl_all, data = pre_process(gl_all)

        tif_narr = np.zeros(shape=tif_shape, dtype=np.int32)
        tif_narr_vis = np.zeros(shape=tif_shape, dtype=np.int32)
        
        for i, d in zip(range(len(data)), data):
            for j in range(len(d)):
                tif_narr[tuple(d[j].astype(np.int))] = i+1
                tif_narr_vis[tuple(d[j].astype(np.int))] = 1
                
                coords_info[i_ins] = [int(i) for i in d[0]]#list(d[0].astype(np.int))
                tif[tuple(d[j].astype(np.int))] = i_ins
                #debug
                tif_vis[tuple(d[j].astype(np.int32))] = 1
            i_ins += 1

    # 背景噪声
    noise_mask = np.zeros(shape=tif_shape, dtype=np.int32)
    index = np.random.choice(tif_shape[0]*tif_shape[1]*tif_shape[2], size=int(20*random.gauss(0, 1)+180), replace=False)
    coords = np.array(np.unravel_index(index, dims=tif_shape))
    noise_mask[coords[0], coords[1], coords[2]] = 1
    tif_index = tif > 0
    tif_ = deepcopy(tif)
    tif[noise_mask>0] = 1000
    tif[tif_index] = tif_[tif_index]
    tif_vis[noise_mask>0] = 1




    tifffile.imsave(osp.join(filetif_dir, tif_name+".tif"), tif.astype(np.int16))
    #debug
    tifffile.imsave(osp.join(filetif_dir, tif_name+"_vis.tif"), colors[tif_vis])
    print("{} have been saved.\n".format(tif_name+".tif"))

    json_path = osp.join(filetif_dir, tif_name+".json")
    write_json(coords_info, json_path)

def pre_process(gl=None):
    
    gl_all = gl
    data3D = []
    
    x_max_v = -1000; x_min_v = 10000
    y_max_v = -1000; y_min_v = 10000
    z_max_v = -1000; z_min_v = 10000
    
    for gl in gl_all:
        x_max_v = max(x_max_v, max(gl.x_values))
        x_min_v = min(x_min_v, min(gl.x_values))
        y_max_v = max(y_max_v, max(gl.y_values))
        y_min_v = min(y_min_v, min(gl.y_values))
        z_max_v = max(z_max_v, max(gl.z_values))
        z_min_v = min(z_min_v, min(gl.z_values))

    # 缩放到(300, 300)
    #x_max = max(gl.x_values); x_min = min(gl.x_values)
    
    for gl in gl_all:

        gl.x_values = ((np.array(gl.x_values) - x_min_v) / (x_max_v - x_min_v)) * 299
        
        #y_max = max(gl.y_values); y_min = min(gl.y_values)
        gl.y_values = ((np.array(gl.y_values) - y_min_v) / (y_max_v - y_min_v)) * 299
    
        gl.z_values = ((np.array(gl.z_values) - z_min_v) / (z_max_v - z_min_v)) * 299
        
        #print(gl.z_values.shape)
        coords = np.stack([gl.x_values, gl.y_values, gl.z_values], axis=0).T
        
        data3D.append(coords)
    
    return gl_all, data3D



def main():
    tif_num = 5
    fpath = "./SyntheticData/1x3glitch"

    if not osp.exists(fpath):
        os.mkdir(fpath)

    for i in range(tif_num):
        tif_name = "SD_{:0>3d}".format(i)
        generate_tif_v2(filetif_dir=fpath, tif_name=tif_name, ins_num=1)



if __name__ == "__main__":
    main()
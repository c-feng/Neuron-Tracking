import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.external import tifffile
import os.path as osp
import time
import random

from generate_data import *

colors = np.array([0,2000],np.uint16)

def vis_2Ddata():
    while True:
        
        #gl_all = generate_branch2D(num_branch=3)
        gl_all = gene_n_degree_branch2D(num_degree=[1, 3])

        # 抖动
        for gl in gl_all:
            for i in range(gl.num_points):
                deltax = random.gauss(mu=0, sigma=1.5)
                deltay = random.gauss(mu=0, sigma=1.5)
                # deltaz = random.gauss(mu=0, sigma=random.uniform(PI/18, PI/3))
                
                gl.x_values[i] = gl.x_values[i] + deltax
                gl.y_values[i] = gl.y_values[i] + deltay




        x_max_v = -1000; x_min_v = 10000
        y_max_v = -1000; y_min_v = 10000
        for gl in gl_all:
            x_max_v = max(x_max_v, max(gl.x_values))
            x_min_v = min(x_min_v, min(gl.x_values))
            y_max_v = max(y_max_v, max(gl.y_values))
            y_min_v = min(y_min_v, min(gl.y_values))


        # Set the size of the plotting window.
        plt.figure(dpi=128, figsize=(3, 3))
        for gl in gl_all:
            # 缩放到(300, 300)
            #x_max = max(gl.x_values); x_min = min(gl.x_values)
            gl.x_values = ((np.array(gl.x_values) - x_min_v) / (x_max_v - x_min_v)) * 299
            
            #y_max = max(gl.y_values); y_min = min(gl.y_values)
            gl.y_values = ((np.array(gl.y_values) - y_min_v) / (y_max_v - y_min_v)) * 299

            
            #point_numbers = list(range(gl.num_points))
            #plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
            #    edgecolor='none', s=1)
            #plt.plot(gl.x_values, gl.y_values) 
            plt.scatter(gl.x_values, gl.y_values, s=12)

            # Emphasize the first and last points.
            plt.scatter(gl.x_values[0], gl.y_values[0], marker='x', c='black', edgecolors='none', s=10)
            plt.scatter(gl.x_values[-1], gl.y_values[-1], marker='x', c='red', edgecolors='none', s=10)
            
        # Remove the axes.
        plt.axes().get_xaxis().set_visible(True)
        plt.axes().get_yaxis().set_visible(True)
            
        plt.show()
        
        keep_running = input("Make another data? (y/n): ")
        if keep_running == 'n':
            break


def vis_3Ddata():
    
    while True:
        
        #gl_all = generate_branch3D(num_branch=3)
        gl_all = gene_n_degree_branch3D(num_degree=[1, 3])
        
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

        # Set the size of the plotting window.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for gl in gl_all:
            # 缩放到(300, 300)
            #x_max = max(gl.x_values); x_min = min(gl.x_values)
            gl.x_values = ((np.array(gl.x_values) - x_min_v) / (x_max_v - x_min_v)) * 299
            
            #y_max = max(gl.y_values); y_min = min(gl.y_values)
            gl.y_values = ((np.array(gl.y_values) - y_min_v) / (y_max_v - y_min_v)) * 299

            gl.z_values = ((np.array(gl.z_values) - z_min_v) / (z_max_v - z_min_v)) * 299

            point_numbers = list(range(gl.num_points))
            #plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
            #    edgecolor='none', s=1)
            #ax.plot(gl.x_values, gl.y_values, gl.z_values) 
            ax.scatter(gl.x_values, gl.y_values, gl.z_values, s=10)

            # Emphasize the first and last points.
            ax.scatter(gl.x_values[0], gl.y_values[0], gl.z_values[0], marker='x', c='black', edgecolors='none', s=20)
            ax.scatter(gl.x_values[-1], gl.y_values[-1], gl.z_values[-1], marker='x', c='red', edgecolors='none', s=20)
            
        # Remove the axes.
#        plt.axes().get_xaxis().set_visible(True)
#        plt.axes().get_yaxis().set_visible(True)
#        plt.axes().get_yaxis().set_visible(True)

        #plt.savefig("3Ddata.tif")  
        plt.show()
        
        keep_running = input("Make another data? (y/n): ")
        if keep_running == 'n':
            break


def vis_carton():
    
    while True:

        gl_all = gene_n_degree_branch2D(num_degree=[1, 3])
        
        x_max_v = -1000; x_min_v = 10000
        y_max_v = -1000; y_min_v = 10000
        for gl in gl_all:
            x_max_v = max(x_max_v, max(gl.x_values))
            x_min_v = min(x_min_v, min(gl.x_values))
            y_max_v = max(y_max_v, max(gl.y_values))
            y_min_v = min(y_min_v, min(gl.y_values))

        # Set the size of the plotting window.
        plt.figure(dpi=144, figsize=(3, 3))
        plt.ion()
        plt.scatter(gl_all[0].x_values[0], gl_all[0].y_values[0], marker='x', c='w', edgecolors='none', s=20)

        plt.pause(10)
        colo = ['b', 'g', 'purple', 'orange']
        for gl, c in zip(gl_all, colo):
            # 缩放到(300, 300)
            #x_max = max(gl.x_values); x_min = min(gl.x_values)
            gl.x_values = ((np.array(gl.x_values) - x_min_v) / (x_max_v - x_min_v)) * 299
            
            #y_max = max(gl.y_values); y_min = min(gl.y_values)
            gl.y_values = ((np.array(gl.y_values) - y_min_v) / (y_max_v - y_min_v)) * 299

            
            #point_numbers = list(range(gl.num_points))
            #plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
            #    edgecolor='none', s=1)
            #plt.plot(gl.x_values, gl.y_values)
            plt.scatter(gl.x_values[0], gl.y_values[0], marker='x', c='black', edgecolors='none', s=20)
            time.sleep(0.05)
            for i in range(gl.num_points):
                plt.scatter(gl.x_values[i], gl.y_values[i], c=c, s=12)
                #time.sleep(0.1)
                plt.pause(0.02)

            # Emphasize the first and last points.
            #plt.scatter(gl.x_values[0], gl.y_values[0], marker='x', c='black', edgecolors='none', s=10)
            plt.scatter(gl.x_values[-1], gl.y_values[-1], marker='x', c='red', edgecolors='none', s=20)
            
        # Remove the axes.
        # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_yaxis().set_visible(False)
        
        plt.ioff()
        plt.show()
        
        keep_running = input("Make another data? (y/n): ")
        if keep_running == 'n':
            break



if __name__ == "__main__":
     vis_3Ddata()
    
    
    
    
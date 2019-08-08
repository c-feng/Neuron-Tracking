import numpy as np
import torch
from skimage.morphology import ball,dilation

class Canvas(object):
    def __init__(self,canvas_shape,fov_shape = [24,24,24]):
        self.canvas = np.zeros(canvas_shape)
        self.canvas_label = np.zeros(canvas_shape,dtype = np.uint16) 
        self.canvas_label_mask = np.zeros(canvas_shape,dtype = np.bool)

        self.mask_modified = np.zeros(canvas_shape,dtype = np.bool)
        self.canvas_shape = canvas_shape
        self.fov_shape = fov_shape 
    def update(self,fov,fov_position):
        fov_l = np.array(fov_position) - np.array(self.fov_shape)//2
        fov_r = np.array(fov_position) - np.array(self.fov_shape)//2 + np.array(self.fov_shape)
        
        canvas_fov = self.canvas[fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]]
        
        fov_mask_modified  = self.mask_modified[fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]]
        
        fov_mask_not_modified = np.logical_not(fov_mask_modified)

        canvas_fov[fov_mask_not_modified] = fov[fov_mask_not_modified]

        #fov_mask_gt_thres = np.logical_and(canvas_fov > 0.5)
        #fov_mask_lt_thres = np.logical_and(fov < canvas_fov,canvas_fov < 0.5)
        #fov_mask_thres = np.logical_or(fov_mask_gt_thres,fov_mask_lt_thres)
        #fov_mask_thres = np.logical_and(fov_mask_modified,fov_mask_thres)
        fov_mask_lt_thres = np.logical_and(fov < canvas_fov,canvas_fov < 0.5)
        fov_mask_thres = np.logical_or(fov_mask_lt_thres,canvas_fov > 0.5)
        fov_mask_thres = np.logical_and(fov_mask_modified,fov_mask_thres)
        
        canvas_fov[fov_mask_thres] = fov[fov_mask_thres]
        
        fov_mask_modified = np.logical_or(fov_mask_not_modified,fov_mask_thres)
        self.mask_modified[fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]] = fov_mask_modified

        self.canvas[fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]] = canvas_fov
    def label(self,label = 1,thres = 0.5):
        mask = self.canvas > 0.5
        self.canvas_label[mask] = label
        self.canvas_label_mask = self.canvas_label >0

        self.canvas = np.zeros(self.canvas_shape)
        self.mask_modified = np.zeros(self.canvas_shape,dtype = np.bool)
    def crop(self,fov_coord):
        fov_coord_l = np.array(fov_coord) - np.array(self.fov_shape)//2
        fov_coord_r = np.array(fov_coord) + np.array(self.fov_shape) - np.array(self.fov_shape)//2
        return self.canvas[fov_coord_l[0]:fov_coord_r[0],\
                fov_coord_l[1]:fov_coord_r[1],\
                fov_coord_l[2]:fov_coord_r[2]]

class CanvasBatch(object):
    def __init__(self,canvas_shape,fov_shape = [24,24,24]):
        self.canvas = np.zeros(canvas_shape)
        self.canvas_label = np.zeros(canvas_shape,dtype = np.uint16) 
        self.canvas_label_mask = np.zeros(canvas_shape,dtype = np.bool)

        self.mask_modified = np.zeros(canvas_shape,dtype = np.bool)
        self.canvas_shape = canvas_shape
        self.fov_shape = fov_shape 
    def update(self,fov,fov_position):
        fov_l = np.array(fov_position) - np.array(self.fov_shape)//2
        fov_r = np.array(fov_position) - np.array(self.fov_shape)//2 + np.array(self.fov_shape)
        #print(fov_position)
        #print(fov_l,fov_r) 

        canvas_fov = self.canvas[:,fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]]
        
        fov_mask_modified  = self.mask_modified[:,fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]]
        
        fov_mask_not_modified = np.logical_not(fov_mask_modified)
        
        canvas_fov[fov_mask_not_modified] = fov[fov_mask_not_modified]

        #fov_mask_gt_thres = np.logical_and(fov > canvas_fov,canvas_fov > 0.5)
        fov_mask_lt_thres = np.logical_and(fov < canvas_fov,canvas_fov < 0.5)
        fov_mask_thres = np.logical_or(fov_mask_lt_thres,canvas_fov > 0.5)
        fov_mask_thres = np.logical_and(fov_mask_modified,fov_mask_thres)

        canvas_fov[fov_mask_thres] = fov[fov_mask_thres]
        
        fov_mask_modified = np.logical_or(fov_mask_not_modified,fov_mask_thres)
        self.mask_modified[:,fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]] = fov_mask_modified

        self.canvas[:,fov_l[0]:fov_r[0],fov_l[1]:fov_r[1],fov_l[2]:fov_r[2]] = canvas_fov
    def label(self,label = 1,thres = 0.5):
        mask = self.canvas > 0.5
        self.canvas_label[mask] = label
        self.canvas_label_mask = self.canvas_label >0

        self.canvas = np.zeros(self.canvas_shape)
        self.mask_modified = np.zeros(self.canvas_shape,dtype = np.bool)
    def crop(self,fov_coord):
        fov_coord_l = np.array(fov_coord) - np.array(self.fov_shape)//2
        fov_coord_r = np.array(fov_coord) + np.array(self.fov_shape) - np.array(self.fov_shape)//2
        return self.canvas[:,fov_coord_l[0]:fov_coord_r[0],\
                fov_coord_l[1]:fov_coord_r[1],\
                fov_coord_l[2]:fov_coord_r[2]]


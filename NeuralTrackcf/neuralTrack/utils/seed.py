import numpy as np
import queue
import numbers
from skimage.morphology import ball,dilation

class SeedPolicy(object):
    def __init__(self,mask_shape,fov_shape,deltas):
        if isinstance(fov_shape,numbers.Number):
            fov_shape = [fov_shape,fov_shape,fov_shape]
        self.mask_shape = mask_shape
        self.fov_shape = fov_shape
        self.deltas = deltas 

        self.seeds_list = queue.Queue()
        self.seeds_viewed_list = []
        self.seed_range_mask = np.zeros(mask_shape)
        fov_shape_l = np.array(fov_shape)//2
        fov_shape_r = np.array(fov_shape) - fov_shape_l
        self.seed_range_mask[fov_shape_l[0]:-fov_shape_r[0],\
                    fov_shape_l[1]:-fov_shape_r[1],\
                    fov_shape_l[2]:-fov_shape_r[2]] = 1
    def coords_trans(self,ind):
        return np.unravel_index(ind,self.mask_shape)

        

    def seeds_viewed_coord_add(self,coords_viewed):
        print("{} {} {} now is added to the viewed list".format(*coords_viewed))
        seeds_viewed = np.ravel_multi_index(coords_viewed,self.mask_shape)
        if isinstance(seeds_viewed,np.ndarray):
            seeds_viewed = list(seeds_viewed)
        elif isinstance(seeds_viewed,numbers.Number):
            seeds_viewed = [seeds_viewed]
        else:
            print("cannot recognize the coord to be added")
            sys.exit(0)

        seeds_viewed_list = self.seeds_viewed_list 
        seeds_viewed_list.extend(seeds_viewed)
        seeds_viewed_list = np.unique(seeds_viewed_list)
        seeds_viewed_list = list(seeds_viewed_list) 
        self.seeds_viewed_list = seeds_viewed_list

    def seeds_viewed_add(self,seeds_viewed):
        seeds_viewed_list = self.seeds_viewed_list 
        seeds_viewed_list.extend(seeds_viewed)
        seeds_viewed_list = np.unique(seeds_viewed_list)
        seeds_viewed_list = list(seeds_viewed_list) 
        self.seeds_viewed_list = seeds_viewed_list

    def seeds_list_coord_add(self,coords):
        seeds = np.ravel_multi_index(coords,self.mask_shape)
        if isinstance(seeds,np.ndarray):
            for seed in seeds:
                self.seeds_list.put(seed)
        elif isinstance(seeds,numbers.Number):
            self.seeds_list.put(seeds)
        else:
            print("cannot recognize the coord to be added")
            sys.exit(0)
    def seeds_list_add(self,seeds):
        if isinstance(seeds,np.ndarray):
            #print(seeds)
            for seed in seeds:
                self.seeds_list.put(seed)
        elif isinstance(seeds,numbers.Number):
            #print(seeds)
            self.seeds_list.put(seeds)
        else:
            print("cannot recognize the ind to be added")
            sys.exit(0)
            
    def seeds_find(self,position_fov,mask,deltas = [8,8,8],t = 0.85):

        fov_shape = mask.shape
        center_coord = np.array(fov_shape)//2
        coords_increment = np.array(position_fov) - center_coord 

        coords_l = center_coord - np.array(deltas)
        coords_r = center_coord + np.array(deltas)

        cubic_mask = mask[coords_l[0]:coords_r[0],\
                coords_l[1]:coords_r[1],\
                coords_l[2]:coords_r[2]]
        planes = [cubic_mask[0],cubic_mask[-1],\
            cubic_mask[:,0],cubic_mask[:,-1],\
            cubic_mask[:,:,0],cubic_mask[:,:,-1]]
        seeds_coord = []
        
        for i,plane in enumerate(planes):
            axis = i//2
            #print(plane)
            #print(i,np.max(plane))
            if np.max(plane) > t:
                #print(i,np.max(plane))
                ind = np.argmax(plane)
                coord = list(np.unravel_index(ind,plane.shape))
                if i%2 == 0:
                    coord.insert(axis,coords_l[axis])
                else:
                    coord.insert(axis,coords_r[axis])
                #print(i,coord)
                coord = np.array(coord) + coords_increment 
                if self.seed_range_mask[coord[0],coord[1],coord[2]]>0:
                    seeds_coord.append(coord)
        #print(seeds_coord)
        if seeds_coord:
            #print(seeds_coord)
            seeds_coord = np.array(seeds_coord)
            #seeds_coord += coords_increment
            self.seeds_list_coord_add(np.array(seeds_coord).transpose())

    def seeds_viewed_find(self,position_fov,mask,deltas = [8,8,8]):
        fov_shape = mask.shape
        center_coord = np.array(fov_shape)//2
        coords_increment = np.array(position_fov) - center_coord 

        fov_mask = dilation(mask > 0.5,ball(1)) 
        inds = np.arange(fov_mask.size)[fov_mask.flatten()>0]
        seeds_viewed_coord = np.unravel_index(inds,fov_shape)
        seeds_viewed_coord = np.array(seeds_viewed_coord)
        seeds_viewed_coord += coords_increment
        self.seeds_viewed_coord_add(np.array(seeds_viewed_coord))

    def __iter__(self):
        return self
    
    def __next__(self):
        while not self.seeds_list.empty():
            seed = self.seeds_list.get()
            if self.is_seed_valid(seed):
                return self.coords_trans(seed)
            else:
                print("{} has been visited,drop it".format(seed))
        raise StopIteration()
    
    def next(self):
        return self.__next__()

    def is_seed_valid(self,ind):
        print("now is check the {}".format(ind))
        if not self.seeds_viewed_list:
            return True
        #print(seed)
        seeds_viewed_coord = np.array(np.unravel_index(self.seeds_viewed_list,\
                self.mask_shape)).transpose()
        seeds_viewed_coord_reduced = np.array(seeds_viewed_coord)//np.array(self.deltas)
        #seeds_viewed_reduced_list = np.ravel_multi_index(seeds_viewed_coord_reduced,\
        #        np.array(self.mask_shape)//np.array(deltas))
        #seeds_viewed_reduced_list = list(np.unique(seeds_viewed_reduced_list))
        seeds_viewed_coord_reduced_list = seeds_viewed_coord_reduced.tolist()

        ind_coord_reduced = np.unravel_index(ind,self.mask_shape)//np.array(self.deltas)
        ind_coord_reduced = np.unique(ind_coord_reduced).tolist()

        #seed_reduced = list(np.array(seed)//np.array(self.deltas))
        if ind_coord_reduced in seeds_viewed_coord_reduced_list:
            print(ind_coord_reduced)
            print(seeds_viewed_coord_reduced_list)
            return False
        else:
            print(ind_coord_reduced,len(seeds_viewed_coord_reduced_list),self.seeds_list.qsize())
            return True 

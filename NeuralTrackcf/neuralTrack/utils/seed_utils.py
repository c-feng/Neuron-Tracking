import numpy as np
from skimage.morphology import dilation,ball,cube

from .mask_utils import ball_mask_generate,cube_mask_generate

def random_seeds_generate(centerline, num_seeds = 20):
    seeds = np.zeros_like(centerline)
    inds = np.arange(seeds.size)[(centerline > 0).flatten()]
    inds_sel = np.random.choice(inds, num_seeds, replace = False)
    xs,ys,zs = np.unravel_index(inds_sel,seeds.shape)
    #seeds[xs,ys,zs] = 1
    
    return inds_sel

def selective_seeds_generate(centerline, junc, num_seeds = 20):
    junc_mask = dilation(junc,cube(8))
    p = centerline
    mask = np.logical_and(junc,centerline)
    p[mask] = 20
    p = p.astype(float)/np.sum(p)

    seeds = np.zeros_like(centerline)
    inds = np.arange(seeds.size)[(centerline > 0).flatten()]
    inds_sel = np.random.choice(inds, num_seeds, replace = False, p.flatten())
    #xs,ys,zs = np.unravel_index(inds_sel,seeds.shape)
    #seeds[xs,ys,zs] = 1
    #return seeds
    return inds_sel

def balance_seeds_generate(center, ins, radius = 8, num_seeds = 40):
    inds = np.arange(ins.size)
    seed_mask = np.zeros_like(ins, dtype = np.bool)
    seed_mask[center[0], center[1], center[2]]

    label_c = ins[center[0], center[1], center[2]]
    
    mask_p = np.logical_and(seed_mask, ins == label_c).flatten()
    inds_p = inds[mask_p]

    mask_b = np.logical_and(seed_mask, ins == 0).flatten()
    inds_b = inds[mask_b]

    mask_n = np.logical_xor(mask_b == False, mask_p)
    inds_n = inds[mask_n]

    num_seeds_p = num_seeds //2
    num_seeds_n = num_seeds -  num_seeds_p 

    if len(inds_p) > num_seeds_p:
        inds_p_sel = np.random.choice(inds_p, num_seeds_p, replace = False)
    else:
        inds_p_sel = np.random.choice(inds_p, num_seeds_p)

    if len(inds_n) > num_seeds_p:
        inds_n_sel = np.random.choice(inds_n, num_seeds_n, replace = False)
    elif len(inds_n) == 0:
        inds_n_sel = np.random.choice(inds_b, num_seeds_n, replace = False)
    else:
        inds_b_sel = np.random.choice(inds_n, num_seeds_n)
    
    inds_s = inds_n_sel.tolist() + inds_p_sel.tolist()

    return inds_s

def batch_random_seeds_generate(batch_centerline, num_seeds = 100):
    seeds_list = []
    for centerline in batch_centerline:
        seeds = random_seeds_generate(centerline, num_seeds)
        seeds_list.append(seeds)
    #seeds_list = np.concatenate(seeds_list, axis = 0)
    seeds_list = np.array(seeds_list)
    return seeds_list

def batch_selective_seeds_generate(batch_centerline, batch_junc, num_seeds = 100):
    seeds_list = []
    for centerline, junc in zip(batch_centerline, batch_junc):
        seeds = selective_seeds_generate(centerline, junc, num_seeds)
        seeds_list.append(seeds)
    #seeds_list = np.concatenate(seeds_list, axis = 0)
    seeds_list = np.array(seeds_list)
    return seeds_list

import kimimaro
import numpy as np
from skimage.external import tifffile

labels = tifffile.imread(\
        "../ins_pred.tif")

skels = kimimaro.skeletonize(
  labels, 
  teasar_params={
    'scale': 4,
    'const': 500, # physical units
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    'soma_detection_threshold': 1100, # physical units
    'soma_acceptance_threshold': 3500, # physical units
    'soma_invalidation_scale': 1.0,
    'soma_invalidation_const': 300, # physical units
    'max_paths': 15, # default None
  },
  dust_threshold=50,
  anisotropy=(200,200,1000), # default True
  fix_branching=True, # default True
  fix_borders=True, # default True
  progress=True, # default False
  parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
)
print(np.unique(labels))
print(skels.keys())
skel = skels[115]
#print(skel.edges)
#print(skel.vertices)

# skel_array = np.zeros_like(labels)
# coords_xyz = (skel.vertices/ np.array([200,200,1000])).astype(int)
# skel_array[coords_xyz[:,0], coords_xyz[:,1],coords_xyz[:,2]] = 1
# tifffile.imsave("test.tif", skel_array)



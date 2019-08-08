from skimage.external import tifffile
import numpy as np
from scipy.spatial.distance import cdist
import math
from numpy import linalg as la

FILE_PATH = r"C:\Users\Administrator\Desktop\check_data\sample_points\ins_modified\3700_32600_4400.tif"

def sample_singleins(ins_tif):
    pass


def find_endpoints_(ins_tif):
    ind = np.where(ins_tif!=0)[0]
    idx = np.where(single_ins!=0)
    coords = np.stack(idx, axis=0).transpose()

    endpoints = []
    for i, c in enumerate(coords):
        dist = cdist(XA=np.expand_dims(c, axis=0), XB=coords, metric='euclidean')
        idx_dless5 = np.where(dist<5)[1]
        dless5 = coords[idx_dless5[idx_dless5!=i], :]
        s = ((dless5 - c) > 0).tolist()
        for j in range(len(s)):
            if s[j]!=s[0]:
                break
            if j == len(s)-1:
                endpoints.append(c.tolist())
    endpoints = np.array(endpoints)
    return endpoints

def find_endpoints(ins_tif):
    idx = np.where(ins_tif!=0)
    coords = np.stack(idx, axis=0).transpose()
    
    endpoints = []
    for i, c in enumerate(coords):
        dist = cdist(XA=np.expand_dims(c, axis=0), XB=coords, metric='euclidean')
        idx_dless5 = np.where(dist<5)[1]
        dless5 = coords[idx_dless5[idx_dless5!=i], :]
        #gap_vectors.append((dless5 - c).tolist())
        gap_vector = dless5 - c
        theta_phi = [cal_theta_phi(i) for i in gap_vector]
    
        for j in range(len(theta_phi)):
            if theta_phi[j][1] > 100 or theta_phi[j][1] < -100:
                break
            if j == len(theta_phi)-1:
                endpoints.append(c.tolist())
    endpoints = np.array(endpoints)
    return endpoints

def cal_theta_phi(p):
    """ Generate the vector of two points
        Return the radius, theta, phi
    """ 
    EPS = 1e-8
    gap_xyz = np.array([p[0], p[1], p[2]])
    direction = gap_xyz / ( np.power(np.power(gap_xyz, 2).sum(), 0.5) )
    r = np.power(np.power(gap_xyz, 2).sum(), 0.5)

    theta = math.acos(gap_xyz[2] / (r+EPS))
    phi = math.atan2(gap_xyz[1], gap_xyz[0])

    return [(theta/3.14)*180, (phi/3.14)*180]

def judge_quadrant(p):
    x,y,z = p
    if x >= 0:
        return 1
    else:
        return 0

def center_direction(points):
    """ 根据一组点拟合出一个方向向量, 尽可能经过所有点
    """
    xyz0 = np.mean(points,axis=0)
    centerLine = points - xyz0
    
    U,sigma,VT = la.svd(centerLine)
        
    direction = VT[0]  #法线方向
    return direction, xyz0
    
tif = tifffile.imread(FILE_PATH)

print(np.unique(tif))

single_ins = np.zeros(shape=[301,301,301])

single_ins[tif==2] = 200



idx = np.where(single_ins!=0)
coords = np.stack(idx, axis=0).transpose()

endpoints = []
gap_vectors = []
for i, c in enumerate(coords):
    dist = cdist(XA=np.expand_dims(c, axis=0), XB=coords, metric='euclidean')
    idx_dless5 = np.where(dist<20)[1]
    dless5 = coords[idx_dless5[idx_dless5!=i], :]
    gap_vectors.append((dless5 - c).tolist())
    gap_vector = dless5 - c
    theta_phi = [cal_theta_phi(i) for i in gap_vector]

    for j in range(len(theta_phi)):
        if theta_phi[j][1] > 100 or theta_phi[j][1] < -100:
            break
        if j == len(theta_phi)-1:
            endpoints.append(c.tolist())
endpoints = np.array(endpoints)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(dless5[:,0], dless5[:,1], dless5[:,2])

direction, xyz0 = center_direction(dless5)
ax.scatter(xyz0[0]+direction[0]*range(0,50,5), xyz0[1]+direction[1]*range(0,50,5), xyz0[2]+direction[2]*range(0,50,5), marker='x', s=5)
plt.show()

# single_ins[endpoints[:,0], endpoints[:,1], endpoints[:,2]] = 2000
# tifffile.imsave("test.tif", single_ins.astype(np.int16))


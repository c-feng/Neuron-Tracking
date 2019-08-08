import numpy as np
from .FINCH_Core.clustRank import clustRank
from .FINCH_Core.get_clust import get_clust
from .FINCH_Core.get_merge import get_merge
import time
from scipy.sparse import coo_matrix

a = [[0,0,0,1,1,0,0,0,0],
     [0,0,1,0,0,0,0,0,0],
     [0,1,0,1,1,0,0,0,0],
     [1,0,1,0,1,0,0,0,0],
     [1,0,1,1,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,1],
     [0,0,0,0,0,0,0,1,0]]

def finch(data, initial_rank, verbose):
    """ Input
            data: feature Matrix (feature vectors in rows)
            initial_rank [Opitional]: Nx1  1-neighbour indices vector ... pass empty [] to compute the 1st neighbor via pdist or flann
            verbos: 1 for print some output
    """
    # Initialize FINCH clustering
    min_sim = np.inf
    t0 = time.time()
    Affinity_, orig_dist, _ = clustRank(data, initial_rank)
    # Affinity_ = coo_matrix(a)
    # orig_dist = None
    print("Affinity Time: {}".format(time.time()-t0))
    
    initial_rank = []

    t1= time.time()
    Group_ = get_clust(Affinity_, [], np.inf)
    print("Clusting Time: {}".format(time.time()-t1))
    # print(Group_.shape, np.max(Group_), np.unique(Group_))
    
    t2= time.time()
    c, num_clust, mat = get_merge([], Group_, data)
    print("Merge Time: {}".format(time.time()-t2))

    if verbose == 1:
        print("Partition 0 : %d clusters\n" % (num_clust))
    
    if orig_dist is not None:
        min_sim = np.max(orig_dist[Affinity_.toarray() > 0])

    exit_clust = np.inf
    c_ = c

    num_clust = [num_clust]
    k = 1
    while exit_clust > 0:
        Affinity_, orig_dist, _ = clustRank(mat, initial_rank)

        u = get_clust(Affinity_, orig_dist, min_sim)
        c_, num_clust_curr, mat = get_merge(c_, u, data)

        num_clust += [num_clust_curr]
        if len(c_.shape) < 2:
            c_ = c_[:, None]
        if len(c.shape) < 2:
            c = c[:, None]
        c = np.concatenate([c, c_], axis=1)

        # exit if cluster is 1
        exit_clust = num_clust[-2] - num_clust_curr
        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            exit_clust = 0
            break
        if verbose == 1:
            print("Partition %d : %d clusters\n" % (k, num_clust[k]))
        k = k + 1

    return c, num_clust

if __name__ == "__main__":
    import scipy.io
    import h5py
    file_path = "/media/fcheng/FINCH-Clustering/data/STL-10/data.mat"
    # file_path = "/media/fcheng/FINCH-Clustering/data/mnist10k/data.mat"
    # file_path = r"D:\cf\Repositories\FINCH-Clustering\data\STL-10\data.mat"
    
    with h5py.File(file_path, 'r') as f:
        data = f['data']
        data = np.array(data).T

    tic = time.time()
    c, num_clust = finch(data,[], 1)
    print(num_clust)
    # print(c)
    print("Times: %d" % (time.time()-tic))
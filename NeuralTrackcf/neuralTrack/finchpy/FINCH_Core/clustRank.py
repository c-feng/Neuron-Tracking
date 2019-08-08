import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix, csr_matrix
# import matSimilarity

def Modular(A):
    return np.power(np.sum(A*A, axis=1), 0.5)

def MatCosSimlarity(A, B):
    axb = A @ B.T
    return 1. - axb / (Modular(A)[None, :].T @ Modular(B)[None,:])

def clustRank(mat, initial_rank):
    """ Implements the clustering equation
    """
    s = mat.shape[0]

    if initial_rank:
        orig_dist = None
    elif s <= 70000:
        # orig_dist = distance.cdist(mat, mat, metric='cosine')
        orig_dist = MatCosSimlarity(mat, mat)
        orig_dist[np.eye(orig_dist.shape[0], orig_dist.shape[1])>0] = np.inf
        d = np.min(orig_dist, axis=1)
        initial_rank = np.argmin(orig_dist, axis=1)
        min_sim = np.max(d)  # 最小相似性
    else:
        print('finding exact neghbours via pdist is not fesable on ram with data size of %d points.\nUsing flann to compute 1st-neighbours at this step ...\n\n '%s)
        initial_rank, d = flann_nn(mat, 8)
        print('step flann done...')
        min_sim = np.max(d)
        orig_dist = None
    
    # A = np.zeros(shape=(s, s))
    # idx = np.stack([np.arange(s), initial_rank], axis=0).T
    # A[idx[:, 0], idx[:, 1]] = 1
    # A = A + np.eye(s, s)
    # A = A * A.T
    # A[np.eye(s, s)>0] = 0

    A = coo_matrix((np.ones(s), (np.arange(s), initial_rank)), shape=(s, s))
    A = A + coo_matrix((np.ones(s), (np.arange(s), np.arange(s))), shape=(s, s))
    A = A @ A.T
    A[coo_matrix((np.ones(s), (np.arange(s), np.arange(s))), shape=(s, s))>0] = 0

    return A.tocoo(), orig_dist, min_sim

if __name__ == "__main__":
    import h5py
    import time
    file_path = "/media/fcheng/FINCH-Clustering/data/STL-10/data.mat"
    with h5py.File(file_path, 'r') as f:
        data = f['data']
        data = np.array(data).T

    a = np.array([[1, 3, 5, 6, 7, 8]])
    modu = Modular(a)
    print(modu)

    b = np.array([[0, 2, 3, 6, 7, 9]])
    cos = MatCosSimlarity(a, b)
    print(cos)

    A = np.array([[1, 2,3], [4, 5,6],[5,7,8]], dtype=np.float)
    B = np.array([[0, 2,3],[3,6, 7],[7,9, 8]], dtype=np.float)
    t0 = time.time()
    print(MatCosSimlarity(data, data))
    print("----------", time.time()-t0)

    # import matSimilarity
    # a_ = np.array([[1, 3, 5, 6, 7, 8]], dtype=np.float)
    # b_ = np.array([[0, 2, 3, 6, 7, 9]], dtype=np.float)
    # t1 = time.time()
    # s = matSimilarity.cal(data[:, :].tolist(), data[:, :].tolist())
    # print(np.array(s))
    # print("**********", time.time()-t1)
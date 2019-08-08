import numpy as np

def getC(G, u):
    b, m, n = np.unique(G, return_index=True, return_inverse=True)
    G_ = u[n]
    return G_

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None: 
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

def coolMean(M_, u):
    u_ = ind2vec(u.T).T
    nf = np.sum(u_, 1)
    idx = np.argsort(u)
    M = np.zeros(shape=(len(idx), M_.shape[1]))
    M = M_[idx, :]

    M = np.cumsum(np.concatenate([np.zeros((1, M.shape[1])), M]), axis=0)

    cnf = np.cumsum(nf)
    nf1 = [0] + (cnf).tolist()
    nf1 = np.array(nf1[:-1])
    s = np.stack([nf1, cnf], axis=1)
    
    M = M[np.array(s)[:, 1], :] - M[np.array(s)[:, 0], :]
    M = M / nf[:, None]

    return M

def get_merge(c, u, data):
    """ core procedure for mergeing in algorithm 1
    """
    u_ = ind2vec(u).T
    num_clust = u_.shape[0]

    if len(c):
        c = getC(c, u.T)
    else:
        c = u.T

    if num_clust <= 5e6:
        mat = coolMean(data, c)
    else:
        print("resorting to approx combining method ...")
    
        # 这里是Fisrt 模式, .m中是last
        _, ic, _ = np.unique(c, return_index=True, return_inverse=True)

        mat = data[ic, :]

    return c, num_clust, mat

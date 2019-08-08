import numpy as np
from scipy.sparse.csgraph import connected_components
from .graph import mat2edgeGraph

def get_clust(A_, orig_dist, min_sim):
    if min_sim != np.inf:
        # ind = np.stack(np.where(orig_dist*A > min_sim), axis=0).T
        A = A_.toarray()
        mask = orig_dist * A
        mask[np.isnan(mask)>0] = np.inf
        A[mask > min_sim] = 0
        # A = A.tocoo()
    else:
        A = A_
    # n_components, labels = connected_components(csgraph=A, directed=False, connection='weak', return_labels=True)
    g = mat2edgeGraph(A)
    cc = g.connectedComponents()
    labels = np.zeros(A.shape[0])
    for i, c in enumerate(cc):
        labels[c] = i

    return labels


"""
    [[0,0,0,1,1,0,0,0,0],
     [0,0,1,0,0,0,0,0,0],
     [0,1,0,1,1,0,0,0,0],
     [1,0,1,0,1,0,0,0,0],
     [1,0,1,1,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,1],
     [0,0,0,0,0,0,0,1,0]]
"""
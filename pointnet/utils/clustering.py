import os
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

def cluster(prediction):
    bandwidth = estimate_bandwidth(prediction, quantile=0.01)
    print("bandwidth: ", bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)

    ms.fit(prediction)
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers
from dtaidistance import dtw
from dtaidistance.util import SeriesContainer
import numpy as np
from sklearn_extra.cluster import KMedoids
from archive.clustering import cluster_labeling_to_dict
import sklearn

def compute_silhouette_score(distance_matrix, clustering_labels, seed = None):
    score = sklearn.metrics.silhouette_score(distance_matrix, clustering_labels, metric = 'precomputed', random_state = seed)
    return score

def determine_nb_of_clusters_silhouette(series, min_k, max_k, window, psi):
    series = SeriesContainer.wrap(series)
    distance_matrix = dtw.distance_matrix_fast(series, window=window, psi=psi, compact=False)
    # so this distance matrix is upper triangular but it needs to be a full matrix for the clusterer
    distance_matrix[np.isinf(distance_matrix)] = 0
    # this works because the diagonal is 0
    full_matrix = distance_matrix + distance_matrix.T

    def k_medoids(n_clusters):
        clusterer = KMedoids(n_clusters, metric='precomputed', init='k-medoids++', max_iter=1000)
        clusterer.old_fit(full_matrix)
        labels = clusterer.labels_
        return labels

    def silhouette_score(labels):
        return compute_silhouette_score(full_matrix,labels)

    best_silhouette = None
    best_k = None
    best_labels = None
    for k in range(min_k, max_k+1):
        labels = k_medoids(k)
        silhouette = silhouette_score(labels)
        if best_silhouette is None or silhouette>best_silhouette:
            best_k = k
            best_silhouette = silhouette
            best_labels = labels
    return best_k, cluster_labeling_to_dict(best_labels)



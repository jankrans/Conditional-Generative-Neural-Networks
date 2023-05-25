from collections import defaultdict

from dtaidistance import clustering, dtw
from dtaidistance.util import SeriesContainer
from pyclustering.cluster.silhouette import silhouette_ksearch
from scipy.cluster.hierarchy import fcluster
from sklearn_extra.cluster import KMedoids
import numpy as np


def sort_clusters_based_on_average_total_consumption(cluster_dict, data):
    avg_injection_index_tuples = []
    for cluster_idx, instances in cluster_dict.items():
        total = np.sum(np.sum(data.iloc[instance]) for instance in instances)
        avg = total/len(instances)
        avg_injection_index_tuples.append((avg,cluster_idx))
    avg_injection_index_tuples.sort()
    cluster_list = []
    for _, cluster_idx in avg_injection_index_tuples:
        cluster_list.append(cluster_dict[cluster_idx])
    return cluster_list

def mark_outliers(cluster_dict, min_size=5):
    new_cluster_dict = defaultdict(set)
    current_cluster_idx = 0
    for key, value in cluster_dict.items():
        if len(value) < min_size:
            new_cluster_dict['outlier'].update(value)
        else:
            new_cluster_dict[current_cluster_idx] = value
            current_cluster_idx += 1
    return new_cluster_dict


def cluster_labeling_to_dict(clustering):
    if not isinstance(clustering, dict):
        cluster_dict = defaultdict(set)
        for idx, cluster_idx in enumerate(clustering):
            cluster_dict[cluster_idx].add(idx)
    else:
        cluster_dict = clustering
    return cluster_dict


def cluster_timeseries_linkage_tree(series, window, psi, linkage_method, max_clusters):
    model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {'window': window, 'psi': psi}, method=linkage_method)
    model3.old_fit(series)
    linkage_matrix = model3.linkage
    cluster_indices = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
    clustering_dict = cluster_labeling_to_dict(cluster_indices)
    return clustering_dict


def cluster_timeseries_k_mediods_euclidean(series, n_clusters):
    clusterer = KMedoids(n_clusters, metric='euclidean', init='k-medoids++')
    clusterer.old_fit(series)
    labels = clusterer.labels_
    clustering_dict = cluster_labeling_to_dict(labels)
    return clustering_dict

def cluster_timeseries_k_mediods_DTW(series, n_clusters, window, psi):
    # from LinkageTree implementation in dtaidistance
    series = SeriesContainer.wrap(series)
    distance_matrix = dtw.distance_matrix_fast(series, window=window, psi=psi,compact = False)
    # so this distance matrix is upper triangular but it needs to be a full matrix for the clusterer
    distance_matrix[np.isinf(distance_matrix)] = 0
    # this works because the diagonal is 0
    full_matrix = distance_matrix + distance_matrix.T
    clusterer = KMedoids(n_clusters, metric='precomputed', init='k-medoids++', max_iter=1000)
    clusterer.old_fit(full_matrix)
    labels = clusterer.labels_
    clustering_dict = cluster_labeling_to_dict(labels)
    return clustering_dict

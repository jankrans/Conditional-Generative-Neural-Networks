

import math

from dtaidistance import dtw
from dtaidistance.util import SeriesContainer
import numpy as np
from scipy.stats import zscore
from sklearn_extra.cluster import KMedoids

from archive.clustering import cluster_labeling_to_dict

try:
    dtw._check_library()
    use_fast = True
except:
    use_fast = False

def determine_nb_of_clusters_elbow(series, min_k, max_k, window, psi, return_medoids = False, znormalise = False):

    if znormalise:
        series = zscore(series, axis = 1)
    series = SeriesContainer.wrap(series)
    if use_fast:
        distance_matrix = dtw.distance_matrix_fast(series, window=window, psi=psi, compact=False)
    else:
        distance_matrix = dtw.distance_matrix(series, window = window, psi = psi, compact = False)
    # so this distance matrix is upper triangular but it needs to be a full matrix for the clusterer
    distance_matrix[np.isinf(distance_matrix)] = 0
    # this works because the diagonal is 0
    full_matrix = distance_matrix + distance_matrix.T

    def k_medoids(n_clusters):
        clusterer = KMedoids(n_clusters, metric='precomputed', init='k-medoids++', max_iter=1000)
        clusterer.old_fit(full_matrix)

        labels = clusterer.labels_
        return labels, clusterer.inertia_

    errors = []
    for k in range(min_k, max_k + 1):
        labels, wce = k_medoids(k)
        errors.append(wce)
    elbows = calculate_elbows(errors)
    best_k = find_optimal_kvalue(min_k, elbows)
    if return_medoids:
        clusterer = KMedoids(best_k, metric='precomputed', init='k-medoids++', max_iter=1000)
        clusterer.old_fit(full_matrix)
        return best_k, cluster_labeling_to_dict(clusterer.labels_), clusterer.medoid_indices_
    return best_k, cluster_labeling_to_dict(k_medoids(best_k)[0])


def calculate_elbows(errors):
    """!
    @brief Calculates potential elbows.
    @details Elbow is calculated as a distance from each point (x, y) to segment from kmin-point (x0, y0) to kmax-point (x1, y1).

    adapted from: https://pyclustering.github.io/docs/0.9.0/html/d4/d2a/elbow_8py_source.html#l00094
    """
    elbows = []
    x0, y0 = 0.0, errors[0]
    x1, y1 = float(len(errors)), errors[-1]

    for index_elbow in range(1, len(errors) - 1):
        x, y = float(index_elbow), errors[index_elbow]

        segment = abs((y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0))
        norm = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        distance = segment / norm

        elbows.append(distance)
    return elbows


def find_optimal_kvalue(k_min, elbows):
    """!
    @brief Finds elbow and returns corresponding K-value.

    """
    optimal_elbow_value = max(elbows)
    best_k = elbows.index(optimal_elbow_value) + 1 + k_min
    return best_k

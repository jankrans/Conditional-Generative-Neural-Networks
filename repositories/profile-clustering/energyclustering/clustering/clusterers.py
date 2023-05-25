import warnings

from energyclustering.clustering.similarity import calculate_full_distance_matrix
from sklearn_extra.cluster import KMedoids
import dask.array as da
from dask_ml.metrics import pairwise_distances
from pathlib import Path
import pandas as pd

from energyclustering.clustering.similarity.wasserstein import wasserstein_distance_between_years

class PrecomputedClustering:
    def __init__(self, clustering):
        if isinstance(clustering, Path):
            clustering = pd.read_pickle(clustering)

        self.clustering = clustering
        self.labels_ = None

        warnings.warn('This is cheating! We construct the clustering using test and train data and then only use the training data but the clustering algorithm had access to both!')

    def fit(self, X, y = None):
        self.labels_ = self.clustering.loc[X.index].to_numpy()

class PrecomputedDistanceMetricClustering:
    def __init__(self, n_clusters, distance_metric, seed= None):
        if isinstance(distance_metric, Path):
            distance_metric = pd.read_pickle(distance_metric)

        # reset the index and columns
        distance_metric = distance_metric.pipe(lambda x: x.set_axis(x.index.map(str), axis = 0).set_axis(x.columns.map(str), axis = 1))
        self.distance_metric = distance_metric
        self.n_clusters = n_clusters
        self.clusterer = None
        self.seed = seed

    def fit(self, X, y = None):
        D = self.distance_metric.loc[X.index, X.index].to_numpy()
        self.clusterer = KMedoids(self.n_clusters, metric = 'precomputed', random_state = self.seed)
        self.clusterer.fit(D)
        return self

    @property
    def labels_(self):
        return self.clusterer.labels_

    @property
    def cluster_centers_(self):
        return self.clusterer.cluster_centers_


class MyKMedoids:
    def __init__(self, n_clusters, metric, seed = None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.seed = seed
        self.clusterer = None

    def fit(self, X, y = None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # chunk the array over the rows
        Xd = da.from_array(X, chunks = {0: 25, 1:-1})

        # calculate the distance matrix using dask
        D = pairwise_distances(Xd, X, metric = wasserstein_distance_between_years)
        D = D.compute()

        self.clusterer = KMedoids(self.n_clusters, metric = 'precomputed', random_state = self.seed)
        self.clusterer.fit(D)

    @property
    def labels_(self):
        return self.clusterer.labels_

    @property
    def cluster_centers_(self):
        return self.clusterer.cluster_centers_

class Clusterer:
    def __init__(self, n_clusters, metric=None,  dask_client = None, seed = None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.dask_client = dask_client
        self.seed = seed
        self.clustering = None

    def fit(self, household_info, consumption_data, distance_matrix = None):
        raise NotImplementedError()

    def k_medoids_cluster(self, data, distance_matrix = None):
        if distance_matrix is not None or not isinstance(self.metric, str):
            # use custom distance matrix
            if distance_matrix is None:
                distance_matrix = calculate_full_distance_matrix(data, self.metric, self.dask_client)
            clusterer = KMedoids(self.n_clusters, metric = 'precomputed', random_state = self.seed)
            clusterer.fit(distance_matrix)
        else:
            # use a metric
            clusterer = KMedoids(self.n_clusters, metric = self.metric, random_state=self.seed)
            clusterer.fit(data)
        self.clustering = clusterer.labels_
        self.centers = clusterer.cluster_centers_
        return self.clustering, self.centers

class ConsumptionClusterer(Clusterer):
    def __init__(self, n_clusters, metric=None,  dask_client=None, seed=None):
        super().__init__(n_clusters, metric, dask_client, seed)

    def fit(self, household_info = None, consumption_data = None, distance_matrix = None):
        return self.k_medoids_cluster(consumption_data, distance_matrix)


class MetadataClusterer(Clusterer):
    def __init__(self, n_clusters, metric=None, dask_client=None, seed=None):
        super().__init__(n_clusters, metric, dask_client, seed)

    def fit(self, household_info = None, consumption_data = None, distance_matrix = None):
        return self.k_medoids_cluster(household_info, distance_matrix)


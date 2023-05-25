import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import logging
import pandas as pd

class PreClusteringClusterer:
    """
        Clustering algorithm that clusters the full dataset using one algorithm and uses the second algorithm to cluster the representatives of the first clustering.
        This can be used to cluster a big dataset by first overclustering the dataset with a simple clustering algorithm but then refining that clustering by using a better clustering algorithm on the cluster representatives.
    """
    def __init__(self, n_clusters, pre_clusterer, post_clusterer):
        self.n_clusters = n_clusters

        self.pre_clusterer = pre_clusterer
        self.post_clusterer = post_clusterer

        self.labels_ = None
        self.inertia_ = None


    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        self.post_clusterer.n_clusters = self.n_clusters

        # if small number of data points, just only use the post_clusterer
        if data.shape[0]<= self.pre_clusterer.n_clusters:
            # logging.info(f'Clustering data of shape {data.shape}  with {repr(self.post_clusterer)}')
            self.labels_ = self.post_clusterer.fit(data).labels_.astype('int')
            self.inertia_ = self.post_clusterer.inertia_
            return self



        # otherwise use the pre_clusterer
        # logging.info(f'PreClustering data of shape {data.shape}  with {repr(self.pre_clusterer)}')
        pre_cluster_labels = self.pre_clusterer.fit(data).labels_.astype('int')

        # calculates the medoids per cluster
        cluster_idxs, medoid_idxs = self.calculate_medoids_per_cluster(pre_cluster_labels, data)

        medoids = data[medoid_idxs, :]
        # logging.info(f"PostClustering data of shape {medoids.shape} with {repr(self.post_clusterer)}")
        post_cluster_labels = self.post_clusterer.fit(medoids).labels_.astype('int')
        post_medoid_idxs = self.post_clusterer.medoids
        post_medoids = medoids[post_medoid_idxs, :]

        # in pre_cluster_labels assign each cluster to the cluster_label of its medoids in the post_cluster_labels
        sort_idx = np.argsort(cluster_idxs)
        idx = np.searchsorted(cluster_idxs, pre_cluster_labels, sorter=sort_idx)
        out = post_cluster_labels[sort_idx][idx]

        # store the result
        self.labels_ = out
        self.inertia_ = self.calculate_inertia(out, data, post_medoids)
        return self

    def calculate_inertia(self, labels, data, medoids):
        # put cluster labels and instance_idxs in the same array
        a = np.stack([labels, np.arange(0, labels.shape[0], dtype='int')], axis=1)

        # sort a on cluster_idx
        sort = a[:,0].argsort()
        a = a[sort]

        # get the unique cluster labels in the correct order!
        unique_labels = np.unique(a[:, 0])

        # split the sorted array on unique elements to get instances per cluster
        instances_per_cluster = np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])

        inertia = 0
        for idx, instances in enumerate(instances_per_cluster):
            distances_in_cluster = euclidean_distances(data[instances, :])
            medoid_index = np.argmin(distances_in_cluster.sum(axis=0))
            inertia += (distances_in_cluster[medoid_index,:]**2).sum()

        return inertia



    def calculate_medoids_per_cluster(self, labels, data):
        # put cluster labels and instance_idxs in the same array
        a = np.stack([labels, np.arange(0, labels.shape[0], dtype = 'int')], axis=1)

        # sort a on cluster_idx
        a = a[a[:, 0].argsort()]

        # get the unique cluster labels in the correct order!
        unique_labels = np.unique(a[:, 0])

        # split the sorted array on unique elements to get instances per cluster
        instances_per_cluster = np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])


        # calculate the medoid of each cluster
        all_medoid_idxs = np.zeros(unique_labels.shape[0], dtype = 'int')
        for idx, instances in enumerate(instances_per_cluster):
            distances_in_cluster = euclidean_distances(data[instances, :])
            medoid_index = np.argmin(distances_in_cluster.sum(axis=0))
            all_medoid_idxs[idx] = instances[medoid_index]
        return unique_labels, all_medoid_idxs

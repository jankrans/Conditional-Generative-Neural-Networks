import numpy as np
from sklearn_extra.cluster import KMedoids

from energyclustering.cobras.superinstance_kmeans import SuperInstance_kmeans
from energyclustering.cobras.cobras import COBRAS

from sklearn.cluster import KMeans

from energyclustering.cobras.superinstance_kmedoids import SuperInstance_kmedoids, get_prototype


class COBRAS_kmedoids(COBRAS):

    def split_superinstance(self, si, k, seed):
        """
            Splits the given super-instance using k-means
        """
        # pandas like indexing (take all rows with indices and all columns with indices)
        relevant_distances = self.data[np.ix_(si.indices, si.indices)]
        km = KMedoids(k, metric = 'precomputed', random_state= seed)
        km.old_fit(relevant_distances)

        split_labels = km.labels_.astype(np.int)

        training = []
        no_training = []

        for new_si_idx in set(split_labels):
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance_kmedoids(self.data, cur_indices, self.train_indices, si))
            else:
                no_training.append((cur_indices, get_prototype(self.data, cur_indices)))

        for indices, representative_idx in no_training:
            closest_train = min(training, key=lambda x: self.data[x.representative_idx, representative_idx])
            closest_train.indices.extend(indices)

        si.children = training

        return training

    def create_superinstance(self, indices, parent=None):
        """
            Creates a super-instance of type SuperInstance_kmeans
        """

        return SuperInstance_kmedoids(self.data, indices, self.train_indices, parent)
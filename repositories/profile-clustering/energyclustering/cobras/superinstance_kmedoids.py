import numpy as np
from .superinstance import SuperInstance



def get_prototype(A,indices):
    # pandas like indexing (take all rows with indices and all columns with indices)
    relevant_distances = A[np.ix_(indices,indices)]
    total_distance_per_instance = np.sum(relevant_distances, axis = 1)
    minimum_relative_idx = np.argmin(total_distance_per_instance)
    return indices[minimum_relative_idx]


class SuperInstance_kmedoids(SuperInstance):

    def __init__(self, data, indices, train_indices, parent=None):
        """
            Chooses the super-instance representative as the instance closest to the super-instance centroid
        """
        super(SuperInstance_kmedoids, self).__init__(data, indices, train_indices, parent)

        self.representative_idx = get_prototype(self.data, self.train_indices)



    def distance_to(self, other_superinstance):
        """
            The distance between two super-instances is equal to the distance between there centroids  
        """
        return self.data[self.representative_idx, other_superinstance.representative_idx]

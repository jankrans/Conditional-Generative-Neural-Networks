import kmedoids

class CustomKMedoids:
    def __init__(self, n_clusters, metric, random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.labels_ = None
        self.medoids = None
        self.inartia_ = None

    def fit(self, data):
        matrix = self.metric(data)
        km = kmedoids.KMedoids(self.n_clusters, method='fasterpam', random_state=self.random_state)
        c = km.fit(matrix)
        self.inertia_ = c.inertia_
        self.labels_ = c.labels_.astype('int')
        self.medoids = c.medoid_indices_
        return self


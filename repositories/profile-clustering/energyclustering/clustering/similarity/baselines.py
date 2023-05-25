from sklearn.metrics import euclidean_distances
import numpy as np
import dtaidistance.dtw as dtw


class EuclideanDistance:
    def __init__(self):
        pass

    def distance(self, profile1, profile2):
        profile1 = profile1.fillna(profile1.mean())
        profile2 = profile2.fillna(profile2.mean())
        return np.linalg.norm(profile1 - profile2)


class DTWDistance:
    def __init__(self, window):
        self.window = window

    def distance(self, profile1, profile2):
        profile1 = profile1.fillna(profile1.mean())
        profile2 = profile2.fillna(profile2.mean())
        return dtw.distance(profile1, profile2, window = self.window)

def euclidean_distance_matrix(data_df):
    # replace NaN values with average consumption of profile
    data_df = data_df.apply(lambda row: row.fillna(row.mean()), axis = 1)
    distances = euclidean_distances(data_df.to_numpy())
    return distances

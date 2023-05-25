from sklearn.metrics import euclidean_distances
import pandas as pd

def get_euclidean_distance_matrix(matrix1, matrix2, **kwargs):
    m1 = matrix1.to_numpy()
    m2 = None if matrix2 is None else matrix2.to_numpy()
    distances = euclidean_distances(m1, m2)
    return pd.DataFrame(distances, index = matrix1.index, columns = matrix2.index)
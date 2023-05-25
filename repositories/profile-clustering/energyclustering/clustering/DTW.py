from joblib import parallel_backend
from tslearn.metrics import cdist_dtw, dtw
import pandas as pd
import numpy as np

def get_DTW_distance_matrix_w_mask(matrix1, matrix2, mask, window = 4, n_jobs = 1):
    def calculate_distance(profiles):
        profile1, profile2 = profiles
        return dtw(matrix1.loc[profile1], matrix2.loc[profile2], sakoe_chiba_radius=window)
    # figure out entries that need to be calculated
    entries_to_calculate = (
        mask.stack().pipe(lambda x: x[x==0]).index
    )
    distances = entries_to_calculate.map(calculate_distance)
    cost_matrix = (
        pd.Series(distances, index = entries_to_calculate)
        .to_frame('distance')
        .rename_axis(['meterID1', 'meterID2'], axis = 0)
        .reset_index()
        .pivot_table(index = 'meterID1', columns = 'meterID2', values = 'distance')
        .fillna(np.inf)
    )
    return cost_matrix

def get_DTW_distance_matrix(matrix1, matrix2 = None, window = 4, n_jobs = 4):
    """
        Calculate similarity between all rows of matrix 1 with all rows of matrix 2 (if matrix2 is non calculate self similarity of matrix1)
    """
    m1 = matrix1.to_numpy()
    m2 = None if matrix2 is None else matrix2.to_numpy()
    if n_jobs > 1:
        with parallel_backend('loky', n_jobs=n_jobs):
            distance_matrix = cdist_dtw(m1, m2, sakoe_chiba_radius = window, n_jobs = n_jobs)
    else:
        distance_matrix = cdist_dtw(m1, m2, sakoe_chiba_radius=window, n_jobs = n_jobs)
    return pd.DataFrame(distance_matrix, index = matrix1.index, columns = matrix2.index)
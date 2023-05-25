"""
    File that contains the code to calculate the similarity metric
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from energyclustering.clustering.DTW import get_DTW_distance_matrix, get_DTW_distance_matrix_w_mask
from energyclustering.clustering.euclidean import get_euclidean_distance_matrix
from energyclustering.util import series_to_daily_dataframe

DISTANCE_METRICS = dict(
    DTW=get_DTW_distance_matrix,
    euclidean=get_euclidean_distance_matrix,
)

def calculate_day_to_day_matching(cost_matrix):
    min_dim = min(cost_matrix.shape)
    return np.arange(1,min_dim), np.arange(1, min_dim)

DAY_MATCHINGS = dict(
    minimal_cost=linear_sum_assignment,
    one_to_one=calculate_day_to_day_matching,
)


class MatchingDistanceMeasure:
    def __init__(self, distance_metric='DTW', day_matching='minimal_cost', window=4):
        self.distance_metric = distance_metric
        self.day_matching = day_matching
        self.window = window

    def distance(self, profile1, profile2):
        return profile_distance_based_on_matching((profile1, profile2), self.distance_metric, self.day_matching,
                                                  self.window)


def profile_distance_based_on_matching(profile_tuple, distance_metric='DTW', day_matching='minimal_cost', window=4,
                                       n_jobs=1):
    """
        profile1 and profile2 are just pandas series with the full timeseries of each profile
        NaNs are allowed
    """
    profile1, profile2 = profile_tuple

    # convert profiles to daily dfs
    daily_df1 = series_to_daily_dataframe(profile1)
    daily_df2 = series_to_daily_dataframe(profile2)

    # correction factor
    number_of_days = min(daily_df1.shape[0], daily_df2.shape[0])
    correction_factor = 365 / number_of_days

    # calculate the distance matrix
    distance_metric = DISTANCE_METRICS[distance_metric]
    distance_matrix = distance_metric(daily_df1, daily_df2, window=window, n_jobs=n_jobs)

    # calculate the matching problem
    calculate_day_matching = DAY_MATCHINGS[day_matching]
    cost_matrix = distance_matrix.to_numpy()
    row_idx, col_idx = calculate_day_matching(cost_matrix)
    best_cost = cost_matrix[row_idx, col_idx].sum()

    # apply the correction factor
    corrected_best_cost = best_cost * correction_factor

    return corrected_best_cost


def profile_distance_based_on_matching_w_seasonality(profile_tuple, seasonality_window=30, window=4, n_jobs=1):
    profile1, profile2 = profile_tuple
    # convert profiles to daily dfs
    daily_df1 = series_to_daily_dataframe(profile1, dropna=False)
    daily_df2 = series_to_daily_dataframe(profile2, dropna=False)

    # make mask for disallowed assignments
    n = 366
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, 1 + seasonality_window)] = 1
    mask[np.triu_indices(n, n - seasonality_window)] = 0
    mask[np.tril_indices(n, -1 - seasonality_window)] = 1
    mask[np.tril_indices(n, -n + seasonality_window)] = 0
    mask = pd.DataFrame(mask, index=daily_df1.index, columns=daily_df2.index)

    # drop the na days
    is_na1 = daily_df1.isna().any(axis=1)
    is_na2 = daily_df2.isna().any(axis=1)
    mask = mask.loc[~is_na1, ~is_na2]
    daily_df1 = daily_df1.dropna(axis=0)
    daily_df2 = daily_df2.dropna(axis=0)

    # correction factor
    number_of_days = min(daily_df1.shape[0], daily_df2.shape[0])
    correction_factor = 365 / number_of_days

    # calculate the dtw distance matrix
    distance_matrix = get_DTW_distance_matrix_w_mask(daily_df1, daily_df2, mask, window=window, n_jobs=n_jobs)

    # calculate the matching problem
    cost_matrix = distance_matrix.to_numpy()
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    best_cost = cost_matrix[row_idx, col_idx].sum()

    corrected_best_cost = best_cost * correction_factor

    return corrected_best_cost


def distance_matrix(data_df, client=None, distance_config=None, seasonality_window=None, total_blocks=500):
    """
        Main function of this module, calculates a distance matrix by dividing it into blocks and calculating each block
        separately using 'calculate_block'

        Args:
            data_df: the data_df containing the consumption data
            total_blocks: the amount of blocks that will be used
            client: a dask_client used for parallel computation OR None in case you want local execution
            window, seasonality_window: arguments for the calculate_block_function
    """
    if distance_config is None:
        distance_config = dict()
    # calculate the distances block per block
    nb_series = data_df.shape[0]
    blocks = _generate_blocks(nb_series, total_blocks)
    tasks = [(data_df.iloc[row_start:row_end, :], data_df.iloc[column_start: column_end]) for
             (row_start, row_end), (column_start, column_end) in tqdm(blocks)]
    if client is None:
        results = [_calculate_block(task, distance_config, seasonality_window=seasonality_window) for task in
                   tqdm(tasks)]
    else:
        futures = client.map(_calculate_block, tasks, distance_config=distance_config,
                             seasonality_window=seasonality_window)
        results = client.gather(futures)

    # fill in blocks into a single distance_matrix (upper triangular)
    dist_matrix = np.zeros((nb_series, nb_series))
    for result, block in zip(results, blocks):
        dist_matrix[block[0][0]: block[0][1], block[1][0]:block[1][1]] = result

    # make upper triangular matrix into full symmetrical distance matrix
    dist_matrix[np.triu_indices(data_df.shape[0], k=1)] = 0
    dist_matrix = dist_matrix + dist_matrix.T
    return dist_matrix


def _generate_blocks(nb_series, total_blocks=500):
    """
        A util function that divides the full matrix into several (equally-sized) blocks that can be calculated in parallel
        The function won't generate 'total_blocks' directly but will simply try to find a number close enough

        Returns a list of (start_row, end_row),(start_col, end_col)
    """
    blocks_each_dimension = math.ceil(math.sqrt(total_blocks))
    profiles_per_block = math.ceil(nb_series / blocks_each_dimension)
    blocks = []
    for row_start in range(0, nb_series, profiles_per_block):
        row_end = min(row_start + profiles_per_block, nb_series)
        for column_start in range(0, row_start + 1, profiles_per_block):
            column_end = min(column_start + profiles_per_block, nb_series)
            blocks.append(((row_start, row_end), (column_start, column_end)))
    return blocks


def _calculate_block(profile_tuple, distance_config, seasonality_window=None):
    """
        Calculates the distances between the first and second collection of profiles (in tuple profile_tuple)
    """
    profiles1, profiles2 = profile_tuple
    distance_matrix = np.zeros((profiles1.shape[0], profiles2.shape[0]))
    for idx1, (index, profile1) in enumerate(profiles1.iterrows()):
        for idx2, (index, profile2) in enumerate(profiles2.iterrows()):
            if seasonality_window is None:
                distance = profile_distance_based_on_matching((profile1, profile2), **distance_config)
            else:
                distance = profile_distance_based_on_matching_w_seasonality((profile1, profile2),
                                                                            seasonality_window=seasonality_window,
                                                                            **distance_config)
            distance_matrix[idx1, idx2] = distance
    return distance_matrix


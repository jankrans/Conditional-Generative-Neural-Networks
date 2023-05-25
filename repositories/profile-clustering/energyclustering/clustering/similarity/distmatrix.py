import math

from distributed import Client
from tqdm import tqdm
import numpy as np
import pandas as pd

def calculate_distance_between_queries(data_df, queries, metric, dask_client: Client= None, n_blocks = None):
    involved_instances = np.unique(queries, axis = None)
    relevant_data = data_df.reset_index(drop=True).loc[involved_instances]
    chunks = np.array_split(queries, n_blocks)
    if dask_client is None:
        results = [_calculate_pair_list(task, metric, relevant_data) for task in tqdm(chunks, desc='calculating distances')]
    else:
        data_df_future = dask_client.scatter(relevant_data, broadcast=True)
        futures = dask_client.map(_calculate_pair_list, chunks, metric = metric, data_df = data_df_future)
        results = dask_client.gather(futures)

    # collect the results in a distance matrix
    n_series = relevant_data.shape[0]
    dist_matrix = np.full((n_series, n_series), np.nan)
    dist_matrix = pd.DataFrame(dist_matrix, index = relevant_data.index, columns = relevant_data.index)
    for chunk, result in zip(chunks, results):
        for (i1,i2), r in zip(chunk, result):
            dist_matrix.loc[i1, i2] = r
            dist_matrix.loc[i2, i1] = r


    # make into df with original index
    distance_df = pd.DataFrame(dist_matrix.to_numpy(), index= data_df.index[involved_instances], columns = data_df.index[involved_instances])
    return distance_df

def calculate_full_distance_matrix(data_df, metric, dask_client:Client=None, n_blocks = None):
    """
        calculates the distance matrix for the given data_df
    """
    if n_blocks is None:
        if dask_client is not None:
            n_blocks = len(dask_client.scheduler_info()['workers'])*10
        else:
            n_blocks = 1

    # Make the tasks
    n_series = data_df.shape[0]
    print('generating blocks')
    blocks = _generate_blocks(n_series, n_blocks)
    # tasks = [(data_df.iloc[row_start: row_end,:],data_df.iloc[column_start:column_end]) for
    #          (row_start, row_end), (column_start, column_end) in tqdm(blocks, desc='Making blocks')]
    print('calculating blocks')
    # execute the tasks
    if dask_client is None:
        results = [_calculate_block(task, metric, data_df) for task in tqdm(blocks, desc='Calculating distances')]
    else:
        data_df_future = dask_client.scatter(data_df, broadcast = True)
        futures = dask_client.map(_calculate_block, blocks, metric = metric, data_df = data_df_future)
        results = dask_client.gather(futures)

    # gather the results
    dist_matrix = np.zeros((n_series, n_series))
    for result, block in zip(results, blocks):
        dist_matrix[block[0][0]: block[0][1], block[1][0]:block[1][1]] = result

    # make upper triangular matrix into full symmetrical distance matrix
    dist_matrix[np.triu_indices(data_df.shape[0], k=1)] = 0
    dist_matrix = dist_matrix + dist_matrix.T

    # make into a nice dataframe
    distance_df = pd.DataFrame(dist_matrix, index=data_df.index, columns=data_df.index)
    return distance_df

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

def _calculate_pair_list(query_indices, metric, data_df):
    result = []
    for i1, i2 in query_indices:
        profile1 = data_df.loc[i1]
        profile2 = data_df.loc[i2]
        distance = metric.distance(profile1, profile2)
        result.append(distance)
    return result

def _calculate_block(block_indices, metric, data_df):
    """
        Calculates the distances between the first and second collection of profiles (in tuple profile_tuple)
    """
    (row_start, row_end), (column_start, column_end) = block_indices
    profiles1 = data_df.iloc[row_start: row_end]
    profiles2 = data_df.iloc[column_start: column_end]
    distance_matrix = np.zeros((profiles1.shape[0], profiles2.shape[0]))
    for idx1, (index, profile1) in enumerate(profiles1.iterrows()):
        for idx2, (index, profile2) in enumerate(profiles2.iterrows()):
            distance = metric.distance(profile1, profile2)
            distance_matrix[idx1, idx2] = distance
    return distance_matrix
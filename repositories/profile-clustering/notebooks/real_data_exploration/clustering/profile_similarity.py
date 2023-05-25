import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
import itertools 

def profile_distance_matrix_based_on_daily_clustering(labels, medoid_distance_matrix):
    """
        This can be optimized! Is sequential but can be parallelized!
    """
    all_profiles = labels.index.get_level_values(0).unique()
    distance_matrix = np.zeros((len(all_profiles), len(all_profiles)))
    for idx1, idx2 in itertools.combinations(range(0,len(all_profiles)), 2):
        meterID1 = all_profiles[idx1]
        meterID2 = all_profiles[idx2] 
        distance = profile_distance_based_on_daily_clustering(meterID1, meterID2, labels, medoid_distance_matrix)
        distance_matrix[idx1, idx2] = distance 
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix = pd.DataFrame(distance_matrix, index = all_profiles, columns = all_profiles)
    return distance_matrix

def profile_distance_based_on_daily_clustering(profile1, profile2, labels, medoid_distance_matrix): 
 
    # cluster labels of each profile
    labels1 = labels.loc[profile1].value_counts()
    labels2 = labels.loc[profile2].value_counts()
#     display(labels1.to_frame('labels1'))
#     display(labels2.to_frame('labels2'))

    # put them in the same df 
    both_labels = labels1.to_frame('labels1').join(labels2.to_frame('labels2'), how = 'outer')
#     display(both_labels)
    # remove the matches 
    both_labels = both_labels.subtract(both_labels.min(skipna = False, axis = 1).fillna(0), axis = 0)
#     display(both_labels)
    # replace zero with Nan 
    both_labels = both_labels.replace({0.0:np.NaN})

    # remove all rows with NaN twice 
    both_labels = both_labels.dropna(axis = 0, how = 'all')
#     display(both_labels)
    # get the row clusters and column clusters 
    rows = both_labels['labels1'].dropna()
    columns = both_labels['labels2'].dropna()

    # preallocate the cost matrix (use pandas to keep it easy, speedup with numpy probably possible)
    row_index = []
    for cluster, times in rows.iteritems():
        row_index.extend([cluster]*int(times))
    column_index = []
    for cluster,times in columns.iteritems(): 
        column_index.extend([cluster]*int(times))
    cost_matrix = pd.DataFrame(index = row_index, columns = column_index, dtype = 'float')

    # fill the cost matrix with DTW distances between medoids 
    for row, column in itertools.product(cost_matrix.index.unique(), cost_matrix.columns.unique()):
        distance = medoid_distance_matrix[row,column]
#         medoid1 = centers.iloc[row].to_numpy()
#         medoid2 = centers.iloc[column].to_numpy()
#         distance = dtw.distance(medoid1, medoid2, window =4, psi = 0, use_c = True)
        cost_matrix.loc[row,column] = distance
    
    # solve the matching problem
    cost_array = cost_matrix.to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost_array)
    
    # calculate the optimal distance
    best_cost = cost_array[row_ind, col_ind].sum()
    return best_cost
from pathlib import Path 
import pandas as pd 
import numpy as np 
from dtaidistance import clustering, dtw
from dtaidistance.util import SeriesContainer
from tslearn.metrics import cdist_dtw
from dtaidistance.dtw_barycenter import dba
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering, KMeans
from scipy.spatial import distance_matrix
from joblib import parallel_backend
from sklearn.neighbors import LocalOutlierFactor
PRE_PATH = Path('/cw/dtaiproj/ml/2020-FLAIR-VITO/profile-clustering/preprocessed/combined')

def read_data(nrows = 100): 
    info_path = PRE_PATH/'reindexed_info.csv'
    data_path = PRE_PATH/'reindexed_DST_data.csv'
    info_df = pd.read_csv(info_path, index_col = [0,1], nrows = nrows)
    data_df = pd.read_csv(data_path, index_col = [0,1],nrows =nrows)
    data_df.columns = pd.to_datetime(data_df.columns)
    data_df.columns.name = 'timestamp'
    return info_df, data_df 

def read_data_pickle(): 
    info_path = PRE_PATH/'reindexed_info.pkl'
    data_path = PRE_PATH/'reindexed_DST_data_w_errors.pkl'
    info_df = pd.read_pickle(info_path)
    data_df = pd.read_pickle(data_path)
    return info_df, data_df

def get_profile_matrix(profile_series): 
    return (profile_series
           .to_frame('value')
            .assign(
                time = lambda x: add_date(x.index.time),
                date = lambda x: x.index.date.astype('str')
            )
            .pipe(lambda x: pd.pivot_table(x, index = 'date', columns = 'time', values = 'value'))
            .dropna(axis = 0)
           )

def get_day_df(data_df): 
    day_df = (
        data_df
        .stack()
        .to_frame('value')
        .assign(
            time = lambda x: add_date(x.index.get_level_values(2).time),
            date = lambda x: x.index.get_level_values(2).date.astype('str')
        )
        .pipe(lambda x: pd.pivot_table(x, index = ['meterID','year','date'], columns = 'time', values = 'value'))
        .dropna(axis = 0)
    )
    return day_df

def get_euclidean_distance_matrix(matrix): 
    return distance_matrix(matrix, matrix)

def get_medoid(matrix): 
    distance_matrix = get_euclidean_distance_matrix(matrix)
    medoid_id = np.argmin(distance_matrix.sum(axis=0))
    return medoid_id

def get_medoids_per_cluster(labels, data_df): 
    all_labels = labels.unique()
    medoids = pd.DataFrame(index = all_labels, columns = data_df.columns)
    for cluster_label in all_labels: 
        instances_in_cluster = data_df.loc[labels == cluster_label]
        medoid_index = get_medoid(instances_in_cluster.to_numpy())
        medoids.loc[medoid_index,:] = instances_in_cluster.iloc[medoid_index]
    return medoids
        
def get_DTW_distance_matrix(matrix, window, psi, njobs = 4): 
    with parallel_backend('loky', n_jobs=njobs):
        return cdist_dtw(matrix, sakoe_chiba_radius = window, n_jobs = njobs)
#     pass
def get_DTW_distance_matrix_old(matrix, window, psi, parallel = True): 
    series = SeriesContainer.wrap(matrix.to_numpy())
    distance_matrix = dtw.distance_matrix_fast(series, window=window, psi=psi,compact = False, parallel = True)
    return distance_matrix


    
    


def barycentric_averaging(profile_matrix): 
    series = SeriesContainer.wrap(profile_matrix.to_numpy())
    barycentric_average = pd.Series(dba(series,None), index = profile_matrix.columns)
    return barycentric_average

def lof_anomaly_detection(profile_matrix, distance_matrix, n_neighbors = 20, contamination = 0.1): 
    detector = LocalOutlierFactor(n_neighbors = n_neighbors, metric = 'precomputed', contamination = contamination)
    labels = detector.fit_predict(distance_matrix)
    anomaly_labels = pd.Series(labels == -1, index = profile_matrix.index, name = 'anomaly')
    return anomaly_labels

def cluster_spectral(profile_matrix, distance_matrix, nb_of_clusters, random_state = None):
    beta = 1
    similarity_matrix = np.exp(-beta * distance_matrix / distance_matrix.std())
    clusterer = SpectralClustering(n_clusters = nb_of_clusters, affinity = 'precomputed', random_state = random_state)
    clusterer.fit(distance_matrix)
    labels = clusterer.labels_
    labels = pd.Series(labels, index = profile_matrix.index, name = 'labels')
    return labels
    
def cluster_KMedoids(profile_matrix,  nb_of_clusters,distance_matrix = None, random_state = None): 
    if distance_matrix is not None: 
        clusterer = KMedoids(nb_of_clusters, metric='precomputed', init='k-medoids++', max_iter=1000, random_state = random_state)
        clusterer.old_fit(distance_matrix)
    else: 
        clusterer = KMedoids(nb_of_clusters, metric='euclidean', init='k-medoids++', max_iter=1000, random_state = random_state)
        clusterer.old_fit(profile_matrix.to_numpy())
    labels = clusterer.labels_
    centers = profile_matrix.iloc[clusterer.medoid_indices_]
    labels = pd.Series(labels, index = profile_matrix.index, name = 'labels')
    return labels, centers

def cluster_KMeans(profile_matrix,  nb_of_clusters, random_state = None): 
    clusterer = KMeans(nb_of_clusters, max_iter=1000, random_state = random_state)
    clusterer.fit(profile_matrix.to_numpy())
    labels = clusterer.labels_
    centers = pd.DataFrame(clusterer.cluster_centers_, columns = profile_matrix.columns)
    labels = pd.Series(labels, index = profile_matrix.index, name = 'labels')
    return labels, centers

def add_date(series): 
    return pd.to_datetime(series, format='%H:%M:%S', exact = False)

def add_time(series): 
    return pd.to_datetime(series)
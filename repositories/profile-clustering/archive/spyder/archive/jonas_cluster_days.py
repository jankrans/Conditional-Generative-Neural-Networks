#%% imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import clustering
from dtaidistance import dtw
import archive.visualisation.visualise_ts_clustering as clustering_vis
from energyclustering import data as data
from scipy.cluster.hierarchy import dendrogram, fcluster

#%%
measurements_per_day = data.get_consumption_per_day()
master_table = data.get_master_table()

#%%
# iID's that have local production and that only have a single meter
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]
ids_without_production = master_table[(master_table['Lokale productie'] == 'Nee') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]
print(f"Number of normal installations with production: {len(ids_with_production)}")
print(f"Number of normal installations without production: {len(ids_without_production)}")
#%%
# choose one of these
id_to_investigate = ids_with_production.iat[0]
print(f"investigating: {id_to_investigate}")
df_per_day:pd.DataFrame = measurements_per_day.loc[id_to_investigate]
#%%
#
print("for now we just ignore missing data")
df_per_day.dropna(inplace=True,axis=0)
nb_of_zeros = (df_per_day == 0).sum().sum()
total = (df_per_day.shape[0]*df_per_day.shape[1])
print(f"nb of zero readings: {nb_of_zeros} or {(nb_of_zeros/total)*100 :0.5f}%")

#%%
# convert to numpy array and apply clustering
series = df_per_day.to_numpy()
print(series.shape)
#%%
"""
# dists = dtw.distance_matrix_fast(series,window = 4)
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {'window': 4})
# model1.fit(series)
model = clustering.HierarchicalTree(model1)
model.fit(series)
"""
#%%
"""
# ok so getting a single clustering out of this does not seem very simple
# so I'll take a detour I'll just retreive all distance thresholds used.
# and then based on these I'll rerun the clustering model to get an actual clustering!

# this variable contains indices of the merge but also the distance
linkage = model.linkage
# note: the last value in the tuple is always 0?
distances = [distance for i1, i2, distance,_ in linkage]
# reverse such that distances for last merges are first in the list
distances.reverse()
# so if I want x clusters I need to go x distances further
nb_of_desired_clusters = 10
distance_threshold = (distances[nb_of_desired_clusters] + distances[nb_of_desired_clusters])/2
print(f"threshold to get {nb_of_desired_clusters} clusters: {distance_threshold}")
#%%
model2 = clustering.Hierarchical(dtw.distance_matrix_fast, {'window':4}, max_dist = distance_threshold)
cluster_indices = model2.fit(series)
print(f"found {len(cluster_indices)} clusters")
"""

#%%
# cluster the series
show_dendogram = False
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {'window':4}, method = 'centroid')
model3.old_fit(series)
linkage_matrix = model3.linkage
if show_dendogram:
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(linkage_matrix)
    plt.show()
max_clusters = 40
cluster_indices = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
#%%
figure_dir = Path().absolute()/"figures"/"initial_clustering"
fig, axes = clustering_vis.visualise_ts_clusters_per_day(cluster_indices, df_per_day, minimal_size=3)
plt.savefig(figure_dir/"centroid_production.png", layout = 'tight')
#%%
clustering_vis.show_daily_cluster_indices_over_year(cluster_indices, df_per_day, path =figure_dir / "centroid_production_daily.png")

clustering_vis.show_cluster_indices_over_time_weekdays(cluster_indices, df_per_day, path = figure_dir/"centroid_production_per_month.png")




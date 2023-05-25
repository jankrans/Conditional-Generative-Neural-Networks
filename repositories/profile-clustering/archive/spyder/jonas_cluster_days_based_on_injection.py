# %% imports
from pathlib import Path

import pandas as pd

from energyclustering import data as data
import archive.visualisation.visualise_ts_clustering as clustering_vis
# %%
from archive.clustering import determine_nb_of_clusters_elbow

measurements_per_day = data.get_timeseries_per_day('Injection')
master_table = data.get_master_table()
FIGURE_DIR = Path() / "figures" / "initial_clustering_consumption_silhouette"
print(measurements_per_day.index.names)

# %%
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]

# choose one or just loop over several
id_to_investigate = ids_with_production.iloc[2]
# for id_to_investigate in ids_with_production:
print(f"investigating: {id_to_investigate}")

# %%
# gather the data and put in correct format
df_per_day: pd.DataFrame = measurements_per_day.loc[id_to_investigate]
print("for now we just ignore missing data")
df_per_day.dropna(inplace=True, axis=0)
nb_of_zeros = (df_per_day == 0).sum().sum()
total = (df_per_day.shape[0] * df_per_day.shape[1])
print(f"nb of zero readings: {nb_of_zeros} or {(nb_of_zeros / total) * 100 :0.5f}%")
series = df_per_day.to_numpy()
# %%
# convert to numpy array and apply clustering

# %%
# cluster the series
# cluster_dict = cluster_timeseries_linkage_tree(series, 6, 4, 'centroid', 40)

n_clusters ,cluster_dict = determine_nb_of_clusters_elbow(series, 4, 20, 6, 4)
# cluster_dict = mark_outliers(cluster_dict, min_size=5)

# rename clusters to be consistent
# if id_to_investigate == 'yORhoNfIwXwAbQ':
#     new_cluster_dict = dict()
#     new_cluster_dict[0] = cluster_dict[1]
#     new_cluster_dict[1] = cluster_dict[0]
#     new_cluster_dict[2] = cluster_dict[3]
#     new_cluster_dict[3] = cluster_dict[2]
#     cluster_dict = new_cluster_dict
# elif id_to_investigate == 'e6qF0dxU5g1/gw':
#     new_cluster_dict = dict()
#     new_cluster_dict[0] = cluster_dict[0]
#     new_cluster_dict[1] = cluster_dict[3]
#     new_cluster_dict[2] = cluster_dict[1]
#     new_cluster_dict[3] = cluster_dict[2]
#     cluster_dict = new_cluster_dict
# #%%

# clustering_vis.visualise_ts_cluster_per_day_altair_interactive(cluster_dict, df_per_day)
#%%
figure_dir = FIGURE_DIR/ str(id_to_investigate).replace("/", "")
figure_dir.mkdir(parents=True, exist_ok=True)
clustering_vis.visualise_ts_clusters_per_day(cluster_dict, df_per_day, figure_dir / 'clustering.png')
clustering_vis.show_daily_cluster_indices_over_year(cluster_dict, df_per_day, path=figure_dir / "yearly_overview.png")
clustering_vis.show_daily_cluster_indices_over_calendar(cluster_dict, df_per_day,
                                                        path=figure_dir / 'calendar_overview.png')


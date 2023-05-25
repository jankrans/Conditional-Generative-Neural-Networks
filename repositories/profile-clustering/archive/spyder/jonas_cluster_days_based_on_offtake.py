# %% imports
from pathlib import Path

import pandas as pd

from energyclustering import data as data
import archive.visualisation.visualise_ts_clustering as clustering_vis
# %%
from archive.clustering import cluster_timeseries_k_mediods_DTW

measurements_per_day = data.get_timeseries_per_day('Offtake', only_single_meters=True)
master_table = data.get_master_table()
print(measurements_per_day.index.names)

# %%
all_ids = measurements_per_day.reset_index()['iID'].unique()
# choose one of these
id_to_investigate = all_ids[5]
# id_to_investigate = 's53aMdSoUJNFuQ'

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
n_clusters = 5
cluster_dict = cluster_timeseries_k_mediods_DTW(series, n_clusters, 6, 4)
# cluster_dict = mark_outliers(cluster_dict, min_size=5)
# %%
figure_dir = Path() / "figures" / "initial_clustering_offtake" / str(id_to_investigate).replace("/", "")
figure_dir.mkdir(parents=True, exist_ok=True)
clustering_vis.visualise_ts_clusters_per_day(cluster_dict, df_per_day, figure_dir / 'clustering.png')
clustering_vis.show_daily_cluster_indices_over_year(cluster_dict, df_per_day, path=figure_dir / "yearly_overview.png")
clustering_vis.show_daily_cluster_indices_over_calendar(cluster_dict, df_per_day,
                                                        path=figure_dir / 'calendar_overview.png')

# %% imports
from pathlib import Path

import pandas as pd

from energyclustering import data as data
import archive.visualisation.visualise_ts_clustering as clustering_vis
# %%
from archive.clustering import mark_outliers, cluster_timeseries_k_mediods_DTW, \
    sort_clusters_based_on_average_total_consumption

measurements_per_day = data.get_timeseries_per_day('Injection')
master_table = data.get_master_table()
FIGURE_DIR = Path() / "figures" / "Injection clustering over households"
FIGURE_DIR.mkdir(exist_ok=True, parents = True)
# %%
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]

clusterings = []
# choose one or just loop over several
# for id_to_investigate in ids_with_production:
all_data = []
for id_to_investigate in ids_with_production:
    print(f"investigating: {id_to_investigate}")
    df_per_day: pd.DataFrame = measurements_per_day.loc[id_to_investigate]
    all_data.append(df_per_day)
    print("for now we just ignore missing data")
    df_per_day.dropna(inplace=True, axis=0)
    series = df_per_day.to_numpy()
    n_clusters = 4
    cluster_dict = cluster_timeseries_k_mediods_DTW(series, n_clusters, 6, 4)
    cluster_dict = mark_outliers(cluster_dict, min_size=5)
    sorted_cluster_list = sort_clusters_based_on_average_total_consumption(cluster_dict, df_per_day)
    to_dict = {idx:item for idx,item in enumerate(sorted_cluster_list)}
    clustering_vis.show_daily_cluster_indices_over_calendar(to_dict, df_per_day,
                                                            path=FIGURE_DIR / f"{id_to_investigate.replace('/', '')}_overview2.png")
    clustering_vis.show_daily_cluster_indices_over_calendar(cluster_dict, df_per_day,
                                                            path=FIGURE_DIR/ f"{id_to_investigate.replace('/','')}_overview.png")
    clusterings.append(to_dict)
clustering_vis.compare_daily_cluster_indices_over_year(clusterings, all_data, ids_with_production, FIGURE_DIR/'first_try.png')
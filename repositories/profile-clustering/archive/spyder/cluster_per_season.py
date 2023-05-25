#!/usr/bin/env python
from datetime import date, datetime
from pathlib import Path

from energyclustering import data as data
import archive.clustering as clust
import archive.visualisation.visualise_ts_clustering as vis_clust
import pandas as pd
import numpy as np
Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

FIGURE_DIR = Path(__file__).parent / 'figures'/'clustering_per_season'
def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)

master_table = data.get_master_table()
ids_without_production = master_table[(master_table['Lokale productie'] == 'Nee')].loc[:,"InstallatieID"].unique()
df:pd.DataFrame = data.get_timeseries_per_day("Consumption")
df = df[df.index.isin(ids_without_production, level = 0)].dropna()
ids_to_investigate = df.index.get_level_values(0).unique()

def investigate_id(id, euclidean):
    print(f"investigation: {id}")
    figure_path = FIGURE_DIR / id.replace('/', '')
    house_df = df.loc[id]
    nb_of_clusters = 6
    if euclidean:
        full_clustering = clust.cluster_timeseries_k_mediods_euclidean(house_df.to_numpy(), nb_of_clusters)
    else:
        full_clustering = clust.cluster_timeseries_k_mediods_DTW(house_df.to_numpy(), nb_of_clusters, 4, 4)

    vis_clust.visualise_ts_clusters_per_day(full_clustering, house_df, figure_path / f"all.png")
    vis_clust.show_daily_cluster_indices_over_calendar(full_clustering, house_df, figure_path / f'all_calendar.png')

    # add season column
    seasons = np.array([get_season(date) for date in house_df.index])
    for season in np.unique(seasons):
        # only days of a certain season
        season_df = house_df[seasons == season]
        # play with this value
        nb_of_clusters = 8
        if euclidean:
            clustering = clust.cluster_timeseries_k_mediods_euclidean(season_df.to_numpy(), nb_of_clusters)
        else:
            clustering = clust.cluster_timeseries_k_mediods_DTW(season_df.to_numpy(), nb_of_clusters, 4, 4)

        figure_path.mkdir(parents=True, exist_ok = True)
        vis_clust.visualise_ts_clusters_per_day(clustering, season_df,figure_path / f"{season},{'euc' if euclidean else 'DTW'}.png")
        vis_clust.show_daily_cluster_indices_over_calendar(clustering, season_df,figure_path /f"{season},{'euc' if euclidean else 'DTW'}_calendar.png" )

# and with this value
investigate_id(ids_to_investigate[21],False)
investigate_id(ids_to_investigate[21],True)
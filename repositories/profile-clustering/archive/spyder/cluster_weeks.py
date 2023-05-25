# So basicly this notebook is just to illustrate that clustering based on the weeks is probably NOT going to work
from pathlib import Path

from energyclustering import data as data
import archive.visualisation.visualise_ts_clustering as clust_vis
from archive.clustering import cluster_timeseries_k_mediods_DTW
from archive.clustering import determine_nb_of_clusters_silhouette

FIGURE_DIR = Path()/ "figures" / "Weekly clustering"

FIGURE_DIR2 = Path()/ "figures" / "Weekly clustering silhouette"
weekly_df = data.get_timeseries_per_week("Consumption")
ids_to_investigate = data.get_ids(production = False, only_single_meters=True)


def do_weekly_clustering(id, n_clusters):
    print(f"investigation {id}")
    df_per_week = (
        # get information of the correct id
        weekly_df.loc[id]
        # drop all weeks that have missing data
        .dropna(axis= 0)
    )
    series = df_per_week.to_numpy()
    cluster_dict = cluster_timeseries_k_mediods_DTW(series, n_clusters, 6, 4)
    figure_dir = FIGURE_DIR / str(id).replace('/','')
    figure_dir.mkdir(parents=True, exist_ok= True)

    clust_vis.visualise_ts_clusters_per_week(cluster_dict, df_per_week, figure_dir / 'clustering.png')

def do_weekly_clustering_determine_k(id):
    print(f"investigation {id}")
    df_per_week = (
        # get information of the correct id
        weekly_df.loc[id]
        # drop all weeks that have missing data
        .dropna(axis= 0)
    )
    series = df_per_week.to_numpy()
    nb_of_clusters, cluster_dict = determine_nb_of_clusters_silhouette(series, 5, 20, 6, 4)
    figure_dir = FIGURE_DIR2 / str(id).replace('/','')
    figure_dir.mkdir(parents=True, exist_ok= True)

    clust_vis.visualise_ts_clusters_per_week(cluster_dict, df_per_week, figure_dir / 'clustering.png')


do_weekly_clustering_determine_k(ids_to_investigate[3])
from pathlib import Path

from energyclustering.data import get_timeseries_per_day, get_ids
from archive.clustering import determine_nb_of_clusters_elbow
from archive.datetime_util import get_season, get_day_type
from archive.visualisation.visualise_query import plot_daily_df_with_cluster_size
from archive.visualisation import visualise_ts_clusters_per_day
import altair as alt


def summarize_profile(id_to_investigate, path = None):
    # get daily df of id_to_investigate and drop N/A's
    daily_df = (
        get_timeseries_per_day("Consumption")
        .loc[id_to_investigate]
        .dropna(axis =1)
        .reset_index()
    )
    # get season and day_type based on the date
    daily_df['season'] = daily_df['date'].apply(get_season)
    daily_df['day_type'] = daily_df['date'].apply(get_day_type)

    # all seasons
    seasons = daily_df['season'].unique()

    charts = []
    # for each season
    for season in seasons:
        season_df = daily_df[daily_df['season'] == season]
        # for each day_type
        for day_type in ['weekday', 'weekend']:
            season_daytype_df = (
                season_df[season_df['day_type']==day_type]
                .drop(labels = ['season', 'day_type'], axis = 1)
                .set_index(keys = 'date')
            )
            best_k, cluster_dict, medoid_indices = determine_nb_of_clusters_elbow(season_daytype_df.to_numpy(), 4, 20, 10, 4, return_medoids=True, znormalise=False)
            medoids_df = season_daytype_df.iloc[medoid_indices]
            medoids_df['medoid_index'] = medoid_indices
            medoids_df['cluster_size'] = [len(cluster_dict[idx]) for idx in range(len(medoid_indices))]
            medoids_df['relative_cluster_size'] = medoids_df['cluster_size']/ len(season_daytype_df)
            if path is not None:
                visualise_ts_clusters_per_day(cluster_dict, season_daytype_df, path/f"{season},{day_type}.png")
            df = (
                medoids_df
                    .reset_index()
                    .set_index(['date', 'medoid_index', 'cluster_size', 'relative_cluster_size'])
                    # stack makes columns a row
                    .stack()
                    .reset_index()
                    .rename(columns={0: 'value'})
            )
            chart = plot_daily_df_with_cluster_size(df)
            chart = chart.properties(title = f'{season},{day_type}')
            charts.append(chart)
    result_chart = alt.hconcat(*charts).resolve_scale(color='independent', y = 'shared')
    # altair_viewer.display(result_chart)
    return result_chart


if __name__ == '__main__':
    ids = get_ids(production=False, only_single_meters=True)
    for i in range(5):
        try:
            id_to_investigate = ids[i]
            path = Path().absolute().parent.parent / 'figures' / 'daily_cluster_medoids' / id_to_investigate.replace("/", "")
            file_path = path / 'test.png'
            print(id_to_investigate)
            chart = summarize_profile(id_to_investigate, path = None)
            # path.parent.mkdir(parents=True, exist_ok=True)
            # chart.save(str(path))
            print(i)
        except:
            print("error")


import itertools
from math import ceil

import plotly.colors as colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_clustering(labels, data_df, max_shown_instances=5, max_shown_clusters=10, type = 'daily'):
    plot_function = PLOT_TYPES[type]
    # determine the number of instances to show per cluster
    max_nb_instances_per_cluster = labels.groupby('label').size().max()
    nb_instances_per_cluster_to_show = int(min(max_shown_instances, max_nb_instances_per_cluster))

    # determine the number of clusters to show
    nb_clusters = labels.label.max() + 1
    nb_clusters_to_show = int(min(max_shown_clusters, nb_clusters))

    # figure
    fig = make_subplots(
        rows=nb_instances_per_cluster_to_show,
        cols=nb_clusters_to_show,
        #             subplot_titles = (str(left_profile.name), str(right_profile.name)),
        shared_yaxes='all',
        shared_xaxes='all',
        horizontal_spacing= 0.01,
        vertical_spacing= 0.01,
        column_titles= [f'Cluster {i}' for i in range(0, nb_clusters_to_show )],

    )
    fig.update_xaxes(
        dtick=7200000,
        tickformat = "%H:00",
    )
    fig.update_layout(
        showlegend=False,
        )

    # for each cluster to show
    for cluster_idx, instances in sorted(labels.groupby('label')):
        if cluster_idx >= nb_clusters_to_show:
            continue
            # plot
        for profile_idx, profile in enumerate(instances.index):
            if profile_idx >= nb_instances_per_cluster_to_show:
                continue
            daily_df = series_to_daily_df(data_df.loc[profile])
            plot_function(fig, profile_idx, int(cluster_idx), daily_df.dropna(axis=0))
    return fig

def plot_profiles(data_df, columns = 1, type = 'daily', sharey = None):
    plot_function = PLOT_TYPES[type]
    number_of_profiles = data_df.index.nunique()
    rows = ceil(number_of_profiles / columns)

    if type == 'heatmap':
        shared_yaxis = 'all'
    else:
        shared_yaxis = False
    if sharey is not None and sharey == True:
        shared_yaxis = 'all'

    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_yaxes=shared_yaxis,
        shared_xaxes='all',
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    # print(f"{rows}x{columns}")
    fig.update_xaxes(
        dtick=7200000,
        tickformat="%H:00",
    )
    fig.update_layout(
        showlegend=False,
    )
    coordinate_iterator = itertools.product(range(0, rows ), range(0, columns ))
    for profile in data_df.index:
        row, column = next(coordinate_iterator)
        # print(f'{row}x{column}')
        daily_df = series_to_daily_df(data_df.loc[profile])
        plot_function(fig, row, column, daily_df.dropna(axis=0))
    return fig






def add_daily_plot_to_figure(fig, row_idx, col_idx, daily_df):
    for name, row in daily_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=daily_df.columns.values,
                y=row.values,
                name=row.name,
                opacity=0.3,
                line=dict(width=0.5),
                marker_colorscale=colors.sequential.Rainbow
            ),
            row=row_idx + 1,
            col=col_idx + 1,
        )


def add_heatmap_plot_to_figure(fig, row_idx, col_idx, daily_df):
    fig.add_trace(
        go.Heatmap(
            z=daily_df.to_numpy(),
            x=daily_df.columns,
            y=daily_df.index,
            coloraxis = 'coloraxis'
        ),
        row = row_idx + 1,
        col = col_idx + 1
    )
    fig.update_layout(coloraxis={'colorscale': 'turbo'})
    fig.update_yaxes(autorange="reversed")

def series_to_daily_df(profile, dropna = True):
    daily_df = (
        profile.to_frame('value')
        .assign(
            time=lambda x: add_date(x.index.time),
            date=lambda x: x.index.date.astype('str')
        )
        .pipe(lambda x: pd.pivot_table(x, index='date', columns='time', values='value', dropna= False))
    )
    if dropna:
        daily_df = daily_df.dropna(axis = 0)
    return daily_df

def add_date(series):
    return pd.to_datetime(series, format='%H:%M:%S', exact = False)


PLOT_TYPES = {
    'daily': add_daily_plot_to_figure,
    'heatmap': add_heatmap_plot_to_figure
}
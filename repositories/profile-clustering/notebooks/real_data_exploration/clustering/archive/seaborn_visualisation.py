import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from util import add_date
def plot_clustering(labels, data_df, plot_function): 
    # determine the number of plots 
    max_shown_instances = 10
    instances_per_cluster = labels.value_counts()
    nb_of_clusters = len(instances_per_cluster)
    max_instances = instances_per_cluster.max()
    shown_instances = min(max_instances, max_shown_instances)

    # size of each subplot 
    single_fig_size = (7,7)

    # way to long matplotlib call 
    fig, ax = plt.subplots(nrows = shown_instances, ncols = nb_of_clusters, figsize = (nb_of_clusters*single_fig_size[1],shown_instances*single_fig_size[0]))

    # for each cluster and each instance in a cluster plot 
    for cluster in sorted(labels.labels.unique()): 
        instances_in_cluster = labels[labels.labels == cluster].iloc[:shown_instances].index
        for row_idx, meterId in enumerate(instances_in_cluster): 
            plot_function(meterId, data_df, ax = ax[row_idx, cluster])
def energy_heatmap(meterID, data_df, ax = None):
    min_value = 0
    max_value = data_df.max().max()
    subset = (
        data_df.loc[[meterID],:]
        .droplevel(level= 1, axis = 0)
        .stack().to_frame('value')
        .reset_index()
        .assign(
            time = lambda x: x.timestamp.dt.time,
            date = lambda x: x.timestamp.dt.date,
        )
        .pipe(lambda x: pd.pivot_table(x, index = 'date', columns = 'time', values = 'value'))
        .dropna(axis = 0)
    )
    return sns.heatmap(
        data = subset,
        cmap = 'viridis',
        vmin = min_value, 
        vmax = max_value, 
        ax = ax
    )

def day_plot(meterID, data_df, ax = None): 
    subset = (
        data_df.loc[[meterID],:]
        .droplevel(level= 1, axis = 0)
        .stack().to_frame('value')
        .reset_index()
        .assign(
            time = lambda x: add_date(x.timestamp.dt.time),
            date = lambda x: x.timestamp.dt.date,
        )
    )
    return sns.lineplot(
        data = subset,
        x = 'time', 
        y = 'value', 
        hue = 'date',
        size = 0.1,

    #     kind = 'line', 
        legend = False,
        ax = ax, 
        alpha = 0.2,
    )
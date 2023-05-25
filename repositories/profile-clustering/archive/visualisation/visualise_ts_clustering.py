import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from archive.clustering import cluster_labeling_to_dict

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DAYS = ("Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun")


def show_daily_cluster_indices_over_calendar(cluster_dict, data, path=None):
    dates = data.index
    # list of tuples month, week_nb, weekday_nb, cluster_idx
    dataframe_data = []

    for cluster_label, instances in cluster_dict.items():

        for instance in instances:

            date = dates[instance]
            year = date.year
            month_id = date.month - 1
            month = MONTHS[date.month - 1]
            day_id = date.weekday()
            day = DAYS[date.weekday()]
            day_of_month = date.day
            week_nb = date.isocalendar()[1]
            # fix because first week of januari is the 53'th week in ISO format
            if week_nb == 53:
                week_nb = 0
            dataframe_data.append(
                (year, month, month_id, week_nb, day, day_id, day_of_month, cluster_label))
    df = pd.DataFrame(data=dataframe_data,
                      columns=['year', 'month', 'month_id', 'week', 'day', 'day_id', 'day_of_month', 'cluster_label'])

    chart = alt.Chart(df).mark_rect().encode(
        alt.X('day:O', sort=alt.EncodingSortField('day_id')),
        alt.Y('week:O'),
        alt.Color('cluster_label:N')
    ).facet(
        alt.Facet('month:O', sort=alt.EncodingSortField('month_id')),
        columns=3
    ).resolve_scale(y='independent',
                    x='independent')
    chart.save(str(path))


def show_daily_cluster_indices_over_year(cluster_dict, data, path=None):
    dates = data.index
    plot_data = np.zeros((len(cluster_dict), len(data)))
    for cluster_idx, instances in cluster_dict.items():
        for instance in instances:
            if cluster_idx == 'outlier':
                cluster_idx = len(plot_data) - 1
            plot_data[cluster_idx, instance] += 1

    fig, ax = plt.subplots()
    legend_labels = [f"cluster{i}" for i in range(len(cluster_dict) - 1)] + ["unique patterns"]
    for cluster_idx in range(plot_data.shape[0]):
        ax.bar(dates, plot_data[cluster_idx, :], width=1, label=legend_labels[cluster_idx])

    # ax.stackplot(dates, plot_data, labels = )
    ax.legend(loc='upper left')
    if path is not None:
        plt.savefig(path)
    plt.show()

def visualise_ts_group(data):
    stacked_df = data.stack()

def visualise_ts_cluster_per_day_altair_interactive(cluster_dict, data):
    charts = []
    for idx, cluster in cluster_dict.items():
        single_nearest = alt.selection_single(on='mouseover', nearest=True)
        single_day = data.iloc[cluster].reset_index()
        single_day.columns = ['time', 'value']
        random_date = datetime.date(2000, 1, 1)
        single_day['time'] = [datetime.datetime.combine(random_date, time) for time in single_day['time']]
        chart = alt.Chart(single_day).mark_line().encode(
            x = 'time:T',
            y = 'value:Q',
            color = alt.condition(single_nearest, 'instance_id','gray')
        ).add_selection(single_nearest)
        layers.append(chart)
        layered_chart = alt.layer(*layers)
        charts.append(layered_chart)
    chart = alt.hconcat(*charts)
    import altair_viewer
    altair_viewer.display(chart)

def compare_daily_cluster_indices_over_year(sorted_clusterings_list, all_data, ids, path):
    # tuples of form id, date, cluster_idx
    data = []
    for sorted_clustering, id, df_per_day in zip(sorted_clusterings_list,ids, all_data):
        dates = df_per_day.reset_index()['date']
        for idx, cluster in sorted_clustering.items():
            for instance in cluster:
                data.append((id, MONTHS[dates[instance].month-1],dates[instance].month, dates[instance].day, idx))
    df = pd.DataFrame(data, columns = ['installation', 'month', 'month_idx', 'day','cluster_label'])
    chart = alt.Chart(df).mark_rect().encode(
        alt.X('day:O'),
        alt.Y('installation:N'),
        alt.Color('cluster_label:N',scale = alt.Scale(scheme = 'plasma'))
    ).facet(
        alt.Facet('month:O', sort = alt.EncodingSortField('month_idx')),
        columns = 3
    ).resolve_axis(x ='independent')
    chart.save(str(path))

def show_time_serie(timeseries, idx, ax_label = None, path = None):
    fig, ax = plt.subplots()
    random_date = datetime.datetime(2000, 8, 1)
    dates = [random_date + datetime.timedelta(hours=time.hour, minutes=time.minute) for time in timeseries.columns]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.plot(dates, timeseries.iloc[idx].values)
    if ax_label is None:
        ax.set_ylabel("kWh")
    else:
        ax.set_ylabel(f"{ax_label} (kWh)")
    ax.set_xlabel("time of day")
    ax.set_title(timeseries.index[idx])
    ax.yaxis.set_tick_params(labelleft=True)
    fig.autofmt_xdate()
    if path is not None:
        plt.savefig(path)
    plt.show()

def visualise_ts_clusters_per_week(cluster_dict, week_data, path=None):
    nb_of_clusters = len(cluster_dict)
    # size of a single plot (width, height)
    single_plot_size = (20, 5)
    fig, axes = plt.subplots(nb_of_clusters, 1, sharex=True, sharey=True,
                             figsize=(single_plot_size[0], single_plot_size[1] * nb_of_clusters))
    # make the axes one dimensional
    axes = np.array(axes).reshape((-1,))
    for (ax, (idx, cluster)) in zip(axes, sorted(list(cluster_dict.items()))):
        dates = [datetime.datetime.fromisocalendar(2000,1,day+1) + datetime.timedelta(hours=time.hour, minutes=time.minute) for day,time in week_data.columns]
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%A %H:%M"))
        for idx in cluster:
            ax.plot(dates, week_data.iloc[idx].values, label=f"week: {week_data.index[idx]}")
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylabel("kWh")
        ax.set_xlabel("time")
        ax.set_title(f"cluster{idx} #members {len(cluster)}")

        # ax.set_xticks(rotation=45)
        if len(cluster) <= 10:
            ax.legend()
    fig.autofmt_xdate()
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_visible(True)
    if path is not None:
        plt.savefig(path)
    plt.show()

def visualise_ts_clusters_per_day(cluster_dict, data, path = None):
    """
        For each cluster makes a subplot and plots all timeseries from that cluster on the subplot
    """

    nb_of_clusters = len(cluster_dict)
    # size of a single plot (width, height)
    single_plot_size = (10, 5)
    fig, axes = plt.subplots(nb_of_clusters, 1, sharex=False, sharey=True,
                             figsize=(single_plot_size[0], single_plot_size[1] * nb_of_clusters))
    # make the axes one dimensional
    axes = np.array(axes).reshape((-1,))
    for (ax, (idx,cluster)) in zip(axes, sorted(list(cluster_dict.items()))):
        plot_cluster_to_axis(ax, cluster, idx,data)
    fig.autofmt_xdate()
    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_cluster_to_axis(ax, cluster, cluster_idx,  data):
    # ok so matplotlib can only work with datetime objects not with time objects --'
    # so use a random date and only show time labels
    random_date = datetime.datetime(2000, 8, 1)
    dates = [random_date + datetime.timedelta(hours = time.hour, minutes = time.minute) for time in data.columns]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for idx in cluster:
        ax.plot(dates, data.iloc[idx].values, label=f"{data.index[idx].weekday()} {data.index[idx]}")
    ax.yaxis.set_tick_params(labelleft=True)
    ax.set_ylabel("kWh")
    ax.set_xlabel("time")
    ax.set_title(f"cluster{cluster_idx} #members {len(cluster)}")


    # ax.set_xticks(rotation=45)
    if len(cluster)<=10:
        ax.legend()

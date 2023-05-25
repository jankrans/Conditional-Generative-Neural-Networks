import altair as alt 
import pandas as pd 
from util import add_date


def show_clustering(data_df, labels, type = 'all_days', max_shown_instances = 10): 
    assert type in TYPES, f'unknown type {type}'
    plot_function = TYPES[type]
    
    charts = [] 
    for cluster in sorted(labels.labels.unique()): 
        instances_in_cluster = labels[labels.labels == cluster].iloc[:max_shown_instances].index       
        column_charts = []
        for meterID in instances_in_cluster: 
            chart = plot_function(meterID, data_df)
            column_charts.append(chart)
        column = alt.vconcat(*column_charts).resolve_scale(color = 'shared', y = 'shared')
        charts.append(column)
    return alt.hconcat(*charts).resolve_scale(color = 'shared', y = 'shared')
        
def energy_heatmap_chart(meterID, data_df): 
    subset = (
        data_df.loc[[meterID],:]
        .droplevel(level = 1, axis =0)
        .stack().to_frame('value')
        .reset_index()
        .assign(
            time = lambda x: add_date(x.timestamp.dt.time),
            date = lambda x: x.timestamp.dt.date.astype('str'),
        )
    )
    return alt.Chart(subset, height = 1000, width = 1000).mark_rect(strokeOpacity = 0).encode(
        x = alt.X('time:O', axis = alt.Axis(labels = False, grid = False)),
        y = alt.Y('date:O', axis = alt.Axis(labels = False, grid = False)), 
        color = alt.Color('value:Q', scale = alt.Scale(scheme = 'viridis'))
    )

def all_day_plot(meterID, data_df): 
    subset = (
            data_df.loc[[meterID],:]
            .droplevel(level = 1, axis =0)
            .stack().to_frame('value')
            .reset_index()
            .assign(
                time = lambda x: add_date(x.timestamp.dt.time),
                date = lambda x: x.timestamp.dt.date.astype('str'),
            )
        )
    return alt.Chart(subset).mark_line(strokeWidth = 0.2, strokeOpacity = 0.2).encode(
        x = alt.X('time:T', axis = alt.Axis(format = '%H:%M')),
        y = alt.Y('value:Q'), 
        color = alt.Color('date', legend = None, scale = alt.Scale(scheme = 'rainbow'))
    )

TYPES = dict(
    all_days = all_day_plot, 
    heatmap = energy_heatmap_chart, 
)
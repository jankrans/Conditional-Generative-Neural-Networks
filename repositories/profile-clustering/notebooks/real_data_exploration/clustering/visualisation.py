import pandas as pd 
import altair as alt
from util import add_date
def anomaly_detection_chart(profile_matrix, anomaly_labels): 
    profile_vis_cluster = profile_matrix.stack().to_frame('value').join(anomaly_labels).reset_index()
    profile_vis_cluster.time = add_date(profile_vis_cluster.time)
    return alt.Chart(profile_vis_cluster.reset_index()).mark_line().encode(
        x = 'time:T', 
        y = 'value', 
        color = 'date'
    ).facet(row = 'anomaly')

def show_profiles(profile_matrix): 
    return all_day_chart(profile_matrix.stack().to_frame('value').reset_index()).facet(facet = 'meterID', columns = 5).resolve_scale(color = 'independent').resolve_axis(x='independent', y='independent')
    
def all_day_chart(profile_matrix): 
    all_day_chart = alt.Chart(profile_matrix, title = 'All days').mark_line(strokeWidth = 0.2, strokeOpacity = 0.2).encode(
        x = 'time:T',
        y = 'value', 
        color = alt.Color('date' ,legend = None,  scale = alt.Scale(scheme = 'rainbow'))
    )
    return all_day_chart

def daily_clustering_chart(data_df, labels):
    vis_df = (
        data_df
        .rename_axis(columns = 'time')
        .stack()
        .to_frame('value')
        .join(labels.astype('int'))
        .reset_index()
        .assign(
            time = lambda x: add_date(x.time),
            color = lambda x: x.date + x.meterID
        )
    )
    return alt.Chart(vis_df).mark_line(strokeWidth = 0.3).encode(
        x = 'time:T', 
        y = 'value', 
        color = alt.Color('color',legend = None,  scale = alt.Scale(scheme = 'rainbow'))
    ).facet(facet = 'labels:N', columns = 5) 

import altair as alt

from archive.datetime_util import to_datetime


def plot_daily_df_with_cluster_size(daily_df):
    df = daily_df.reset_index()
    print(df.head())
    df['date'] = df['date'].apply(to_datetime)
    df['time'] = df['time'].apply(to_datetime)

    SMALL_CLUSTER_THRESHOLD = 2
    selector = alt.selection_multi(empty='all', nearest=True, fields=['date'])
    # selector = alt.selection_interval(fields = ['date'])

    # this is just a selection layer ignore it ;)
    selection_chart = alt.Chart(df).mark_point(color='black', size=1).encode(
        x='time:T',
        y='value:Q',
        opacity=alt.value(0)
    ).add_selection(
        selector
    )

    chart1 = alt.Chart(df[df['cluster_size'] <= SMALL_CLUSTER_THRESHOLD]).mark_line().encode(
        x = alt.X('time:T'),
        y = alt.Y('value:Q'),
        color = alt.Color('date:N', legend = None),
        # color = alt.Color('date:N'),
        strokeWidth = alt.StrokeWidthValue(0.8),
        opacity = alt.OpacityValue(0.5)
    )
    chart2 = alt.Chart(df[df['cluster_size'] > SMALL_CLUSTER_THRESHOLD]).mark_line().encode(
        x = alt.X('time:T'),
        y = alt.Y('value:Q'),
        color = alt.Color('date:N', legend = None),
        strokeWidth = alt.StrokeWidth('relative_cluster_size:Q', scale= alt.Scale(range = [1,5])),
        opacity = alt.condition(selector, alt.value(1), alt.value(0.4))

    )
    return alt.layer(selection_chart, chart1, chart2)
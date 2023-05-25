import datetime

from energyclustering import data as data
import altair as alt
import altair_viewer

## this is just a small test to see if interactive plots might be usefull
df = data.get_timeseries_per_day("Consumption")
all_ids = df.index.get_level_values(0)
one_df = df.loc[all_ids[0]]

stacked_df = one_df.stack().reset_index().rename(columns={0:'value'}).iloc[:1000]
random_date = datetime.date(2000, 1, 1)
stacked_df['time'] = [datetime.datetime.combine(random_date, time) for time in stacked_df['time']]
stacked_df['date'] = [datetime.datetime.combine(date, datetime.datetime.min.time()) for date in stacked_df['date']]
selector = alt.selection_multi(empty ='all',nearest = True, fields = ['date'])
# selector = alt.selection_interval(fields = ['date'])
chart1 = alt.Chart(stacked_df).mark_point(color='black', size=1).encode(
    x='time:T',
    y='value:Q',
    opacity = alt.value(0)
).add_selection(
    selector
)
chart = alt.Chart(stacked_df).mark_line().encode(
            x = 'time:T',
            y = 'value:Q',
            color = 'date:N',
            opacity = alt.condition(selector, alt.value(1), alt.value(0.4))
        )
highlight_chart = alt.Chart(stacked_df).mark_line().encode(
            x = 'time:T',
            y = 'value:Q',
            color = 'date:N',
            opacity = alt.condition(selector, alt.value(1), alt.value(0))
        )

altair_viewer.display(alt.layer(chart,highlight_chart,chart1).interactive())
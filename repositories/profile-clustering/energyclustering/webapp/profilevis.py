import dash_core_components as dcc
import dash_html_components as html
import plotly.colors as colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from energyclustering.util import series_to_daily_dataframe


class SimpleProfileVisualiser:
    def __init__(self, name, data):
        self.name = name
        self.left_profile_idx = None
        self.right_profile_idx = None

    def get_visualize_code(self, app):
        return html.Table([
            html.Tr([
                html.Td([
                    html.H1(self.name),
                    html.P(children=f'profile_index = {self.left_profile_idx}\n')
                ]),
                html.Td([
                    html.H1(self.name),
                    html.P(children=f'profile_index = {self.right_profile_idx}\n')
                ]),

            ]),


        ])

    def update_profile_ids(self, left_profile_idx, right_profile_idx, app ):
        self.left_profile_idx = left_profile_idx
        self.right_profile_idx = right_profile_idx
        return self.get_visualize_code(None)


class HeatmapProfileVisualizer:
    def __init__(self, name, data):
        self.name = name
        self.left_profile_idx = None
        self.right_profile_idx = None
        self.data = data

    def update_profile_ids(self, left_profile_idx, right_profile_idx, app):
        self.left_profile_idx = left_profile_idx
        self.right_profile_idx = right_profile_idx

        return self.get_visualize_code(None)

    def get_visualize_code(self, app):
        figure = self.get_figcode(self.left_profile_idx, self.right_profile_idx)
        div_children = [
            dcc.Graph(id=f'{self.name}_graph', figure=figure),
            ]

        layout = html.Div(div_children ,id= f'{self.name}_placeholder')

        return layout

    def get_figcode(self, left_profile_idx, right_profile_idx):
        left_profile = self.data.get_profile_series(left_profile_idx)
        right_profile = self.data.get_profile_series(right_profile_idx)
        min_value = min(0, left_profile.min(), right_profile.min())
        max_value = max(left_profile.max(), right_profile.max())
        left_day_matrix = series_to_daily_dataframe(left_profile, dropna=False)
        right_day_matrix = series_to_daily_dataframe(right_profile, dropna=False)

        fig = make_subplots(
            rows = 1,
            cols = 2,
            subplot_titles = (str(left_profile.name), str(right_profile.name)),

        )
        data = go.Heatmap(
            z=left_day_matrix.to_numpy(),
            x=left_day_matrix.columns,
            y=left_day_matrix.index,
            zmin = min_value,
            zmax = max_value,
            colorbar={"title": 'consumption in kW'},
        )
        fig.add_trace(
            data,
            row = 1,
            col = 1,
            # title = f"Offtake (in kW) of profile{left_profile_idx}"
        )

        data2 = go.Heatmap(
            z=right_day_matrix.to_numpy(),
            x=right_day_matrix.columns,
            y=right_day_matrix.index,
            zmin=min_value,
            zmax=max_value,
            colorbar={"title": 'consumption in kW'},
        )
        fig.add_trace(
            data2,
            row=1,
            col=2,
            # title=f"Offtake (in kW) of profile{right_profile_idx}"
        )


        # raw_df = self.data.get_profile_series(profile_idx)
        # profile_id = raw_df.index.get_level_values(0).drop_duplicates()[profile_idx]
        # daily_df = (
        #     raw_df
        #         .loc[profile_id]
        #         .dropna(axis=1)
        # )
        # fig = px.imshow(daily_df, title = f"Offtake (in kW) of profile {profile_id}")
        fig.update_layout(title_x = 0.5)
        # fig.update_layout(transition_duration=500)
        fig.layout.width = 1600
        fig.layout.height = 800
        return fig

class DayProfileVisualizer:
    def __init__(self, name, data):
        self.name = name
        self.left_profile_idx = None
        self.right_profile_idx = None
        self.data = data
        self.vis_code = None

    def update_profile_ids(self, left_profile_idx, right_profile_idx, app):
        if self.vis_code is not None and self.left_profile_idx == left_profile_idx and self.right_profile_idx == right_profile_idx:
            return self.vis_code
        self.left_profile_idx = left_profile_idx
        self.right_profile_idx = right_profile_idx
        new_code = self.get_visualize_code(None)
        self.vis_code = new_code
        return self.vis_code

    def get_visualize_code(self, app):
        figure = self.get_figcode(self.left_profile_idx, self.right_profile_idx)
        div_children = [
            dcc.Graph(id=f'{self.name}_graph', figure=figure),
            ]

        layout = html.Div(div_children ,id= f'{self.name}_placeholder')

        return layout

    def get_figcode(self, left_profile_idx, right_profile_idx):
        left_profile = self.data.get_profile_series(left_profile_idx)
        right_profile = self.data.get_profile_series(right_profile_idx)
        left_day_matrix = series_to_daily_dataframe(left_profile, dropna=False)
        right_day_matrix = series_to_daily_dataframe(right_profile, dropna=False)

        fig = make_subplots(
            rows = 1,
            cols = 2,
            subplot_titles = (str(left_profile.name), str(right_profile.name)),
            shared_yaxes=True,
            shared_xaxes=True,

        )
        for col, matrix in zip([1,2], [left_day_matrix, right_day_matrix]):
            for idx, row in matrix.iterrows():
                data = go.Scatter(
                    x = left_day_matrix.columns,
                    y = row,
                    name = row.name,
                    opacity = 0.3,
                    line = dict(width = 0.5),
                    marker_colorscale=colors.sequential.Rainbow,
                )
                fig.add_trace(
                    data,
                    row = 1,
                    col = col,
                    # title = f"Offtake (in kW) of profile{left_profile_idx}"
                )
        fig.update_layout(
            title_x = 0.5,
            showlegend=False,
            xaxis_title="Time of day",
            yaxis_title="Consumption (in kW)",
        )
        # fig.update_layout(transition_duration=500)
        fig.layout.width = 1600
        fig.layout.height = 800
        return fig

# class ClusteringProfileVisualizer:
#     """ A class that knows how to visualise a certain profile"""
#     def __init__(self,name, data):
#         self.name = name
#         self.data = data
#         self.profile_idx = None
#
#     def get_visualize_code(self,app):
#         """ At the building of the webpage this code is called to build the layout for webpage """
#
#         # At this time I also register a callback that will update the figure a little bit whenever you change a slider
#         # the slider is in this visualisation part
#         # The callback works by specifying where the return result should be placed
#         # and where the input needs to come from
#
#         figure, clustering_failed = self.get_figcode(self.profile_idx)
#         div_children =[
#             html.H1(f'{self.name} placeholder'),
#             html.P(id=f'{self.name}_text', children=f'profile_index ={self.profile_idx}\n'),
#             dcc.Graph(id=f'{self.name}_graph', figure=figure),
#         ]
#         if clustering_failed:
#             div_children.append('\nSome visualisation failed! Showing the available plots')
#         layout = html.Div(div_children, id=f'{self.name}_placeholder')
#         return layout
#
#     def get_figcode(self, profile_idx):
#         # get the data
#         raw_df = self.data.get_profile_series(profile_idx)
#         profile_id = raw_df.index.get_level_values(0).drop_duplicates()[profile_idx]
#         daily_df = (
#             raw_df
#                 .loc[profile_id]
#                 .fillna(value=0)
#         )
#         # weird issue where daily_df is empty
#
#         maximum_value = np.max(daily_df.values)
#
#         daily_df = daily_df.reset_index()
#         fig = make_subplots(rows=4, cols=2, column_width=[200] * 2, row_width=[200] * 4,shared_xaxes=False, shared_yaxes=False, column_titles= ['weekday', 'weekend'], row_titles = ['winter', 'spring', 'summer', 'autumn'])
#         fig.layout.width = 600
#         fig.layout.height = 1000
#         fig.update_layout(showlegend = False)
#         # get season and day_type based on the date
#         daily_df['season'] = daily_df['date'].apply(get_season)
#         daily_df['day_type'] = daily_df['date'].apply(get_day_type)
#
#         # all seasons
#         seasons = daily_df['season'].unique()
#         a_clustering_failed = False
#         for row, season in enumerate(seasons):
#             season_df = daily_df[daily_df['season'] == season].copy()
#             # for each day_type
#             for column, day_type in enumerate(['weekday', 'weekend']):
#                 # take correct subset of the data
#                 season_daytype_df = (
#                     season_df[season_df['day_type'] == day_type].copy()
#                         .drop(labels=['season', 'day_type'], axis=1)
#                         .set_index(keys='date')
#                 )
#                 # cluster the
#                 try:
#                     best_k, cluster_dict, medoid_indices = determine_nb_of_clusters_elbow(season_daytype_df.to_numpy(), 4,
#                                                                                           20, 10, 4, return_medoids=True,
#                                                                                           znormalise=False)
#                 except Exception as e:
#                     a_clustering_failed = True
#                     continue
#
#                 medoids_df = season_daytype_df.iloc[medoid_indices].copy()
#                 medoids_df['medoid_index'] = medoid_indices
#                 medoids_df['cluster_size'] = [len(cluster_dict[idx]) for idx in range(len(medoid_indices))]
#                 medoids_df['relative_cluster_size'] = medoids_df['cluster_size'] / len(season_daytype_df)
#
#                 df: DataFrame = (
#                     medoids_df
#                         .reset_index()
#                         .set_index(['date', 'medoid_index', 'cluster_size', 'relative_cluster_size'])
#                         # stack makes columns a row
#                         .stack()
#                         .reset_index()
#                         .rename(columns={0: 'value'})
#                 )
#
#                 df['date'] = df['date'].apply(to_datetime)
#                 df['time'] = df['time'].apply(to_datetime)
#                 for date in df['date'].drop_duplicates():
#                     subset_df = df[df['date'] == date]
#
#                     fig.add_trace(
#                         go.Scatter(x=subset_df['time'], y=subset_df['value'], mode='lines', line= dict(
#                             width = subset_df.iloc[0].relative_cluster_size*10
#                         )),
#                         row=row + 1, col=column + 1
#                     )
#
#         fig.update_layout(transition_duration=500)
#         fig.update_layout(title_text = f'Representative offtake profiles for profile {profile_id}',title_x=0.5, yaxis_title_text = 'offtake (in kW)')
#         if maximum_value is not None:
#             fig.update_yaxes(range=[0,maximum_value])
#         return fig, a_clustering_failed
#
#     def update_profile_id(self, profile_id, app):
#         self.profile_idx = profile_id
#         return self.get_visualize_code(None)
#
#
#
#
#

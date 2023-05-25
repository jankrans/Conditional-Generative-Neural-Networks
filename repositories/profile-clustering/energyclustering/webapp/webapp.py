from datetime import datetime
from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
import toml
from dash.exceptions import PreventUpdate

from energyclustering.cobras.cobras_kmedoids import COBRAS_kmedoids
from energyclustering.data.fluvius.data import FluviusDataContainer
from energyclustering.webapp.webapp_querier import DummyQuerier


class WebApp:
    def __init__(self, visualizer_factory):
        # read the config information
        # self.data = FluviusDataContainer('random_profiles_v1').read_data() # First dataset and the one Koen used for queries
        self.data = FluviusDataContainer('full_distance_matrix_wasserstein').read_data() # dataset with non data issues
        self.visualizer = visualizer_factory('profile1', self.data)

        self.cobras = None
        self.query_generator = None
        self.app = None


    def submit_query_answer(self, answer):
        try:
            profile1, profile2 = self.query_generator.send(answer)
        except StopIteration:
            print("COBRAS is all DONE!")
            return "ALL DONE"

        # display the new query
        # let the visualisers generate the new necessary code and return it
        new_fig_code = self.visualizer.update_profile_ids(profile1,profile2,self.app)
        return new_fig_code


    def run(self):
        # find the correct paths (ensures uniqueness to not overwrite previous data)
        root_result_path = Path('result')
        log_directory = root_result_path / datetime.now().strftime("%d%b_%H.%M.%S")
        assert not log_directory.exists(), 'this directory should not exist!'
        log_directory.mkdir(parents = True, exist_ok= True)

        log_path = log_directory/'queries.csv'
        print(f'saving results to: {log_path}')

        # set-up COBRAS
        # init cobras
        print("RESTARTING COBRAS")
        self.cobras = COBRAS_kmedoids(self.data.distance_matrix.to_numpy(), DummyQuerier(log_path), 100000000000000000000000, log_path = log_directory, seed = 3132345)
        self.query_generator = self.cobras.cluster()

        # run the app
        app = dash.Dash(__name__)
        self.app = app

        # get the first query to display (prime the generator)
        profile1, profile2= next(self.query_generator)

        # get the app layout
        app.layout = html.Div([
            dcc.Store(id='query_counter', storage_type='memory', data= 0),
            html.H2('0 Queries done', id = 'query_counter_txt', style = {'textAlign': 'center'}),
            html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td([
                            self.visualizer.update_profile_ids(profile1,profile2, app)
                        ]),
                    ], id = 'query_table_row'),
                    html.Tr([
                        html.Td([
                            html.Button(id='must_link_button', children='MUST-LINK', className='button button_mustlink'),
                            html.Button(id='cannot_link_button', children='CANNOT-LINK', className='button button_cannotlink'),
                        ],
                            style = {'textAlign': 'center'},

                        ),
                    ]),
                ]),
            ]),
        ], style = {'margin':'auto'})

        @app.callback(
            dash.dependencies.Output('query_counter_txt', 'children'),
            dash.dependencies.Input('query_counter', 'data')
        )
        def update_query_title(counter):
            return f"{counter} Queries answered"


        @app.callback(
            [dash.dependencies.Output(component_id='query_table_row', component_property='children'),
            dash.dependencies.Output(component_id = 'query_counter', component_property = 'data'),
            ],
            [dash.dependencies.Input(component_id='cannot_link_button', component_property='n_clicks'),
             dash.dependencies.Input(component_id = 'must_link_button',component_property='n_clicks'),
             ],
            dash.dependencies.State('query_counter', 'data')
             )
        def query_callback(cl_n_clicks, ml_n_clicks, query_counter):
            ctx = dash.callback_context

            # if nothing changed don't do anything
            if not ctx.triggered:
                raise PreventUpdate

            # find the pressed button
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'must_link_button':
                value = True
            else:
                value = False
            return self.submit_query_answer(value), query_counter + 1




        app.run_server(debug = False, processes =1)




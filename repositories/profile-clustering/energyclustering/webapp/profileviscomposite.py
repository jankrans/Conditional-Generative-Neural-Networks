import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate


class ProfileVisualizerDropbox:
    def __init__(self, name, names, visualizers, data):
        self.name = name
        self.left_profile_idx = None
        self.right_profile_idx = None

        # manage visualizers
        self.visualizer_names = names
        self.visualizers = [visualizer(name, data) for visualizer in visualizers]
        self.active_visualizer = 0

    def update_profile_ids(self, left_profile_id, right_profile_id, app):
        self.left_profile_idx = left_profile_id
        self.right_profile_idx = right_profile_id


        sublayout = self.visualizers[self.active_visualizer].update_profile_ids(left_profile_id, right_profile_id, app)
        layout = html.Div([
            dcc.Dropdown(
                id=f'{self.name}demo-dropdown',
                options=
                [{'label': name, 'value': index} for index, name in enumerate(self.visualizer_names)],
                value=self.active_visualizer
            ),
            html.Div(sublayout,id = f"{self.name}_profile_switch"),

        ])

        @app.callback(
        dash.dependencies.Output(f'{self.name}_profile_switch', 'children'),
        [dash.dependencies.Input(f'{self.name}demo-dropdown', 'value')])
        def update_output(value):
            ctx = dash.callback_context
            # if nothing changed don't do anything
            if not ctx.triggered:
                raise PreventUpdate
            self.active_visualizer = int(value)
            return self.visualizers[self.active_visualizer].update_profile_ids(self.left_profile_idx, self.right_profile_idx, app)

        return layout


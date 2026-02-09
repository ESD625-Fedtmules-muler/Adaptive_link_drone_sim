import numpy
import plotly.graph_objs as obj_go ##Libs for plotting
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import time

from drone import drone

class simulation:

    def __init__(self):

        pass



drone1 = drone("hans", "testpath1.csv")
drone2 = drone("holger", "happy_path.csv")
drone3 = drone("poul", "happy_path2.csv")


t_start = time.time()

app = Dash(__name__)

app.layout = html.Div(
    style={"height": "100vh"},  # full viewport height
    children=[
        dcc.Graph(
            id="live-graph",
            style={"width": "100%", "height": "90%"}  # fills parent div
        ),
        dcc.Interval(id="interval", interval=200, n_intervals=0)
    ]
)

@app.callback(
    Output("live-graph", "figure"),
    Input("interval", "n_intervals")
)


def update(n):
    t_now = time.time()-t_start
    
    
    
    
    fig = go.Figure()
    fig.add_traces(drone1.render(t_now))
    fig.add_traces(drone2.render(t_now))
    fig.add_traces(drone3.render(t_now))
    fig.update_layout(
        uirevision="keep-camera",
        scene=dict(
            xaxis=dict(range=[-25, 25], autorange=False),
            yaxis=dict(range=[-25, 25], autorange=False),
            zaxis=dict(range=[0, 10], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
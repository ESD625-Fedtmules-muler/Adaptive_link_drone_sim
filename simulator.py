import numpy as np
import plotly.graph_objs as obj_go ##Libs for plotting
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import time



##Vores simulations-objekter
from drone import Drone
from jammer import Jammer
from ground_station import Ground_station




class simulation:

    def __init__(self, drones: list[Drone] | None, jammers: list[Jammer] | None, ground_stations: list[Ground_station] | None):
        self.drones = drones
        self.jammers = jammers
        self.ground_stations = ground_stations
        pass


    def update_positions(self, t):
        for drone in drones:
            drone.propagate_position(t)
        
    def render_units(self, fig: go.Figure):
        for drone in drones:
            fig.add_traces(drone.render())
    
    def evaluate_links(self):
        for drone in drones:
            









drones = [ 
    Drone("hans", "testpath1.csv"),
    Drone("holger", "happy_path.csv"),
    Drone("poul", "happy_path2.csv")
]

jammers = [
    Jammer("Vlad",          np.array(([20, -10, 0.5]))),
    Jammer("Stanislav",     np.array(([20,   0, 0.5]))),
    Jammer("J.D. Vance",    np.array(([20,  10, 0.5])))
]

ground_stations = [
    Ground_station("Poul",  np.array([-20, -10, 0.5])),
    Ground_station("8700",  np.array([-20,   0, 0.5])),
    Ground_station("pingo", np.array([-20,  10, 0.5]))
]





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
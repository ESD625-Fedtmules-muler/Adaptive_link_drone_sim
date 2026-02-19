from __future__ import annotations
import numpy as np
import plotly.graph_objs as obj_go ##Libs for plotting
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import time
import pandas as pd
import dash_cytoscape as cyto


import matplotlib.pyplot as plt

import seaborn as sns
##Vores simulations-objekter
from drone import Drone
from jammer import Jammer
from ground_station import Ground_station



class simulation:


    def _df_to_cytoscape_elements(self, df, red_path=None, green_path=None):
        elements = []

        nodes = set(df["tx"]).union(set(df["rx"]))

        for node in nodes:
            node_dict = {
                "data": {
                    "id": str(node),
                    "label": str(node)
                }
            }

            classes = []

            if red_path and str(node) in red_path:
                classes.append("path-red")

            if green_path and str(node) in green_path:
                classes.append("path-green")

            if classes:
                node_dict["classes"] = " ".join(classes)

            elements.append(node_dict)

        for _, row in df.iterrows():
            edge = {
                "data": {
                    "source": str(row["tx"]),
                    "target": str(row["rx"]),
                    "rssi": row["rssi"],
                    "snr": row["snr"],
                    "label": f"RSSI: {(10*np.log10(np.abs(row['rssi']))):.1f} dBm | "
                            f"SNR: {10*np.log10(np.abs(row['snr'])):.1f} dB"
                }
            }

            classes = []

            if red_path:
                for i in range(len(red_path) - 1):
                    if str(row["tx"]) == red_path[i] and str(row["rx"]) == red_path[i+1]:
                        classes.append("path-red")

            if green_path:
                for i in range(len(green_path) - 1):
                    if str(row["tx"]) == green_path[i] and str(row["rx"]) == green_path[i+1]:
                        classes.append("path-green")

            if classes:
                edge["classes"] = " ".join(classes)

            elements.append(edge)
        self.digraph = elements
        return elements



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

        for ground_station in ground_stations:
            fig.add_traces(ground_station.render())
        
        for jammer in jammers:
            fig.add_traces(jammer.render())
    
    def evaluate_links(self, noise_floor: float = -110) -> pd.DataFrame:
        p_n_floor = 10**(noise_floor/10)*0.001
        rssi = []
        snr = []
        rx = []
        tx = []
        coords = []

        # --- Drone → Drone ---
        for drone_rx in drones:
            for drone_tx in drones:
                if id(drone_rx) == id(drone_tx):
                    continue  # skip self
                p_s, p_n = drone_rx.probe_direction(drone_tx, jammers)
                rssi.append(p_s + p_n)
                snr.append(p_s / p_n)
                rx.append(drone_rx.name)
                tx.append(drone_tx.name)
                coords.append(drone_tx.get_position())


        # --- Drone → Ground Station ---
        for drone in drones:
            for gs in ground_stations:
                p_s, p_n = drone.probe_direction(gs, jammers)
                p_n += p_n_floor
                rssi.append(p_s + p_n)
                snr.append(p_s / p_n)
                rx.append(drone.name)
                tx.append(gs.name)
                coords.append(gs.get_position())

        # --- Ground Station → Drone ---
        for gs in ground_stations:
            for drone in drones:
                p_s, p_n = gs.probe_direction(drone, jammers)
                p_n += p_n_floor
                rssi.append(p_s + p_n)
                snr.append(p_s / p_n)
                rx.append(gs.name)
                tx.append(drone.name)
                coords.append(drone.get_position())


        return pd.DataFrame({
                "rssi"  :   rssi,
                "snr"   :   snr,
                "rx"    :   rx,
                "tx"    :   tx,
                "tx_coords": coords,
                })

import numpy as np

def dijkstra_shortest_path(links, start, goal):
    """
    links: pd.DataFrame with columns ['tx', 'rx', 'snr']
    start: start node
    goal: goal node
    Returns: list of nodes representing shortest path from start to goal
    """

    links["P_n"] = links["rssi"] * np.divide(1, (links["snr"]+1))



    # Build adjacency dictionary: {node: [(neighbor, weight), ...]}
    adj = {}
    for _, row in links.iterrows():
        tx, rx, p_n = row["tx"], row["rx"], row["P_n"]
        if tx not in adj:
            adj[tx] = []
        adj[tx].append((rx, p_n))
        # For directed graph, don't add reverse

    # Initialize distances and previous nodes
    nodes = list(set(links["tx"]).union(links["rx"]))
    dist = {node: np.inf for node in nodes}
    prev = {node: None for node in nodes}
    dist[start] = 0

    unvisited = set(nodes)

    while unvisited:
        # Pick the unvisited node with the smallest distance
        current = min(unvisited, key=lambda node: dist[node])

        # If the smallest distance is infinity, goal is unreachable
        if dist[current] == np.inf:
            break

        # If we reached the goal, stop
        if current == goal:
            break

        unvisited.remove(current)

        # Update distances to neighbors
        for neighbor, weight in adj.get(current, []):
            if neighbor in unvisited:
                alt = dist[current] + weight
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]

    path.reverse()

    if path[0] == start:
        return path
    else:
        print(f"No path from {start} to {goal}")
        return []

def handle_path_coords(links, path):
    x = []
    y = []
    z = []
    for node in path:
        pos = links["tx_coords"][links["tx"] == node].iloc[0]
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
    return (x,y,z)

drones = [ 
    Drone("hans", "backfourth.csv"),
    Drone("holger", "happy_path.csv"),
    Drone("bongers", "happy_path2.csv"),
    Drone("poul", "left2right.csv")
]

jammers = [
    #Jammer("Vlad",          np.array(([20, -10, 0.5]))),
    Jammer("Stanislav",     np.array(([0,  1800, 2])), 27)
    #Jammer("J.D. Vance",    np.array(([20,  10, 0.5])))
]

ground_stations = [
    #Ground_station("Poul",  np.array([-20, -10, 0.5])),
    Ground_station("8700",  np.array([0,   0,   2]))
    #Ground_station("pingo", np.array([-20,  10, 0.5]))
]


sim = simulation(drones, jammers, ground_stations)


snr_list = []
timestamp = []
done_flag = False

t_start = time.time()

app = Dash(__name__)


app.layout = html.Div(
    style={
        "height": "100vh",
        "display": "flex",        # enables side-by-side layout
        "flexDirection": "row"
    },
    children=[

        # LEFT SIDE (3D Plotly graph)
        html.Div(
            style={"width": "50%", "height": "100%"},
            children=[
                dcc.Graph(
                    id="live-graph",
                    style={"width": "100%", "height": "100%"}
                ),
            ],
        ),

        # RIGHT SIDE (Directed graph via Cytoscape)
        html.Div(
            style={"width": "50%", "height": "100%"},
            children=[
                cyto.Cytoscape(
                    id="digraph",
                    elements=[],  # updated via callback
                    layout={"name": "preset"},  # positions are fixed
                    style={"width": "100%", "height": "100%"},
                    stylesheet=[
                        {
                            "selector": "node",
                            "style": {
                                "label": "data(label)",
                                "background-color": "#666666",
                                "color": "white",
                                "text-valign": "center",
                                "text-halign": "center",
                            },
                        },
                        {
                            "selector": "edge",
                            "style": {
                                "label": "data(label)",
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                                "arrow-scale": 1.5,
                                "line-color": "#aaa",
                                "target-arrow-color": "#aaa",
                                "font-size": "10px",
                            },
                        },
                        {
                            "selector": "edge.path-red",
                            "style": {
                                "line-color": "#FF4136",
                                "target-arrow-color": "#FF4136",
                                "width": 4,
                                "arrow-scale": 2,
                            },
                        },
                        {
                            "selector": "edge.path-green",
                            "style": {
                                "line-color": "#2ECC40",
                                "target-arrow-color": "#2ECC40",
                                "width": 4,
                                "arrow-scale": 2,
                            },
                        },
                    ],
                )
            ],
        ),

        # Interval (can stay at root level)
        dcc.Interval(id="interval", interval=100, n_intervals=0),
    ],
)
@app.callback(
        
    Output("live-graph", "figure"),

    Output("digraph", "elements"),

    Input("interval", "n_intervals")
)




def update(n):
    t_now = time.time()-t_start
    fig = go.Figure()

    sim.update_positions(t_now)
    links = sim.evaluate_links()

    
    route_1 = dijkstra_shortest_path(links=links, start="8700", goal="poul")
    route_2 = dijkstra_shortest_path(links=links, start="poul", goal="8700")
    print(route_2)


    pathcoords_1 = handle_path_coords(links, route_1)
    pathcoords_2 = handle_path_coords(links, route_2)


    if (n % 1 == 0):
        sim._df_to_cytoscape_elements(links, route_1, route_2) ##upstream rød, downstream grøn

    sim.render_units(fig) #Renders alle vores droner og jammere


    #RSSI = P_s + P_n
    #SNR = P_s/P_n
    #P_s/SNR = P_n
    #P_s = SNR*P_n
    #RSSI = (SNR+1)   * P_n
    #RSSI = (1/SNR+1) * P_s

    #okay så nu skal vi løse for drone til GS
    snr_min = 10000
    i = 0
    while (True):
        tx = route_2[0+i]
        rx = route_2[1+i]

        rssi = links[(links["tx"] == tx) & (links["rx"] == rx)]["rssi"].iloc[0]
        snr = links[(links["tx"] == tx) & (links["rx"] == rx)]["snr"].iloc[0]
        
        snr_min = min(snr_min, snr)
        
        if(rx == "8700"):
            break
        else:
            i+=1

    if n % 10 == 0:
        print(f"SNR: {10*np.log10(np.abs(snr_min)) :.2f}")
    
    fig.add_trace(
        go.Scatter3d(
            x=pathcoords_1[0],
            y=pathcoords_1[1],
            z=pathcoords_1[2],
            mode='lines',  # line only
            line=dict(color='red', width=4),  # set line color and thickness'
            name="To drone"
        )
    )
    # Second line
    fig.add_trace(
        go.Scatter3d(
            x=pathcoords_2[0],
            y=pathcoords_2[1],
            z=pathcoords_2[2],
            mode='lines',
            line=dict(color='green', width=4),
            name="To ground station"
        )
    )
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
    return fig, sim.digraph



def assign_random_pos(min_y, rng: np.random.Generator):
    return np.array([
        rng.uniform(-1000, 1000),
        rng.uniform(min_y, 1800+500),
        2
    ])




if __name__ == "__main__":
    #app.run(debug=True)
    rng = np.random.default_rng(1)
    
    runs = 300
    snrd2b = []
    snrb2d = []
    final_time = []
    for run in range(runs):
        print(f"iterations: {run}/{runs}")
        jammers = [
            Jammer("0",     assign_random_pos(1800, rng), 27),
            Jammer("1",     assign_random_pos(1400, rng), 10),
            Jammer("2",     assign_random_pos(1400, rng), 10),
            Jammer("3",     assign_random_pos(1400, rng), 10),
        ]


        drones=[
            Drone("poul", "reference_path.csv")
        ]

        ground_stations = [
            Ground_station("8700", np.array([0,0,2]))
        ]

        sim = simulation(drones, jammers, ground_stations)


        timespan = np.linspace(0, 400, 200)



        for i, t in enumerate(timespan):
            sim.update_positions(t)
            links = sim.evaluate_links()
            
            snrb2d.append(10*np.log10(links["snr"][(links["tx"] == "8700") & (links["rx"] == "poul")].iloc[0]))
            snrd2b.append(10*np.log10(links["snr"][(links["tx"] == "poul") & (links["rx"] == "8700")].iloc[0]))
            final_time.append(t)

    df = pd.DataFrame()
    df["time"] = final_time
    df["d2b"] = snrd2b
    df["b2d"] = snrb2d

    sns.lineplot(data=df, x="time", y="d2b", label="drone to base", errorbar=("sd", 2))
    sns.scatterplot(data=df, x="time", y="d2b", label="drone to base", s=1)

    sns.lineplot(data=df, x="time", y="b2d", label="base to drone", errorbar=("sd", 2))
    sns.scatterplot(data=df, x="time", y="b2d", label="base to drone", s=1)
    plt.grid(True)
    plt.show()
    
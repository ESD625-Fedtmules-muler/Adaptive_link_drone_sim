import numpy as np
import numpy.typing as npt

import plotly.graph_objs as obj_go
import plotly.graph_objects as go


class Ground_station:
    def __init__(self, name, position: npt.NDArray[np.floating], tx_power = 0, colour='green',):
        """
        Docstring for __init__
        :param self: Description
        :param position: position in the 3 dimensionl space
        :type position: npt.NDArray[np.floating]
        :param tx_power: Transmission power in dBm
        """

        self.position = position
        self.tx_power = np.power(10, tx_power/10)*0.001
        self.name = name
        self.colour = colour
        pass

    

    def get_position(self):
        pass
    

    

    def render(self) -> go.Scatter3d:
        return go.Scatter3d(
            x=[self.position[0]],
            y=[self.position[1]],
            z=[self.position[2]],
            mode='markers+text',  # enable text display
            marker=dict(size=10, color=self.colour, symbol='cross'),
            text=[self.name],       # or any text you want
            textposition='top center'
        )
    


if __name__ == "__main__":
    fig = go.Figure(
        data=go.Scatter3d(
            x=[3,2,1,0],
            y=[0,1,2,5],
            z=[1,2,3,4],
            mode='lines',
            marker=dict(size=5)
        )
    )
    


    stations = [
        Ground_station("Test1", np.array([0,0,0])),
        Ground_station("Test2", np.array([1,0,0])),        
        Ground_station("Test3", np.array([3,8,0])),        
        Ground_station("Test4", np.array([0,4,0]))     
    ]


    for station in stations:
        fig.add_trace(station.render())
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )

    
    fig.show()
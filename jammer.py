import numpy as np #Geden, numpy.
import numpy.typing as npt #For at håndtere types, (det gør jeres liv nemmere)
import plotly.graph_objs as obj_go ##Libs for plotting
import plotly.graph_objects as go


class jammer:
    def __init__(self, name, position: npt.NDArray[np.floating], tx_power = 0, colour='red',):
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

    def get_position(self, t) -> npt.NDArray[np.floating]:
        return self.position
    
    def get_power(self):
        return self.tx_power

    def render(self, t) -> go.Scatter3d:
        return go.Scatter3d(
            x=[self.position[0]],
            y=[self.position[1]],
            z=[self.position[2]],
            mode='markers+text',  # enable text display
            marker=dict(size=10, color=self.colour, symbol='square'),
            text=[self.name],       
            textposition='top center'
        )



from __future__ import annotations
import numpy as np
import numpy.typing as npt

import plotly.graph_objs as obj_go
import plotly.graph_objects as go

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jammer import Jammer
    from drone import Drone


class Ground_station:
    def __init__(self, name, position: npt.NDArray[np.floating], tx_power = 20, colour='green',):
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
        self.noise_floor = 10**(-110/10)*0.001

        pass

    

    def get_position(self) -> npt.NDArray[np.floating]:
        return self.position
    

    def get_power(self) -> float:
        return self.tx_power

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
    

    def _project_vectors(self, vec1, vec2):
        
        #Normaliser vektor 1
        if (np.linalg.norm(vec1) == 0):
            vec1 = np.array([1,0,0])
        else:
            vec1 = vec1 / np.linalg.norm(vec1)

        #vec1 svarer til forward
        world_up = np.array([0.0, 0.0, 1.0])
        #  vec1 næsten parallel med up er noget lort
        if abs(np.dot(vec1, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])

        right = np.cross(world_up, vec1) / np.linalg.norm(np.cross(world_up, vec1))


        up = np.cross(vec1, right)
        up /= np.linalg.norm(up)

        # World → vec1 local
        R_mat = np.vstack((vec1, right, up))
        vec2_t = R_mat @ vec2   #Skulle eftersigende være matrixmultiplikation


        euler = np.array([
            np.arctan2(vec2_t[1], vec2_t[0]),                       # yaw / heading
            np.arctan2(vec2_t[2], np.sqrt(vec2_t[0]**2 + vec2_t[1]**2))  # pitch
        ])
        return euler

    def get_radiation(self, angle: npt.NDArray[np.floating]) -> float:
        """
        Docstring for get_radiation
        
        :param angle: numpy 2d-vector giving the euler angles (Rad) into the radiation pattern
        :type angle: npt.NDArray[np.floating]
        :return: A real float with the absolute magnitude radiation NOT dB
        :rtype: float
        """

        
        return np.abs((np.cos(angle[0]/2)**20)) *np.abs((np.cos(angle[1]/2)**20)) #ah yes, direktionel antenna
    

    def probe_direction(self, 
                        target: Drone | Ground_station, 
                        jammers: None | list[Jammer] = None, 
                        drones: None | list[Drone] = None
                        ) -> list[float, float]:
        """
        Docstring for probe_direction
        :param self: Description
        :param target: Description
        :param jammers: Description
        :type jammers: None | list[Jammer]
        :param drones: Description
        :type drones: None | list[Drone]
        """


        dir_target = (target.get_position() - self.position) ##giver os retningsvektoren fra os til le' target
        dir_euler = self._project_vectors(dir_target, dir_target) ##I og for sig rigtig meget en vektor som helst sku gi 0,0 i grader 
        G_s_rx = self.get_radiation(dir_euler) #Reciever 
        P_EIRP_tx = target.get_power() ##Assumer at target transmitter med 1.
        P_s = self._fspl(G_s_rx, P_EIRP_tx, np.linalg.norm(dir_target))
        
        P_n = 0

        if type(jammers) != None:
            for jammer in jammers:
                dir_jammer = jammer.get_position() - self.position
                P_EIRP_tx = jammer.get_power()
                G_s_rx = self.get_radiation(self._project_vectors(dir_target, dir_jammer))
                P_n += self._fspl(G_s_rx, P_EIRP_tx, np.linalg.norm(dir_jammer))
        
        return P_s, P_n + self.noise_floor
    def _fspl(self, G_rx, P_tx, distance) -> float:
        """
        Docstring for _fspl
        
        :param self: Description
        :param G_rx: Lineart gain, skal regnes fra dbi
        :param P_tx: Lineart, af Gain og txpower.
        :param distance: afstand i meter
        Gang selv
        """
        lbda = 0.12491352 # meter
        
        return ((lbda)/(4*np.pi*distance))**2 * G_rx * P_tx



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
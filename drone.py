from __future__ import annotations
import numpy as np
import pandas as pd
import logging as log
import numpy.typing as npt

import plotly.graph_objects as go
import plotly.io as pio

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ground_station import Ground_station
    from jammer import Jammer




class Drone:
    def __init__(self, name: str, filename: str, loop=True):
        self.name = name
        self.path_foler = "./paths/"
        self.loop = loop #if the path just should repeat itself.
        self.t_lookup, self.pos_lookup = self._parse_path(filename) ##Konstruere vores lister med
        self.colour = "red"
        self.antenna_direction = npt.NDArray[np.floating]
        self.pos = np.array([0,0,0])
        self.tx_power = 0.1
        self.noise_floor = 10**(-110/10)*0.001

        pass



    def _parse_path(self, filename: str):
        if (type(filename) is None):
            log.error(f"{self.name}:, No path was given assuming 1")
            return ([0], np.array([1,1,1])) 

        data = pd.read_csv(self.path_foler + filename)
        pos = np.transpose(np.array([
            data["x"],
            data["y"],
            data["z"]
        ]))
        
                
        return data["time"], pos


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

        # World => vec1 local
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
        
        return (np.cos(angle[0]/2)**10)* (np.cos(angle[1]/2)**10) + 0.05#ah yes, direktionel antenna


    def propagate_position(self, t) -> npt.NDArray[np.floating]:
        if self.loop: #if we need to loop around
            t = t % np.max(self.t_lookup)
            idx_1 = np.searchsorted(self.t_lookup, t, side="right") - 1
            idx_1 = max(idx_1, 0)
            idx_2 = (idx_1 + 1) % len(self.t_lookup) ## Next index 
        else: 
            #TODO Der lidt skald der skal fikses her, tror den dør og dividerer med nul.. hvis man når enden af listen uden  
            idx_1 = np.searchsorted(self.t_lookup, t, side="right") - 1 
            idx_1 = max(idx_1, 0)

            idx_2 = np.min((idx_1 + 1), len(self.t_lookup)-1) ## Så finder index 2, hvis det uden for arrayet så smider vi det sidste index med. 

        t_1 = self.t_lookup[idx_1]
        t_2 = self.t_lookup[idx_2]
        if(t_1 == t_2):
            raise Exception("Time is not relativistic in this simulation, a done can only be in one position at one time") 
        alpha = (t-t_1) / (t_2 -t_1)
        self.lastpos = self.pos
        self.pos = alpha*self.pos_lookup[idx_2] + (1-alpha)*self.pos_lookup[idx_1]
        
        return self.pos  ##Linear interpolate between the two points

    def get_position(self) -> npt.NDArray[np.floating]:
        """
        Gives the current position vector for the drone
        
        :return: Description
        :rtype: NDArray[floating]
        """
        return self.pos  ##Linear interpolate between the two points


    def get_direction(self):
        direction_vector = self.pos - self.lastpos
        if np.linalg.norm(direction_vector) < 0.001:
            return np.array([1,0,0]) #Så lille retning ikke giver mening

        return np.divide(direction_vector, np.linalg.norm(direction_vector))

    
    def get_power(self, angle: npt.NDArray[np.floating] | None = None) -> float:
        if angle == None:
            return self.tx_power
        else:
            return self.get_radiation(angle) * self.tx_power
        


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


        dir_target = (target.get_position() - self.pos) ##giver os retningsvektoren fra os til le' target
        dir_euler = self._project_vectors(dir_target, dir_target) ##I og for sig rigtig meget en vektor som helst sku gi 0,0 i grader 
        G_s_rx = self.get_radiation(dir_euler) #Reciever 
        P_EIRP_tx = target.get_power() ##Assumer at target transmitter med 1.
        P_s = self._fspl(G_s_rx, P_EIRP_tx, np.linalg.norm(dir_target))
        
        P_n = 0

        if type(jammers) != None:
            for jammer in jammers:
                dir_jammer = jammer.get_position() - self.pos
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

    
    def render(self):
        pos = self.pos
        direction = self.get_direction() / 4


        line = go.Scatter3d(
            x=[pos[0], pos[0] + direction[0]],
            y=[pos[1], pos[1] + direction[1]],
            z=[pos[2], pos[2] + direction[2]],
            mode="lines+text",
            text=[None, self.name],          # tekst kun på spidsen
            textposition="top center",
            line=dict(
            width=6,
            color=self.colour,         
            ),
            name = self.name
        )

        # pilens spids (cone)
        cone = go.Cone(
            x=[pos[0] + direction[0]],
            y=[pos[1] + direction[1]],
            z=[pos[2] + direction[2]],
            u=[direction[0]],
            v=[direction[1]],
            w=[direction[2]],
            sizemode="absolute",
            sizeref=0.1,      # justér efter smag
            anchor="tip",
            showscale=False,
            colorscale=[[0, self.colour], [1, self.colour]], 
        )
        return [line, cone]




if __name__ == "__main__":
    ##magiske hacks så vi får det vist i 3d
    pio.renderers.default = "browser"


    drone_hans = Drone("drone: 1", "testpath1.csv")
    drone_hans2 = Drone("drone: 2", "happy_path.csv")

    print(drone_hans._project_vectors(np.array([0,1,0]), np.array([0,1,0])))

    x,y,z  = [],[],[]
    time = np.linspace(0, 24, 400)
    for t in time:
        pos = drone_hans2.propagate_position(t)
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
    

    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            marker=dict(size=5)
        )
    )

    fig.add_traces( drone_hans2.render())
    fig.update_layout(
        scene=dict(aspectmode='data')
    )
    fig.show()
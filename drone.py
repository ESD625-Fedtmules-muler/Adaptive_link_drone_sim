import numpy as np
import pandas as pd
import logging as log
import numpy.typing as npt

import plotly.graph_objects as go
import plotly.io as pio

class drone:
    def __init__(self, name: str, filename: str, loop=True):
        
        self.name = name
        self.path_foler = "./paths/"
        self.loop = loop #if the path just should repeat itself.
        self.t_lookup, self.pos_lookup = self._parse_path(filename) ##Konstruere vores lister med
        self.colour = "red"
        self.antenna_direction = npt.NDArray[np.floating]
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

    
    def get_radiation(angle: npt.NDArray[np.floating]) -> float:
        """
        Docstring for get_radiation
        
        :param angle: numpy 2d-vector giving the euler angles (Rad) into the radiation pattern
        :type angle: npt.NDArray[np.floating]
        :return: A real float with the absolute magnitude radiation NOT dB
        :rtype: float
        """

        
        return np.cos(angle[0]/2)**2*np.cos(angle[0]/2)**2 #ah yes, direktionel antenna
    

    def get_position(self, t) -> npt.NDArray[np.floating]:
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
        pos = alpha*self.pos_lookup[idx_2] + (1-alpha)*self.pos_lookup[idx_1]
        return pos  ##Linear interpolate between the two points


    def get_direction(self, t):
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

        
        direction_vector = self.pos_lookup[idx_2] - self.pos_lookup[idx_1]
        
        return np.divide(direction_vector, np.linalg.norm(direction_vector))
    
    def probe_direction(self, target_pos):
        """
        Docstring for probe_direction
        
        :param self: Description
        :param target: Description
        """


    
    
    def render(self, t):
        pos = self.get_position(t)
        direction = self.get_direction(t) / 4


        line = go.Scatter3d(
            x=[pos[0], pos[0] + direction[0]],
            y=[pos[1], pos[1] + direction[1]],
            z=[pos[2], pos[2] + direction[2]],
            mode="lines+text",
            text=[None, self.name],          # tekst kun på spidsen
            textposition="top center",
            line=dict(
            width=6,
            color=self.colour         
            ),
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


    drone_hans = drone("drone: 1", "testpath1.csv")
    drone_hans2 = drone("drone: 2", "happy_path.csv")

    x,y,z  = [],[],[]
    time = np.linspace(0, 24, 400)
    for t in time:
        pos = drone_hans2.get_position(t)
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

    fig.add_traces(drone_hans.render(12))
    
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )
    fig.show()


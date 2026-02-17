from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
import numpy.typing as npt
import matplotlib.pyplot as plt




class SDA:
    def __init__(self, num_elements: int, directions = None) -> SDA:

        self.num_elements = num_elements
        if directions != None:
            self.directions = directions
        else:
            self.directions = np.linspace(0, 2*np.pi, num_elements)


        self.HPBW = 2*np.pi / num_elements
        self.tightness = np.ceil(np.log(0.5)/np.log(np.cos(self.HPBW/4)))
        self.yaw_select = 0

    def set_direction(self, yaw):
        self.yaw_select = np.argmin(np.abs(self.directions - yaw))



    def get_radiation(self, angle: npt.NDArray[np.floating]) -> float:
        ##NDA array with yaw, pitch
        
        angle[0] = angle[0] - self.directions[self.yaw_select]


        return (np.cos(angle[0]/2)**self.tightness)



if __name__ == "__main__":
    
    
    antenna = SDA(8)
    antenna.set_direction(np.deg2rad(10))

    
    
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'}, layout='constrained')
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    ax = axs[0]
    ax.plot(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Fewer radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

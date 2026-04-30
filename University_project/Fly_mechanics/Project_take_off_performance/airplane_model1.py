"""
Author: Friedly WOLI
Date: 27th of april 2026

Students in third year in mechanics and energetic at Toulouse university









"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g  # Accélération de la pesanteur en m/s²


#-===========================================================================================
# Definition de la classe aircraft
#-===========================================================================================

class Aircraft :


    def __init__(self, M, hw, Sw, bw ):

        self.M = M                  # mass of the aircraft
        self.hw= hw                 # height of the wing above the ground
        self.Sw = Sw                # wing surface
        self.bw = bw                # wingspan 







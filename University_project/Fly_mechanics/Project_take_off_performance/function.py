
"""
Author: Friedly WOLI
Date: 27th of april 2026

Students in third year in mechanics and energetic at Toulouse university

Modified the 02/05/2026







"""

import numpy as np
import matplotlib.pyplot as plt 
from airplane_model1 import Aircraft


g_ref = 9.80665   # Accélération de la gravité moyenne en m/s^2
rho = 1000



# parameter

# 1ft = 0.3048 m        ; 1 Ibf is one pound froce and 1 Ib = 0.45359237 Kg
ft=0.3048
Sw=180                       #in ft**2                          is the wing surface 
bw= 33                       # in ft                            is the wingspan
hw=6.0                       # in ft                            is the height of the wing above the ground 
W= 2700                      # in Ibf                           is the weight 
# W/Sw = 15                    # in Ibf/ft**2                     is the wing loading
Ra= 6.05                     # without unit as                  is an aspect ratio
Cdo = 0.036                  #  without unit
Cdol = 0                   # without unit
e = 0.82 #                                                      is the span efficiency factor 
Clmax= 1.4     #                                                is the lift coef
mu = 0.04 #                                                     is the rolling coef 
Cl = 0.34885 
Cd = 0.042969
Vlo = 104.44
Vhw = 29.33

g= g_ref
V1 = 29.33
V2= 104.44
Toi = 1200                  # experimentally determined coef for the thrust
Ti = -4                     # experimentally determined coef for the thrust
Tii = 0                     # experimentally determined coef for the thrust


aicraft1= Aircraft( W, hw, Sw, bw )

drag_ceoff = aicraft1.drag_coef ( Cdo, Cdol, e, Ra, Clmax )

dist= aicraft1.dictance ( V1, V2, Toi , Ti , Tii, Vhw, g , mu, Cl, Cd, rho)

print (dist)

"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 20 May 2026


--------------------------------
Goal |                          To convert from US unity to SI unity
--------------------------------

"""

# Which exercices are being validated by my code 
"""
Verify by all the code

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



     # Conversion factors: US -> SI
CONV = {"Sw": 0.3048**2, "bw": 0.3048, "hw": 0.3048, "W": 4.44822,  "T0": 4.44822, "T1": 4.44822 / 0.3048,  "T2": 4.44822 / (0.3048**2),  "rho": 515.3788,  "Vhw": 0.3048,  "Vi": 0.3048,  "g": 0.3048 }

# =====================================================
# Unit conversion
# =====================================================

def convert_units(p, new_unit , last_unit):

    if last_unit == new_unit:
        print(f"Already in {new_unit}")
        return

    if last_unit == "US" and new_unit == "SI":
        for key, value in CONV.items():
            if key in p:
                p[key] *= value

    elif last_unit == "SI" and new_unit == "US":
        for key, value in CONV.items():
            if key in p:
                p[key] /= value

    return p


def modify_dict(data, param):
    data = convert_units(data, "US", "SI")

    for key in data:
        if key in param:
            param[key] = data[key]

    return data, param



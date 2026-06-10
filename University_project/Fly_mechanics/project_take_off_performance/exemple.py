"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON

--------------------------------
Methode developped in this code | Ground run estimation using numerical Integration method ;       page 793 et 799
--------------------------------

------------
Advantages 1| : Handle all the Take-off problems but more complex;      Can model the braking distance to simulate an engine faillure
Advantages 2| : Useful to solve some problem enconterred during the take-off 
------------|

-----------
Asumption | V_lof is assumed to be 1.1* stalling speed  
----------

"""

# Which exercices are being validated by my code 
"""
Example 18.7 ;          Notice:     the ground run distance after running the code is 1111.04 


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
import pandas as pd

# self.p[""]

# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff5:

    def __init__(self, params):
        """
        Initialize all parameters from dictionary
        """
        self.p = params

    #==============================================
    # Lift at rotation speed
    # =====================================================
    def lift(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Cl"]

    # =====================================================
    # Drag at rotation speed
    # =====================================================
    def drag(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Cd"]

    # =====================================================
    # Thrust for piston engine
    # =====================================================
    def thrust_piston2(self):
        return self.p["efficiency"] * 550 * self.p["P"] * np.sqrt(2) / self.Vlo()

    def Friction_force(self, V):
        
        return self.p["mu"] * ( self.p["W"] - self.lift(V) )
    # =====================================================
    # lift off speed
    # =====================================================
    def Vlo(self):
        p = self.p
        return 1.1 * np.sqrt(2 / p["Clmax"]) * np.sqrt(p["W"] / (p["Sw"] * p["rho"]))


    def ground_run(self):

        V = np.linspace(0, self.Vlo(), self.p["n"])

        n = len(V)
        dV = V[1] - V[0]

        A = np.zeros((n, n))
        b = np.zeros(n)

        # Condition S(0)=0
        A[0, 0] = 1
        b[0] = 0

        # Différence avant ordre 2
        for i in range(n - 2):

            A[i + 1, i]     = -3
            A[i + 1, i + 1] = 4
            A[i + 1, i + 2] = -1

            f = (
                self.p["W"] / self.p["g"]
                * V[i]
                / (
                    self.thrust_piston2()
                    - self.drag(V[i])
                    - self.Friction_force(V[i])
                )
            )

            b[i + 1] = 2 * dV * f

        # Dernière équation :
        # S(Vlo) = distance de décollage
        A[-1, -1] = 1
        A[-1, -2] = -1

        b[-1] = dV * (
            self.p["W"] / self.p["g"]
            * V[-1]
            / (
                self.thrust_piston2()
                - self.drag(V[-1])
                - self.Friction_force(V[-1])
            )
        )

        S = np.linalg.solve(A, b)

        return {
            "speed": V,
            "distance": S,
            "ground_run": S[-1]
        }



data2 = {
    "Unit": "US",    "name": "None",    "engine": "jet",
    "W": 2700,    "Sw": 180,    "bw": 33,    "hw": 6,    "Ra": 6.05,
    "Clmax": 1.4,    "Cl": 0.3485,    "Cd": 0.03,    "Cdo": 0.036,    "Cdol": 0.0,    "e": 0.82,
    "g": 32.2,    "mu": 0.04,    "rho": 0.0023769,
    "P": 310,    "T": 1200,    "T0": 1200,    "T1": -4,    "T2": 0,    "efficiency": 0.5,
    "Vhw": 0,    "Vi": 0, "tr": 1 ,"hoc":35 ,
    "alpha": 0,    "alpha_0": 0,
    "A1": 1.158e-2,    "A2": -5.277e-05,    "A3": 9.273e-08,    "A4": -6.21e-11,
    "n": 100
}

param = get_model_params("Model4", data2)
plane=AircraftTakeoff5(param)


print (plane.ground_run())
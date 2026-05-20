"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 20 May 2026

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
Example 18.5 page 797 in book B2     ; with a little bit of error
Example 18.6 page 798 in book B2     ; with      little bit of error


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# self.p[""]

# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff:

    def __init__(self, params, unit):
        """
        Initialize all parameters from dictionary
        """
        self.p = params
        self.unit = unit

        # Conversion factors: US -> SI
        self.CONV = {
            "Sw": 0.3048**2,
            "bw": 0.3048,
            "hw": 0.3048,
            "W": 4.44822,
            "T0": 4.44822,
            "T1": 4.44822 / 0.3048,
            "T2": 4.44822 / (0.3048**2),
            "rho": 515.3788,
            "Vhw": 0.3048,
            "Vi": 0.3048,
            "g": 0.3048
        }

    # =====================================================
    # Unit conversion
    # =====================================================
    def convert_units(self, new_unit="SI"):

        if self.unit == new_unit:
            print(f"Already in {new_unit}")
            return

        if self.unit == "US" and new_unit == "SI":
            for key, value in self.CONV.items():
                if key in self.p:
                    self.p[key] *= value

        elif self.unit == "SI" and new_unit == "US":
            for key, value in self.CONV.items():
                if key in self.p:
                    self.p[key] /= value

        self.unit = new_unit

    #==============================================
    # Lift at rotation speed
    # =====================================================
    def lift(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Clto"]

    # =====================================================
    # Drag at rotation speed
    # =====================================================
    def drag(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Cdto"]

    # =====================================================
    # Thrust for piston engine
    # =====================================================
    def thrust_piston(self, V):
        if V == 0 :
            return self.p["T"]

        cste = self.p["A1"] * V + self.p["A2"] * V**2 + self.p["A3"] * V**3 + self.p["A4"] * V**4
        return 550 * self.p["P"] * cste / (1.688 * V)

    # =====================================================
    # Ground run distance
    # =====================================================
    def ground_run(self):

        t = np.linspace(0, 17, self.p["n"])

        S, dS = np.zeros(self.p["n"]), np.zeros(self.p["n"])
        A, V = np.zeros(self.p["n"]), np.zeros(self.p["n"])
        D, L, T = np.zeros(self.p["n"]), np.zeros(self.p["n"]), np.zeros(self.p["n"])

        D[0], L[0] = 0, 0
        A[0] = 9.78

        for i in range(1, self.p["n"] ):
            D[i] = self.drag(V[i-1])
            L[i] = self.lift(V[i-1])
            T[i] = self.thrust_piston(V[i-1])

            A[i] = (self.p["g"] / self.p["W"]) * (   T[i] - D[i] - self.p["mu"] * ( self.p["W"] - L[i] )  )

            V[i] = V[i-1] + A[i] * (t[i] - t[i-1])

            dS[i] = V[i-1] * (t[i] - t[i-1]) + 0.5 * A[i] * (t[i] - t[i-1])**2

            S[i] = S[i-1] + dS[i]

        return {  "times": t,     "drag": D,     "lift": L,      "thrust": T,    "acceleration": A,   "speed": V,     "distance": S     }


# =====================================================
# Parameters
# =====================================================

param = {  "W": 3400,  "Sw": 144.9,  "Clmax": 1.69,  "Clto": 0.5, "Cdto": 0.0417, "g": 32.2, "mu": 0.04, "P": 310, "rho": 0.002378,  "T": 1169,
                "lamdba": 0,  "n": 35,  "A1": 1.158e-2,  "A2": -5.277e-05,  "A3": 9.273e-8,  "A4": -6.21e-11 }

model = AircraftTakeoff(param, "US")

distance = model.ground_run()


for i in range(len(distance)):
    print(list(distance.keys())[i], "est :", distance[list(distance.keys())[i]])
    print("")
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

B2: Book used General Aviation Aicraft  Design ; Applied Methods 
and Procedures ; SNORRI GUDMUNDSSON

Methode developped in this code: General ground run estimation using 
Average Acceleration ; page 796
--------------------------------

Advantages: Applied to all aicraft as long as a proper values of thrust,
 drag and lift can be quantified ;      Applicable for both propeller and jet ;  
        ONLY DIFFERENCE IS THE THRUST CALCULATION
-----------            ------------

Asumption: As the trust, drag, and lift depend on the speed we approximate 
the speed at the Vr/np.sqrt(2) to calculate the forces : Vr us the rotation speed 
----------
 Which exercices are being validated by my code 
Example 18.5 page 797 in book B2     ; with a little bit of error
Example 18.6 page 798 in book B2     ; with      little bit of error


param = { "W": 3400, "Sw": 144.9, "Clmax": 1.69, "Cl": 0.5, "Cd": 0.0417,
 "engine": "piston", "g": 32.2, "mu": 0.04,  "P": 310, "rho": 0.002378,  "efficiency": 0.5, 
 "T": 7000}

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from matplotlib.figure import Figure

from Conversion import *
from Forces import *


class AircraftTakeoff2:
    """
    This class represents another way to compute the takeoff distance.
    """

    def __init__(self, params):
        """
        Initialize all parameters from the dictionary.
        """
        self.p = params

    def Vr(self):
        return 1.1 * np.sqrt(2 * self.p["W"]
            / (self.p["rho"] * self.p["Sw"] * self.p["Clmax"]))

    def lift2(self):
        return lift2(self.p["rho"], self.Vr(), self.p["Sw"], self.p["Cl"])

    def drag2(self):
        return drag2(self.p["rho"], self.Vr(), self.p["Sw"], self.p["Cd"])

    def thrust2(self):
        if self.p["engine"] == "piston":
            return thrust_piston2(self.p["efficiency"], self.p["P"], self.Vr())
        elif self.p["engine"] == "jet":
            return self.p["T"]

    def ground_run(self):
        D = self.drag2()
        L = self.lift2()
        T = self.thrust2()

        return (self.Vr()**2 * self.p["W"]
        ) / (2 * self.p["g"] * (T - D - self.p["mu"] * (self.p["W"] - L)))

    def set_result(self):
        return {
            "Ground run distance Sa is": round(self.ground_run(),2),
            "Lift-off speed Vlo is": round (self.Vr(),2)
        }

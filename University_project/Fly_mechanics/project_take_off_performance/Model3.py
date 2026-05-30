"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 20 May 2026

B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON

--------------------------------
Methode developped in this code | General ground run estimation using Average Acceleration for tricycle propeller aicraft ONLY ;       page 797
--------------------------------

------------
Advantages 1| : Adapted to piston engine configuration not for taildraggers beacuse the tail on ground/ tail off-ground transition is not accounted for
Advantages 2| : Combined several steps in the general methode and provides better parametrics studies 
------------|

-----------
Asumption | As the trust, drag, and lift depend on the speed we approximate the speed at the Vr/np.sqrt(2) to calculate the forces : Vr us the rotation speed 
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
from conversion import *
import pandas as pd

# self.p[""]
# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff:

    def __init__(self, params):
        """
        Initialize all parameters from dictionary
        """
        self.p = params  


    # =====================================================
    # Rotation speed
    # =====================================================
    def Vr(self):
        return 1.1 * np.sqrt(2 * self.p["W"] / (self.p["rho"] * self.p["Sw"] * self.p["Clmax"]))

    # =====================================================
    # Lift at rotation speed
    # =====================================================
    def lift(self):
        return 0.5 * self.p["rho"] * ( self.Vr() / np.sqrt(2) )**2 * self.p["Sw"] * self.p["Clto"] 

    # =====================================================
    # Drag at rotation speed
    # =====================================================
    def drag(self):
        return 0.5 * self.p["rho"] * ( self.Vr() / np.sqrt(2) )**2 * self.p["Sw"] * self.p["Cdto"]

    # =====================================================
    # Thrust for piston engine
    # =====================================================
    def thrust_piston(self):
        return self.p["efficiency"] * 550 * self.p["P"] * np.sqrt(2) / self.Vr()

    # =====================================================
    # Thrust for jet engine
    # =====================================================
    def thrust_jet_powered(self):
        return self.p["T"] # (self.Vr() / np.sqrt(2))

    # =====================================================
    # Friction
    # =====================================================
    def friction(self):
        return self.p["mu"] * (self.p["W"] - self.lift())

    # =====================================================
    # Ground run distance
    # =====================================================
    def ground_run(self):

        D = self.drag()
        L = self.lift()

        if self.p["engine"] == "piston":
            T = self.thrust_piston()

        elif self.p["engine"] == "jet":
            T = self.thrust_jet_powered()

        else:
            raise ValueError("engine must be 'piston' or 'jet'")

        return (self.Vr()**2 * self.p["W"]) / (2 * self.p["g"] * (T - D - self.p["mu"] * (self.p["W"] - L) ) )


    def summary(self):

        Vr= self.Vr()
        C_lige, C_dige =  0, 0 # need to be modify 
        L =  self.lift()
        D = self.drag()

        if self.p["engine"] == "piston":
            T = self.thrust_piston()

        elif self.p["engine"] == "jet":
            T = self.thrust_jet_powered()

        S= self.ground_run()

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print(" The Unit is  :", self.p["Unit"])
        print(" The name of the aicraft is :", self.p["name"])
        print(" The type of the engine is :", self.p["engine"])
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("The rotation speed is :", Vr)
        print("The C_lige coefficient", C_lige , " and the C_dige coefficient is :", C_dige)
        print ("The lift is :", L)
        print ("The drag is : ", D)
        print(" The thrust is :", T)
        print (" The ground run distance is :", S)



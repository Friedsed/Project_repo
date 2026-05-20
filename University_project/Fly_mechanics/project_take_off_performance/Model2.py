"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 20 May 2026

B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON

Methode developped in this code: General ground run estimation using Average Acceleration ; page 796
--------------------------------

Advantages: Applied to all aicraft as long as a proper values of thrust, drag and lift can be quantified ;      Applicable for both propeller and jet ;         ONLY DIFFERENCE IS THE THRUST CALCULATION
-----------            ------------

Asumption: As the trust, drag, and lift depend on the speed we approximate the speed at the Vr/np.sqrt(2) to calculate the forces : Vr us the rotation speed 
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
    # Drag coefficients C_dige and C_lige
    # =====================================================

    def C_dige(self): # Note complete need to be improved
        

        if self.p["wing type"] == Elliptic :

            beta_L= 1 - (0.269* self.Cl **1.45 ) / ( self.p["AR"] ** 3.18 * (h/b)**1.12 )
            gamma_L = 1 - 2.25 *( self.p["lambda"] **0.00273 - 0.997) * ( self.p["AR"]** 0.717 + 13.6 )
            beta_D = 1 + (0.0361* self.Cl**1.21) / (self.p["AR"]**1.19 * ( h/b)**1.51 )
            gamma_D = 1- 0.157 * (self.p["lamdba"] **0.775 - 0.373) * ( self.p["AR"]**0.417 - 1.27 )

            phi_L = (1/beta_L) * ( 1 + ( 288*(sel.p["h"] / self.p["b"] )**0.787 / self.p["AR"] **0.882 )* np.exp(-9.14*( self.p["h"] / self.p["b"] )**0.327) )
            phi_D =  beta_D * ( 1- np.exp(-4.74*(h/b)**0.814 ) - (h/b)**2 * np.exp(-3.88 * (h/b)**0.758 ) )

        elif self.p["wing type"] == Trapered :

            phi_L = (1/beta_L) * ( 1 + gamma_L * ( 288*( self.p["h"] / self.p["b"])**0.787 / self.p["AR"] **0.882 )* np.exp(-9.14*( self.p["h"] / self.p["b"])**0.327) )
            phi_D =  beta_D * ( 1- gamma_D * np.exp(-4.74*(h/b)*0.814 ) - (h/b)**2 * np.exp(-3.88 * (h/b)**0.758 ) )


        return phi_L**2 * phi_D # need to be complete page 380 


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

        print("The rotation speed is :", Vr)
        print("The C_lige coefficient", C_lige , " and the C_dige coefficient is :", C_dige)
        print ("The lift is :", L)
        print ("The drag is : ", D)
        print(" The thrust is :", T)
        print (" The ground run distance is :", S)




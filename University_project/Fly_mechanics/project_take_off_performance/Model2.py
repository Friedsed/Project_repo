"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

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


param = { "W": 3400, "Sw": 144.9, "Clmax": 1.69, "Cl": 0.5, "Cd": 0.0417,
 "engine": "piston", "g": 32.2, "mu": 0.04,  "P": 310, "rho": 0.002378,  "efficiency": 0.5, 
 "T": 7000}


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
import pandas as pd
from forces import *
from tkinter import *
from matplotlib.figure import Figure

# self.p[""]
# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff2:

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
    # Lift
    # =====================================================
    def lift2(self):
        return lift2( self.p["rho"],  self.Vr(),  self.p["Sw"],  self.p["Cl"]  )
    
    # =====================================================
    # Drag
    # =====================================================
    def drag2(self):
        return drag2(  self.p["rho"],  self.Vr(),   self.p["Sw"],   self.p["Cd"]    )
    # =====================================================
    # Thrust
    # =====================================================
    def thrust2(self):
        if self.p["engine"] == "piston":
            return thrust_piston2(   self.p["efficiency"], self.p["P"],  self.Vr()    )

        elif self.p["engine"] == "jet":

            return self.p["T"]
    # =====================================================
    # Ground run distance
    # =====================================================
    def ground_run(self):

        D = self.drag2()
        L = self.lift2()
        T =  self.thrust2()
        return (self.Vr()**2 * self.p["W"]) / (2 * self.p["g"] * ( T - D - self.p["mu"] * (self.p["W"] - L) ) )


    def set_result(self):

       
        return {"Runaway distance is": self.ground_run() , "The lift off speed": self.lift2() }



"""
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print(" The Unit is  :", self.p["Unit"])
        print(" The name of the aicraft is :", self.p["name"])
        print(" The type of the engine is :", self.p["engine"])
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("The rotation speed is :", Vr)
        print ("The lift is :", L)
        print ("The drag is : ", D)
        print(" The thrust is :", T)
        print (" The ground run distance is :", S)

"""

























"""
Part No used 


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












"""


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

class AircraftTakeoff:

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
    # Plotting forces, speed, and acceleration
    # =====================================================

    def plot_forces (self):

        dict= self.ground_run()
        L = dict["lift"]
        D= dict["drag"]
        T= dict["thrust"]
        S= dict["distance"]
        plt.figure()
        plt.plot(S, L, 'b-', linewidth=2, label='Lift (L)')
        plt.plot(S, D, 'r--', linewidth=2, label='Drag (D)')
        plt.plot(S, T, 'k-.', linewidth=2, label='Thrust (T)')
        plt.xlabel(" the distance ")
        plt.ylabel(" forces ")
        plt.legend()
        plt.grid
        plt.show ( )


    def plot_speed (self):

        dict= self.ground_run()
        V = dict["speed"]
        S= dict["distance"]
        plt.figure()
        plt.plot(S, V, 'b-', linewidth=2, label='Speed ')
        plt.xlabel(" the distance ")
        plt.ylabel(" Speed ")
        plt.legend()
        plt.grid
        plt.show ( )

    def plot_acceleration (self):

        dict= self.ground_run()
        A= dict["acceleration"]
        S= dict["distance"]
        plt.figure()
        plt.plot(S, A, 'r--', linewidth=2, label='Acceleration')
        plt.xlabel(" the distance ")
        plt.ylabel(" Accelaretion ")
        plt.legend()
        plt.grid
        plt.show ( )



    def summary(self):

        dict= self.ground_run()
        L = dict["lift"]
        D= dict["drag"]
        T= dict["thrust"]
        S= dict["distance"]
        V = dict["speed"]
        A= dict["acceleration"]

        self.plot_forces()
        self.plot_speed()
        self.plot_acceleration()


        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("To confirm, the parameters are :", self.p)
        print(" The name of the aicraft is :", self.p["name"])
        print(" The type of the engine is :", self.p["engine"])
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("The ground distance is :", S[-1])
        print(" The lift at the lift off point is :" , L[-1])
        print("The lift off speed is :",  V[-1])
        print("The drag during the lift off is :",  D[-1])
        print("The thrust during the lift off is :",  T[-1])
        print("The acceleration during the lift off is :",  A[-1])




















    
    



"""
  param = {  "W": 3400,  "Sw": 144.9,  "Clmax": 1.69,  "Cl": 0.5, "Cd": 0.0417, "g": 32.2, "mu": 0.04, "P": 310, "rho": 0.002378,  "T": 1169,
                      "n": 35,  "A1": 1.158e-2,  "A2": -5.277e-05,  "A3": 9.273e-08,  "A4": -6.21e-11 }

W : Weight 
Sw : Wing surface 
Clmax : Lift coefficient max 
Cl : 
Cd :
g : Gravity 
mu : Ground frictionnal coefficient
P :
rho :
T : Thrust at the begining
lambda : 
n : number of iteration
A1, A2, A3, A4  : is the thrust coefficient used in the formulat of the thurst  
 


"""

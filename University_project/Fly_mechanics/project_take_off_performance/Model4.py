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
Asumption | Vlof is assumed to be 1.1* stalling speed  
----------

"""

# Which exercices are being validated by my code 
"""
Example 18.7 ;          Notice:     the ground run distance after running the code is 1111.04 


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from matplotlib.figure import Figure

from conversion import *
from forces import *

class AircraftTakeoff4:
    """
    This class represents another way to compute the takeoff distance.
    """

    def __init__(self, params):
        """
        Initialize all parameters from the dictionary.
        """
        self.p = params
        

    def lift(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Cl"]

    def drag(self, V):
        return 0.5 * self.p["rho"] * V**2 * self.p["Sw"] * self.p["Cd"]

    def thrust_piston(self, V):
        if V == 0 or self.p["A1"] == self.p["A2"] == self.p["A3"] == self.p["A4"] == 0:
            return self.p["T"]
        cste = self.p["A1"] * V + self.p["A2"] * V**2 + self.p["A3"] * V**3 + self.p["A4"] * V**4
        return 550 * self.p["P"] * cste / (1.688 * V)

    def ground_run(self):
        """
        Ground run distance.
        """
        t = np.linspace(0, 17, self.p["n"])
        S, dS = np.zeros(self.p["n"]), np.zeros(self.p["n"])
        A, V = np.zeros(self.p["n"]), np.zeros(self.p["n"])
        D, L, T = np.zeros(self.p["n"]), np.zeros(self.p["n"]), np.zeros(self.p["n"])

        D[0], L[0] = 0, 0
        A[0] = 9.78

        for i in range(1, self.p["n"]):
            D[i] = self.drag(V[i - 1])
            L[i] = self.lift(V[i - 1])
            T[i] = self.thrust_piston(V[i - 1])

            A[i] = (self.p["g"] / self.p["W"]) * (
                T[i] - D[i] - self.p["mu"] * (self.p["W"] - L[i])
            )

            V[i] = V[i - 1] + A[i] * (t[i] - t[i - 1])

            dS[i] = (
                V[i - 1] * (t[i] - t[i - 1])
                + 0.5 * A[i] * (t[i] - t[i - 1]) ** 2
            )

            S[i] = S[i - 1] + dS[i]

        return {
            "times": t,
            "drag": D,
            "lift": L,
            "thrust": T,
            "acceleration": A,
            "speed": V,
            "distance": S,
        }

    def part2(self):
        """
        Rotation distance, climb distance, or transition distance.
        """
        W = self.p["W"]
        hoc = self.p["hoc"]

        dict_run = self.ground_run()
        V = dict_run["speed"]

        Vlo = V[len(V) - 1] * 1.15 / 1.1

        T = self.thrust_piston(Vlo)
        D = self.drag(Vlo)
        L = self.lift(Vlo)

        gamma_climb = np.arcsin(np.clip((T - D) / W, -1, 1))

        n = L / W
        R = Vlo**2 / ((n - 1) * self.p["g"])

        Sr = R * np.sin(gamma_climb)
        hr = R * (1 - np.cos(gamma_climb))

        if hr < hoc:
            S_C = (hoc - hr) / np.tan(gamma_climb)
            Sobs = Sr + S_C
        else:
            Sobs = np.sqrt(R**2 - (R - hoc) ** 2)
            S_C = 0

        dico = {
            "V_st * 1.15 is: ": Vlo,
            "T: ": T,
            "D: ": D,
            "L: ": L,
            "gamma_climb: ": gamma_climb,
            "n = L/W: ": n,
            "Radius: ": R,
        }

        return S_C, Sobs, dico

    def plot_forces(self):
        data = self.ground_run()

        L = data["lift"]
        D = data["drag"]
        T = data["thrust"]
        S = data["distance"]

        plt.figure()
        plt.plot(S, L, "b-", linewidth=2, label="Lift (L)")
        plt.plot(S, D, "r--", linewidth=2, label="Drag (D)")
        plt.plot(S, T, "k-.", linewidth=2, label="Thrust (T)")
        plt.xlabel("Distance")
        plt.ylabel("Forces")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_speed(self):
        data = self.ground_run()

        V = data["speed"]
        S = data["distance"]

        plt.figure()
        plt.plot(S, V, "b-", linewidth=2, label="Speed")
        plt.xlabel("Distance")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_acceleration(self):
        data = self.ground_run()

        A = data["acceleration"]
        S = data["distance"]

        plt.figure()
        plt.plot(S, A, "r--", linewidth=2, label="Acceleration")
        plt.xlabel("Distance")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid()
        plt.show()

    def set_result(self):
        data = self.ground_run()
        S = data["distance"]
        V = data["speed"]
        Sg = S[-1]
        Sr = self.part2()[0]
        Sc = self.part2()[1]
        S =  Sg + Sr + Sc
        return {
            "Ground run distance Sa is": round (Sg, 2),
            "The rotation distance Sr, also called the transition distance, is": round(Sr,2),
            "The climb distance Sc is": round (Sc,2),
            "The total distance S for takeoff is ": round(S,2 ),
            "Lift-off speed Vlo is": round( V[-1])
        }














"""
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
        print("The rotating distance also called the transition is  :",  self.part2()[0])
        print("The climbing distance is  :",  self.part2()[1])
        print("Some values are ", self.part2()[2])

        self.plot_forces()
        self.plot_speed()
        self.plot_acceleration()
"""



















    
    



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
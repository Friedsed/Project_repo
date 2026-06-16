"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

B2: Book used General Aviation Aicraft  Design ;
 Applied Methods and Procedures ; SNORRI GUDMUNDSSON

--------------------------------
Methode developped in this code | Ground run
 estimation using numerical Integration method ;       page 793 et 799
--------------------------------

------------
Advantages 1| : Handle all the Take-off problems but more complex;   
               Can model the braking distance to simulate an engine faillure
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

from Conversion import *
from Forces import *

class AircraftTakeoff4:
    """
    This class represents another way to compute the takeoff distance.
    """

    def __init__(self, params):
        self.p = params
        self.p["Ra"] = self.p["bw"]**2 / self.p["Sw"]

    def get_lift(self, V):
        q = 0.5 * self.p["rho"] * V**2
        return q * self.p["Sw"] * self.p["Cl"]

    def get_drag(self, V):
        q = 0.5 * self.p["rho"] * V**2
        h_w = self.p["hw"]
        b_w = self.p["bw"]
        phi = ((16 * h_w / b_w)**2) / (1 + (16 * h_w / b_w)**2)
        CL = self.p["Cl"]
        CD = self.p["Cdo"] + self.p["Cdol"] * CL + phi * (CL**2 / (np.pi * self.p["e"] * self.p["Ra"]))
        return q * self.p["Sw"] * CD

    def get_v_stall(self):
        W = self.p["W"]
        S = self.p["Sw"]
        CL_max = self.p["Clmax"]
        return np.sqrt((2 * W) / (self.p["rho"] * S * CL_max))

    def get_thrust(self, V):
        if V == 0 or self.p["A1"] == self.p["A2"] == self.p["A3"] == self.p["A4"] == 0:
            return self.p["T"]
        else:
            cste = (self.p["A1"] * V + self.p["A2"] * V**2 + self.p["A3"] * V**3 + self.p["A4"] * V**4 ) * 4.4482
            return 550 * self.p["P"] * cste / (1.688 * V)

    def get_v_lof(self):
        return 1.1 * self.get_v_stall()

    def ground_run(self, dt=0.25):
        W = self.p["W"]
        g = self.p["g"]
        mu = self.p["mu"]
        V_lof = self.get_v_lof()

        A, V, S, T, D, L = [], [], [], [], [], []
        v_i = a_i = t = 0
        s_i = 0  # FIX: initialize distance

        while v_i < V_lof:
            T_i = self.get_thrust(v_i)
            D_i = self.get_drag(v_i)
            L_i = self.get_lift(v_i)

            a_i = (g / W) * (T_i - D_i - mu * (W - L_i))
            s_i += v_i * dt + 0.5 * a_i * (dt**2)
            v_i += a_i * dt
            t += dt

            T.append(T_i)
            D.append(D_i)   # FIX: was L_i
            L.append(L_i)
            A.append(a_i)
            S.append(s_i)
            V.append(v_i)

            if t > 1000:
                break

        return {"s_i": s_i, "v_i": v_i, "t": t, "T": T, "D": D, "L": L, "A": A, "S": S, "V": V}

    def rotation_dist(self):
        V_lo = self.get_v_lof()
        V_hw = self.p.get("Vhw", 0.0)
        return (V_lo - V_hw) * self.p["tr"]

    def climb_dist(self):
        W = self.p["W"]
        h_obst = self.p["hoc"]
        Vs1 = self.get_v_stall()
        V_tr = 1.15 * Vs1

        T_tr = self.get_thrust(V_tr)
        D_tr = self.get_drag(V_tr)

        sin_gamma = (T_tr - D_tr) / W
        gamma = np.arcsin(np.clip(sin_gamma, -1, 1))

        q_tr = 0.5 * self.p["rho"] * V_tr**2
        L_s = q_tr * self.p["Sw"] * (0.9 * self.p["Clmax"])
        n = L_s / W

        R = V_tr**2 / (self.p["g"] * (n - 1))  # FIX: self.g → self.p["g"]

        S_R = R * np.sin(gamma)
        h_R = R * (1 - np.cos(gamma))

        if h_R < h_obst:
            S_C = (h_obst - h_R) / np.tan(gamma)
            S_obs = S_R + S_C
        else:
            S_obs = np.sqrt(R**2 - (R - h_obst)**2)
            S_C = 0

        return S_R, S_C, S_obs

    def set_result(self):
        data = self.ground_run()
        Sa = data["s_i"]
        Vlo = data["v_i"]
        Sr = self.rotation_dist()
        Sc = self.climb_dist()[2]
        S = Sa + Sc + Sr

        return {
            "Ground run distance Sa is": round(Sa, 2),
            "The climb distance Sc is": round(Sc, 2),
            "The rotation speed in m/s is :": round(Sr, 2),
            "The total distance S for takeoff is ": round(S, 2),
            "Lift-off speed Vlo is": Vlo
        }





    def plot_forces(self):
        data = self.ground_run()
        L = data["L"]
        D = data["D"]
        T = data["T"]
        S = data["S"]
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
        V = data["V"]
        S = data["S"]
        plt.figure()
        plt.plot(S, V, "b-", linewidth=2, label="Speed")
        plt.xlabel("Distance")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_acceleration(self):
        data = self.ground_run()
        A = data["A"]
        S = data["S"]
        plt.figure()
        plt.plot(S, A, "r--", linewidth=2, label="Acceleration")
        plt.xlabel("Distance")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid()
        plt.show()




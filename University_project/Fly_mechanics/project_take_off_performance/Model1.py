"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 04 juin 2026

B1: Book used Mechanic of flight by Warren

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
Example 3.10.2 ;    Page 350      Notice:     .......................
Example 3.10.1;     Page 347      Notice: ...................................


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
import pandas as pd
from forces import *
from tkinter import *
from matplotlib.figure import Figure



# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff1:

    def __init__(self, params):
        """
        Initialize all parameters from dictionary
        """
        self.p = params
        self.p["Ra"] = self.p["bw"]**2 / self.p["Sw"]
      



    # ========================================================
    # Aerodynamic functions
    # ========================================================

    def C_L_alpha_sect(self):
        return C_L_alpha_sect1( self.p["alpha"],  self.p["alpha_0"]  )

    def C_L_alpha(self):
        return C_L_alpha1( self.p["alpha"],  self.p["alpha_0"],  self.p["Ra"]  )

    def C_L_function(self):
         return C_L_function1(  self.p["alpha"],  self.p["alpha_0"], self.p["Ra"]  )

    def C_D_function(self):
        return C_D_function1(  self.p["Cdo"],  self.p["Cdol"],   self.p["Cl"],  self.p["hw"],   self.p["bw"],  self.p["e"],  self.p["Ra"]  )

    # ========================================================
    # Forces
    # ========================================================

    def lift(self, V):
        return lift1(  V,  self.p["Sw"],    self.p["Cl"]   )

    def drag(self, V):
        Cd = self.C_D_function()
        return drag1(  V,  self.p["Sw"],    Cd   )

    def thrust(self, V):
        return thrust1(  V,   self.p["T0"], self.p["T1"], self.p["T2"]    )

    def friction(self, V):
        L = self.lift(V)
        return friction( self.p["mu"],  self.p["W"],   L    )

    # ========================================================
    # K parameters
    # ========================================================

    def K0(self):
        return self.p["T0"] / self.p["W"] - self.p["mu"]

    def K1(self):
        return self.p["T1"] / self.p["W"]

    def K2(self):
        p = self.p
        Cd = self.C_D_function()

        return (  p["T2"] / p["W"]   + p["rho"] / (2 * p["W"] / p["Sw"])  * (p["Cl"] * p["mu"] - Cd)   )

    def Kr(self):
        return 4 * self.K0() * self.K2() - self.K1()**2

    # ========================================================
    # Intermediate functions
    # ========================================================

    def Vlo(self):
        p = self.p
        return 1.1 * np.sqrt(2 / p["Clmax"]) * np.sqrt(p["W"] / (p["Sw"] * p["rho"]))

    def fi(self, V):
        return self.K0() + self.K1() * V + self.K2() * V**2

    def dfi(self, V):
        return self.K1() + 2 * self.K2() * V

    # ========================================================
    # Kw
    # ========================================================

    def Kw(self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        k0 = self.K0()
        k1 = self.K1()
        k2 = self.K2()
        kr = self.Kr()
        fi = self.fi(Vi)
        fip1 = self.fi(Vip1)
        dfi = self.dfi(Vi)
        dfip1 = self.dfi(Vip1)
        if k2 == 0 and k1 == 0:
            return (Vip1 - Vi) / k0
        elif k1 != 0 and k2 == 0:
            return (1 / k1) * np.log(fip1 / fi)
        elif kr < 0:
            return (1 / np.sqrt(-kr)) * np.log( ((dfip1 - np.sqrt(-kr)) * (dfi + np.sqrt(-kr)))  / ((dfip1 + np.sqrt(-kr)) * (dfi - np.sqrt(-kr)))   )
        elif kr == 0:
            return (2 / dfi) - (2 / dfip1)
        else:
            return (2 / np.sqrt(kr)) * ( 1 / np.tan(dfip1 / np.sqrt(kr)) - 1 / np.tan(dfi / np.sqrt(kr))  )

    # ========================================================
    # Kt
    # ========================================================

    def Kt(self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        k0 = self.K0()
        k1 = self.K1()
        k2 = self.K2()
        fi = self.fi(Vi)
        fip1 = self.fi(Vip1)
        kw = self.Kw()
        if k2 == 0 and k1 == 0:
            return (Vip1**2 - Vi**2) / (2 * k0)
        elif k2 == 0 and k1 != 0:
            return (k0 / k1**2) * np.log(fi / fip1) + (Vip1 - Vi) / k1
        else:
            return (1 / (2 * k2)) * np.log(fip1 / fi) - (k1 * kw) / (2 * k2)

    # ========================================================
    # Distance
    # ========================================================

    def distance(self):
        return (self.Kt() - self.p["Vhw"] * self.Kw()) / self.p["g"]

    # ========================================================
    # Numerical integration
    # ========================================================

    def distance_integral(self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        f = lambda V: (  (-self.p["Vhw"] + V)   / (self.K0() + self.K1() * V + self.K2() * V**2)    )
        I, _ = quad(f, Vi, Vip1)
        
        return I / self.p["g"]

     # ========================================================
    # Distance above they earth
    # ========================================================
    def part(self, gamma):

        V_oc = 1.2 * np.sqrt((2 * self.p["W"]) / (self.p["Clmax"] * self.p["Sw"] * self.p["rho"]))
        V_lo = 1.1 * np.sqrt((2 * self.p["W"]) / (self.p["Clmax"] * self.p["Sw"] * self.p["rho"]))
        C_L_oc = (self.p["W"] * np.cos(gamma)) / (0.5 * self.p["rho"] * self.p["Sw"] * V_oc**2)
        coef = ( (16 * (self.p["hw"] + self.p["hoc"]) / self.p["bw"])**2  ) / ( 1 + (16 * (self.p["hw"] + self.p["hoc"]) / self.p["bw"])**2  )
        C_D_oc = (  self.p["Cdo"]  + self.p["Cdol"] * C_L_oc + coef * C_L_oc**2 / (np.pi * self.p["e"] * self.p["Ra"])   )
        D_oc = (  0.5 * self.p["rho"]  * V_oc**2     * self.p["Sw"]  * C_D_oc  )
        gamma = np.arcsin((self.p["T0"] - D_oc) / self.p["W"])
        # --- Phase lift-off --
        C_L_lo = self.p["W"] / (  0.5 * self.p["rho"] * V_lo**2 * self.p["Sw"]  )
        coeff = ( (16 * self.p["hw"] / self.p["bw"])**2  ) / (   1 + (16 * self.p["hw"] / self.p["bw"])**2  )
        C_D_lo = (   self.p["Cdo"]   + self.p["Cdol"] * C_L_lo  + coeff * C_L_lo**2 / (np.pi * self.p["e"] * self.p["Ra"])   )
        D_lo = (  0.5 * self.p["rho"]  * V_lo**2  * self.p["Sw"]  * C_D_lo  )
        
        return V_oc, V_lo, C_L_oc, C_D_oc, D_oc, gamma, C_L_lo, C_D_lo, D_lo


    def clearance_dist(self):
        list = self.part(0)
        list = self.part(list[5])
        V_oc = list[0]
        V_lo = list[1]
        D_oc = list[4]
        D_lo = list[8]
        gamma = list[5]
        coef_LO = self.p["T0"] - D_lo
        coef_OC = (self.p["T0"] - D_oc) / np.cos(gamma)
        F = (coef_LO + coef_OC) / 2
        Sc = ( (self.p["W"]) / F ) * (  self.p["hoc"] + (V_oc**2 - V_lo**2) / (2 * self.p["g"])    )   
        Sr = self.p["tr"]*V_lo
        Sa = self.distance_integral()
        St = Sa + Sr + Sc

        return Sa, Sr, Sc, St


    # ========================================================
    # Plot
    # ========================================================

    def plot_distance(self):

        list = self.part(0)
        list = self.part(list[5])
        list1 = self.clearance_dist()

        Vi = self.p["Vi"]
        #Vip1 = list[0]

        V = np.linspace(Vi, list[0], 100)

        S = np.array([   self.distance_integral_partial(v)    for v in V     ])

        plt.figure(figsize=(8,5))
        plt.plot(S, V)
        plt.xlabel("Distance ")
        plt.ylabel("Velocity")
        plt.title("Distance vs Velocity")
        plt.grid()
        plt.show()

    def plot_acceleration(self):

        list = self.part(0)
        list = self.part(list[5])
        list1 = self.clearance_dist()
        Vi = self.p["Vi"]
        Vip1 = list[0]
        v = np.linspace(Vi, Vip1, 100)
        a= ((self.K0() + self.K1() *v + self .K2() *v**2 ) /self.p["g"] )
        plt.figure()
        plt.plot(v,a)
        plt.xlabel("la vitesse ")
        plt.ylabel("l'acceleration ")
        plt.grid
        plt.show ( )

    def plot_forces (self):

        list = self.part(0)
        list = self.part(list[5])
        list1 = self.clearance_dist()
        Vi = self.p["Vi"]
        Vip1 = list[0]
        V = np.linspace(Vi, Vip1, 100)
        S = np.array([   self.distance_integral_partial(v)    for v in V     ])
        L = np.array([   self.lift(v)    for v in V     ])
        D= np.array([   self.drag(v)    for v in V     ])
        T= np.array([   self.thrust(v)    for v in V     ])
        F= np.array([   self.friction(v)    for v in V     ])
        plt.figure()
        plt.plot(S, L, 'b-', linewidth=2, label='Lift (L)')
        plt.plot(S, D, 'r--', linewidth=2, label='Drag (D)')
        plt.plot(S, T, 'k-.', linewidth=2, label='Thrust (T)')
        plt.plot(S, F, 'g:', linewidth=2, label='Friction (F)')
        plt.xlabel("la distance ")
        plt.ylabel(" forces ")
        plt.legend()
        plt.grid
        plt.show ( )


    def distance_integral_partial(self, v):
        f = lambda x: (  (-self.p["Vhw"] + x)  / (self.K0() + self.K1()*x + self.K2()*x**2)    )
        I, _ = quad(f, self.p["Vi"], v)
        return I / self.p["g"]

    # ========================================================
    # Summary
    # ========================================================


    def set_result(self):
        return { "Cd" : self.C_D_function()  , "K0": self.K0() ,  "K1" :  self.K1() , 
                    "K2" : self.K2() ,  "Kr": self.Kr(), "Vlo" :  self.Vlo() , 
                    "Kw" : self.Kw(), "Kt": self.Kt(), "Runaway distance is": self.distance_integral() ,
                     "The Rotation distance is":self.clearance_dist()[1] ,
                        "The Climb distance is": self.clearance_dist()[2] }

"""
    def summary(self):

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print(" To ensure the values   :", self.p)
        print(" The name of the aicraft is :", self.p["name"])
        print(" The type of the engine is :", self.p["engine"])
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("Cd =", self.C_D_function())
        print("K0 =", self.K0())
        print("K1 =", self.K1())
        print("K2 =", self.K2())
        print("Kr =", self.Kr())
        print("Vlo =", self.Vlo())
        print("Kw =", self.Kw())
        print("Kt =", self.Kt())
        print("Distance analytical =", self.distance())
        print("Distance integral =", self.distance_integral())
        print("The rotating distance is Sr =", self.clearance_dist()[1])
        print("The distance require to reach the obstacle is Sc = ", self.clearance_dist()[2] )
        print("the total distance to takeoff is then : ", self.clearance_dist()[3])

        self.plot_distance()
        self.plot_acceleration()
        self.plot_forces()


    def plot1(self):
        part_list = self.part(0)
        part_list = self.part(part_list[5])
        Vi = self.p["Vi"]
        Vip1 = part_list[0]
        v = np.linspace(Vi, Vip1, 100)
        a = (self.K0() + self.K1() * v + self.K2() * v**2) / self.p["g"]
        S = np.array([self.distance_integral_partial(x) for x in v])
        L = np.array([self.lift(x) for x in v])
        D = np.array([self.drag(x) for x in v])
        T = np.array([self.thrust(x) for x in v])
        F = np.array([self.friction(x) for x in v])
        fig, axs = plt.subplots(3, 2, figsize=(10, 8), dpi=100)
        axs[0, 0].plot(S, a); axs[0, 0].set_title("Acceleration"); axs[0, 0].grid(True)
        axs[0, 1].plot(S, v); axs[0, 1].set_title("Speed vs Distance"); axs[0, 1].grid(True)
        axs[1, 0].plot(S, L); axs[1, 0].set_title("Lift"); axs[1, 0].grid(True)
        axs[1, 1].plot(S, D); axs[1, 1].set_title("Drag"); axs[1, 1].grid(True)
        axs[2, 0].plot(S, T); axs[2, 0].set_title("Thrust"); axs[2, 0].grid(True)
        axs[2, 1].plot(S, F); axs[2, 1].set_title("Friction"); axs[2, 1].grid(True)
        plt.tight_layout()
        plt.show() 
"""

    








"""

Sw    # Wing surface area
bw    # Wing span
hw    # Height of the wing above the ground
W     # Weight
Ra    # Aspect ratio coefficient
Cdo   # Zero-lift drag coefficient
Cdol  # Additional drag coefficient term
Cd    # Drag coefficient
e     # Oswald efficiency coefficient
Clmax # Maximum lift coefficient
Cl    # Lift coefficient
mu    # Friction coefficient
Vlo   # Lift-off speed
Vhw   # Reference speed
Vi    # Initial speed for integration
Vip1  # Final speed for integration
T0    # Thrust coefficient determined experimentally
T1    # Thrust coefficient determined experimentally
T2    # Thrust coefficient determined experimentally

"""








































































"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 16 mai 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ============================================================
# Class AircraftTakeoff
# ============================================================

class AircraftTakeoff:

    def __init__(self, params, Unit):
        """
        Initialize all parameters from dictionary
        """
        self.p = params
        self.Unit=Unit

        self.CONV = {
        "Sw": 0.3048**2,        # ft² -> m²
        "bw": 0.3048,           # ft -> m
        "hw": 0.3048,           # ft -> m
        "W": 4.44822,           # lbf -> N
        "T0": 4.44822,          # lbf -> N
        "T1": 4.44822/0.3048,   # lbf/ft/s -> N/m/s
        "T2": 4.44822/(0.3048**2),
        "rho": 515.3788,        # slug/ft³ -> kg/m³
        "Vhw": 0.3048,          # ft/s -> m/s
        "Vi": 0.3048,           # ft/s -> m/s
        "g": 0.3048,            # ft/s² -> m/s²
    }

    #========================================================
    #Changing of unity from american to Internationnal system unit
    #========================================================

     # =====================================================
    # conversion factors
    # =====================================================
    



    def convert_units(self, new_unit="SI"):

        if self.Unit== new_unit:
            print(f"Already in {new_unit}")
            return

        # from US to SI
        if self.Unit == "US" and new_unit == "SI":
            for key, value in self.CONV.items():
                if key in self.p:
                    self.p[key] = self.p[key]* value

        # from SI to US
        elif self.Unit == "SI" and new_unit == "US":
            for key, value in self.CONV.items():
                if key in self.p:
                    self.p[key] = self.p[key] / value

        
        



    
    # ========================================================
    # Aerodynamic functions
    # ========================================================

    def C_L_alpha_sect(self):
        return 2 * np.pi * (self.p["alpha"] - self.p["alpha_0"])

    def C_L_alpha(self):
        cl_alpha_sect = self.C_L_alpha_sect()
        return cl_alpha_sect / (1 + cl_alpha_sect / (np.pi * self.p["Ra"]))

    def C_L_function(self):
        return self.C_L_alpha() * (self.p["alpha"] - self.p["alpha_0"])

    def C_D_function(self):
        p = self.p
        return ( p["Cdo"]  + p["Cdol"] * p["Cl"]  + ((16 * p["hw"] / p["bw"]) ** 2 * p["Cl"] ** 2) / ((1 + (16 * p["hw"] / p["bw"]) ** 2) * np.pi * p["e"] * p["Ra"])     )

    # ========================================================
    # Forces
    # ========================================================

    def lift(self, V):
        return 0.5 * V**2 * self.p["Sw"] * self.p["Cl"]

    def drag(self, V):
        return 0.5 * V**2 * self.p["Sw"] * self.C_D_function()

    def thrust(self, V):
        return self.p["T0"] + self.p["T1"] * V + self.p["T2"] * V**2

    def friction(self, V):
        return self.p["mu"] * (self.p["W"] - self.lift(V))

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
    # Plot
    # ========================================================

    def plot_distance(self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()

        V = np.linspace(Vi, Vip1, 100)

        S = np.array([   self.distance_integral_partial(v)    for v in V     ])

        plt.figure(figsize=(8,5))
        plt.plot(S, V)
        plt.xlabel("Velocity ")
        plt.ylabel("Distance ")
        plt.title("Distance vs Velocity")
        plt.grid()
        plt.show()

    def plot_acceleration(self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        v=np.linspace( Vi, Vip1, 100)
        a= ((self.K0() + self.K1() *v + self .K2() *v**2 ) /self.p["g"] )


        plt.figure()
        plt.plot(v,a)
        plt.xlabel("la vitesse ")
        plt.ylabel("l'acceleration ")
        plt.grid
        plt.show ( )

    def plot_forces (self):

        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        v=np.linspace( Vi, Vip1, 100)
        L = self.lift(v)
        D= self.drag(v)
        T= self.thrust(v)
        F= self.friction(v)

        plt.figure()
        plt.plot(v, L, 'b-', linewidth=2, label='Lift (L)')
        plt.plot(v, D, 'r--', linewidth=2, label='Drag (D)')
        plt.plot(v, T, 'k-.', linewidth=2, label='Thrust (T)')
        plt.plot(v, F, 'g:', linewidth=2, label='Friction (F)')

        plt.xlabel("la vitesse ")
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

    def summary(self):

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
        self.plot_distance()
        self.plot_acceleration()
        self.plot_forces()


# ============================================================
# MAIN
# ============================================================

params = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params1 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params2 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params3 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params4 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params5 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params6 = {"Sw": 180, "bw": 33, "hw": 6, "W": 2700,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.4,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }
params7 = {"Sw": 144.9, "bw": 33, "hw": 6, "W": 3400,"Ra": 6.05, "Cdo": 0.036, "Cdol": 0.0,"e": 0.82, "Clmax": 1.69,"Cl": 0.3485,"mu": 0.04,"T0": 1200,"T1": -4, "T2": 0,"rho": 0.0023769, "alpha": 0, "alpha_0": 0,"Vhw": 29.33,"g": 32.2, "Vi": 29.33 }




plane = AircraftTakeoff(params7, Unit="US") # here the parameters are in US unit

#plane.convert_units("US")
#print(plane.p)

#plane.convert_units("SI")
#print(plane.p)

#plane_new=AircraftTakeoff(plane.p,  Unit="IS") # here i change the unit of the dictionnary (params) into IS unit that i used for a new AircraftTakeoff
plane.summary()


















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
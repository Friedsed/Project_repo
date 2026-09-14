

"""
Author: Kevin Larborde

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 14 juin 2026


"""




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
from tkinter import *
from matplotlib.figure import Figure

from Conversion import *
from Forces import *

class AircraftTakeoff5:

    def __init__(self, params):
        """
        Initialise les paramètres à partir du dictionnaire de données.
        """
        self.p = params
        self.p["Ra"] = self.p["bw"]**2 / self.p["Sw"]

    def C_L_alpha_sect(self):
        return C_L_alpha_sect1(self.p["alpha"], self.p["alpha_0"])

    def C_L_alpha(self):
        return C_L_alpha1(self.p["alpha"], self.p["alpha_0"], self.p["Ra"])

    def C_L_function(self):
        return C_L_function1(self.p["alpha"], self.p["alpha_0"], self.p["Ra"])

    def C_D_function(self):
        return C_D_function1(self.p["Cdo"], self.p["Cdol"], self.p["Cl"], self.p["hw"], self.p["bw"], self.p["e"], self.p["Ra"])

    def lift(self, V):
        return lift1(V, self.p["Sw"], self.p["Cl"])

    def drag(self, V):
        Cd = self.C_D_function()
        return drag1(V, self.p["Sw"], Cd)

    def thrust(self, V):
        return thrust1(V, self.p["T0"], self.p["T1"], self.p["T2"])

    def friction(self, V):
        L = self.lift(V)
        return friction(self.p["mu"], self.p["W"], L)

    def A(self):
       
        return ((self.p["T0"] / self.p["W"]) - self.p["mu"]) * self.p["g"]

    def B(self):
       
        Cd = self.C_D_function()
        
        # On extrait la constante k (unique paramètre de dégradation de la poussée)
        k_moteur = self.p["k"]
        
        bloc_aerodynamique = 0.5 * self.p["rho"] * self.p["Sw"] * (Cd - self.p["mu"] * self.p["Cl"]) + k_moteur
        return (self.p["g"] / self.p["W"]) * bloc_aerodynamique

    def Vlo(self):
        p = self.p
        return 1.1 * np.sqrt(2 / p["Clmax"]) * np.sqrt(p["W"] / (p["Sw"] * p["rho"]))


    def distance(self):
        
        A_val = self.A()
        B_val = self.B()
        Vlo_val = self.Vlo()
        
        denominateur = A_val - B_val * (Vlo_val**2)
        if denominateur <= 0:
            return float('inf')  # Sécurité si l'avion ne peut physiquement pas accélérer jusqu'à Vlo
            
        return (1 / (2 * B_val)) * np.log(A_val / denominateur)

    def temps_calculer(self, V):
       
        A_val = self.A()
        B_val = self.B()
        
        u = V * np.sqrt(B_val / A_val)
        if u >= 1:
            return float('inf')
        return (1 / np.sqrt(A_val * B_val)) * np.arctanh(u)

    def distance_integral(self):
       
        Vi = self.p["Vi"]
        Vip1 = self.Vlo()
        A_val = self.A()
        B_val = self.B()
        
        f = lambda V: V / (A_val - B_val * (V**2))
        I, _ = quad(f, Vi, Vip1)
        return I


    def part(self, gamma):
        V_oc = 1.2 * np.sqrt((2 * self.p["W"]) / (self.p["Clmax"] * self.p["Sw"] * self.p["rho"]))
        V_lo = self.Vlo()
        C_L_oc = (self.p["W"] * np.cos(gamma)) / (0.5 * self.p["rho"] * self.p["Sw"] * V_oc**2)
        coef = ((16 * (self.p["hw"] + self.p["hoc"]) / self.p["bw"])**2) / (1 + (16 * (self.p["hw"] + self.p["hoc"]) / self.p["bw"])**2)
        C_D_oc = (self.p["Cdo"] + self.p["Cdol"] * C_L_oc + coef * C_L_oc**2 / (np.pi * self.p["e"] * self.p["Ra"]))
        D_oc = (0.5 * self.p["rho"] * V_oc**2 * self.p["Sw"] * C_D_oc)
        gamma = np.arcsin((self.p["T0"] - D_oc) / self.p["W"])
        
        C_L_lo = self.p["W"] / (0.5 * self.p["rho"] * V_lo**2 * self.p["Sw"])
        coeff = ((16 * self.p["hw"] / self.p["bw"])**2) / (1 + ((16 * self.p["hw"]) / self.p["bw"])**2)
        C_D_lo = (self.p["Cdo"] + self.p["Cdol"] * C_L_lo + coeff * C_L_lo**2 / (np.pi * self.p["e"] * self.p["Ra"]))
        D_lo = (0.5 * self.p["rho"] * V_lo**2 * self.p["Sw"] * C_D_lo)
        
        return V_oc, V_lo, C_L_oc, C_D_oc, D_oc, gamma, C_L_lo, C_D_lo, D_lo

    def clearance_dist(self):
        res_part = self.part(0)
        res_part = self.part(res_part[5])
        V_oc = res_part[0]
        V_lo = res_part[1]
        D_oc = res_part[4]
        D_lo = res_part[8]
        gamma = res_part[5]
        
        coef_LO = self.p["T0"] - D_lo
        coef_OC = (self.p["T0"] - D_oc) / np.cos(gamma)
        F = (coef_LO + coef_OC) / 2
        
        Sc = ((self.p["W"]) / F) * (self.p["hoc"] + (V_oc**2 - V_lo**2) / (2 * self.p["g"]))   
        Sr = self.p["tr"] * V_lo
        Sa = self.distance()  # Course au sol issue de ton Éq (36)
        St = Sa + Sr + Sc

        return Sa, Sr, Sc, St


    def distance_integral_partial(self, v):
        """ Requis pour tracer la courbe de distance pas à pas """
        A_val = self.A()
        B_val = self.B()
        f = lambda x: x / (A_val - B_val * (x**2))
        I, _ = quad(f, self.p["Vi"], v)
        return I

    def plot_distance(self):
        res_part = self.part(0)
        res_part = self.part(res_part[5])
        Vi = self.p["Vi"]
        V = np.linspace(Vi, res_part[0], 100)
        
        S = np.array([self.distance_integral_partial(v) for v in V])

        plt.figure(figsize=(8, 5))
        plt.plot(S, V, color='tab:orange', lw=2, label="Modèle 3 (Analytique Logarithmique)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Vitesse (m/s)")
        plt.title("Modèle 3 : Vitesse en fonction de la Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_acceleration(self):
        res_part = self.part(0)
        res_part = self.part(res_part[5])
        Vi = self.p["Vi"]
        Vip1 = res_part[0]
        v = np.linspace(Vi, Vip1, 100)
        
        # Accélération non-constante liée à l'équation differentielle : a = A - B*V^2
        a = (self.A() - self.B() * (v**2)) / self.p["g"]
        
        plt.figure()
        plt.plot(v, a, color='tab:red', lw=2)
        plt.xlabel("Vitesse (m/s)")
        plt.ylabel("Accélération (g)")
        plt.title("Modèle 3 : Profil d'Accélération (A - B*V²)")
        plt.grid(True)
        plt.show()

    def plot_forces(self):
        res_part = self.part(0)
        res_part = self.part(res_part[5])
        Vi = self.p["Vi"]
        Vip1 = res_part[0]
        V = np.linspace(Vi, Vip1, 100)
        
        S = np.array([self.distance_integral_partial(v) for v in V])
        L = np.array([self.lift(v) for v in V])
        D = np.array([self.drag(v) for v in V])
        T = np.array([self.thrust(v) for v in V])
        F = np.array([self.friction(v) for v in V])
        
        plt.figure()
        plt.plot(S, L, 'b-', linewidth=2, label='Portance (L)')
        plt.plot(S, D, 'r--', linewidth=2, label='Traînée (D)')
        plt.plot(S, T, 'k-.', linewidth=2, label='Poussée (T)')
        plt.plot(S, F, 'g:', linewidth=2, label='Friction (F)')
        plt.xlabel("Distance (m)")
        plt.ylabel("Forces (N)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot1(self):
        """ Synthèse multi-graphiques identique à ton Modèle 1 """
        part_list = self.part(0)
        part_list = self.part(part_list[5])
        Vi = self.p["Vi"]
        Vip1 = part_list[0]
        v = np.linspace(Vi, Vip1, 100)
        
        a = (self.A() - self.B() * v**2) / self.p["g"]
        S = np.array([self.distance_integral_partial(x) for x in v])
        L = np.array([self.lift(x) for x in v])
        D = np.array([self.drag(x) for x in v])
        T = np.array([self.thrust(x) for x in v])
        F = np.array([self.friction(x) for x in v])
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 8), dpi=100)
        axs[0, 0].plot(S, a, 'tab:red'); axs[0, 0].set_title("Accélération (g)"); axs[0, 0].grid(True)
        axs[0, 1].plot(S, v, 'tab:orange'); axs[0, 1].set_title("Vitesse vs Distance"); axs[0, 1].grid(True)
        axs[1, 0].plot(S, L, 'b'); axs[1, 0].set_title("Portance (L)"); axs[1, 0].grid(True)
        axs[1, 1].plot(S, D, 'r'); axs[1, 1].set_title("Traînée (D)"); axs[1, 1].grid(True)
        axs[2, 0].plot(S, T, 'k'); axs[2, 0].set_title("Poussée (T)"); axs[2, 0].grid(True)
        axs[2, 1].plot(S, F, 'g'); axs[2, 1].set_title("Friction (F)"); axs[2, 1].grid(True)
        plt.tight_layout()
        plt.show()


    def set_result(self):
       
        return {
            "Ground run distance Sa is": self.distance(),
            "The rotation distance Sr, also called the transition distance, is": self.clearance_dist()[1],
            "The climb distance Sc is": self.clearance_dist()[2] 
        }

    def summary(self):
        print(" Paramètres appliqués :", self.p)
        print(" Nom de l'aéronef :", self.p.get("name", "Inconnu"))
        print(" Type de motorisation :", self.p.get("engine", "Hélice / Turboprop"))
        print("Cd =", self.C_D_function())
        print("Constante A (Équation 18) =", self.A())
        print("Constante B (Équation 19, incluant k) =", self.B())
        print("Vlo =", self.Vlo())
        print("Distance de roulement analytique (Équation 36) =", self.distance())
        print("Distance de roulement intégrale (Équation 32) =", self.distance_integral())
        print("La distance de rotation Sr =", self.clearance_dist()[1])
        print("La distance sous l'obstacle Sc = ", self.clearance_dist()[2])
        print("La distance totale de décollage St = ", self.clearance_dist()[3])

        self.plot_distance()
        self.plot_acceleration()
        self.plot_forces()

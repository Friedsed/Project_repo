# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

Partie 3 : Modèle balistique avec traînée, portance et poussée
Résolution par EDO (odeint)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.integrate import odeint

import colored_messages as cm
import constantes as cs  # non utilisé ici mais conservé


class Model_3:
    """Modèle balistique avec traînée, portance et poussée (ODE)."""

    def __init__(self, params):
        """
        params doit contenir :
        - h, v_0, alpha, npt
        - mass, rho, Cd, Cl, area, a
        """
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]

        self.mass = params["mass"]
        self.rho = params["rho"]
        self.Cd = params["Cd"]
        self.Cl = params["Cl"]
        self.area = params["area"]
        self.a = params["a"]

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.Cx, self.Cz = None, None
        self.impact_values = None

        # listes pour stocker plusieurs trajectoires
        self.list1 = [None, None]
        self.list2 = [None, None]
        self.list3 = [None, None]

        # grandeurs dérivées
        self.T0 = self.a * self.mass * g       # poussée (proportionnelle au poids)
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass          # accélération de poussée

    # ------------------------------------------------------------------
    def update_param(self, param1):
        """Met à jour tous les paramètres du modèle."""
        self.h = param1["h"]
        self.v_0 = param1["v_0"]
        self.alpha = np.deg2rad(param1["alpha"])
        self.npt = param1["npt"]
        self.mass = param1["mass"]
        self.rho = param1["rho"]
        self.Cd = param1["Cd"]
        self.Cl = param1["Cl"]
        self.area = param1["area"]
        self.a = param1["a"]

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

    # ------------------------------------------------------------------
    @staticmethod
    def initial_message():
        cm.set_title("Création d'une instance du modèle ODE (Model_3)")

    # ------------------------------------------------------------------
    def ode(self, y, t):
        """
        Système d’EDO avec traînée, portance et poussée.

        y = [x, z, v_x, v_z]
        """
        dy = np.zeros(4)
        dy[0] = y[2]
        dy[1] = y[3]
        v2 = y[2]**2 + y[3]**2
        theta = np.arctan2(y[3], y[2])

        # coefficients projetés dans la direction (x,z)
        Cx = -self.Cd * np.cos(theta) - self.Cl * np.sin(theta)
        Cz = self.Cl * np.cos(theta) - self.Cd * np.sin(theta)

        # équations
        dy[2] = self.beta * v2 * Cx + self.Ct * np.cos(theta)
        dy[3] = -g + self.beta * v2 * Cz + self.Ct * np.sin(theta)

        return dy

    # ------------------------------------------------------------------
    def solve_trajectory(self, alpha, t_end):
        """Résout la trajectoire pour un angle alpha (°) jusqu’à t_end."""
        self.t = np.linspace(0.0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)

        y_init = [0,
                  self.h,
                  self.v_0 * np.cos(self.alpha),
                  self.v_0 * np.sin(self.alpha)]
        y = odeint(self.ode, y_init, t=self.t)

        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        self.v = np.sqrt(self.v_x**2 + self.v_z**2)

    # ------------------------------------------------------------------
    def plot_trajectory(self):
        """Trace une trajectoire (numérique) x–z."""
        plt.figure()
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.legend(["Position Z en fonction de la position X"], fontsize=12)
        plt.grid(True)
        plt.title("Trajectoire (Model_3)")
        plt.show()

    # ------------------------------------------------------------------
    def plot_trajectories(self, param1, param2, param3, listC, t_end):
        """
        Trace 3 trajectoires pour trois jeux de paramètres (Cl, Cd).

        listC = [Cl1, Cd1, Cl2, Cd2, Cl3, Cd3]
        """
        # 1ère trajectoire
        self.update_param(param1)
        self.solve_trajectory(param1["alpha"], t_end)
        self.list1[0], self.list1[1] = self.x.copy(), self.z.copy()

        # 2ème trajectoire
        self.update_param(param2)
        self.solve_trajectory(param2["alpha"], t_end)
        self.list2[0], self.list2[1] = self.x.copy(), self.z.copy()

        # 3ème trajectoire
        self.update_param(param3)
        self.solve_trajectory(param3["alpha"], t_end)
        self.list3[0], self.list3[1] = self.x.copy(), self.z.copy()

        lab1 = r"$C_l, C_d$ : " + f"{listC[0]:.2f}, {listC[1]:.2f}"
        lab2 = r"$C_l, C_d$ : " + f"{listC[2]:.2f}, {listC[3]:.2f}"
        lab3 = r"$C_l, C_d$ : " + f"{listC[4]:.2f}, {listC[5]:.2f}"

        plt.figure()
        plt.plot(self.list1[0], self.list1[1], marker="+", color="blue",
                 linewidth=3, label=lab1)
        plt.plot(self.list2[0], self.list2[1], marker="+", color="red",
                 linewidth=3, label=lab2)
        plt.plot(self.list3[0], self.list3[1], marker="+", color="green",
                 linewidth=3, label=lab3)

        plt.title("Tracé de 3 trajectoires (Model_3)")
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    def validation(self, t_end, npt):
        cm.set_msg("Validation")

        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))

        # Courbe analytique
        self.time = np.linspace(0, t_end, self.npt)
        x = self.v_0 * np.cos(self.alpha) * self.time
        z = -(g * 0.5 * (self.time)**2) + self.v_0 * self.time * np.sin(self.alpha) + self.h

        ecart = [np.max(np.abs(x - self.x)),
                 np.max(np.abs(z - self.z))]

        if np.max(ecart) < 1e-7:
            print("La validation est vraie (erreur max < 1e-7)")
        else:
            print("Pas de validation (erreur trop grande)")

        print("L'erreur max est ", np.max(ecart))

        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3, label="Numérique" )
        plt.plot(x, z, marker="+", color="green", linewidth=3, label="Analytique" )
        plt.xlabel("Position X (m)", fontsize=12)
        plt.ylabel("Position Z (m)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.title("Comparaison Model_3 / Pésence de force aérodynamique et force de propulsion", fontsize=12)
        plt.show()


    # ------------------------------------------------------------------
    def set_reference_solution(self, t):
        """Solution analytique sans traînée au temps t."""
        x = self.v_0 * np.cos(self.alpha) * t
        z = self.h + self.v_0 * np.sin(self.alpha) * t - 0.5 * g * t**2
        return x, z

    # ------------------------------------------------------------------
    def set_impact_values(self):
        """
        Calcule les valeurs à l’impact (z=0) par interpolation linéaire.
        """
        def interpo(a, n, u):
            return u[n] + a * (u[n + 1] - u[n])

        # on cherche le premier passage de z de > 0 à <= 0
        n = None
        for i in range(len(self.z) - 1):
            if self.z[i] > 0.0 and self.z[i + 1] <= 0.0:
                n = i
                break

        if n is None:
            raise ValueError("Aucun impact détecté : z ne devient jamais négatif ou nul.")

        # paramètre d’interpolation pour z_i = 0
        a = -self.z[n] / (self.z[n + 1] - self.z[n])

        # temps et position à l'impact
        t_i = interpo(a, n, self.t)
        x_i = interpo(a, n, self.x)
        z_i = 0.0

        # vitesses à l'impact
        v_x = interpo(a, n, self.v_x)
        v_z = interpo(a, n, self.v_z)
        v = np.sqrt(v_x**2 + v_z**2)

        # angle de la vitesse
        theta_i = np.arctan2(v_z, v_x)

        self.impact_values = {
            "t_i": t_i,
            "p": x_i,
            "z_i": z_i,
            "angle": np.rad2deg(theta_i),
            "v": [v_x, v_z, v],
        }
        return self.impact_values

    # ------------------------------------------------------------------
    def get_parameters(self):
        """Affiche les paramètres du modèle."""
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))
        print("mass       : %.2f kg" % self.mass)
        print("rho        : %.2f kg/m^3" % self.rho)
        print("area       : %.2f m^2" % self.area)
        print("Cd         : %.2f" % self.Cd)
        print("Cl         : %.2f" % self.Cl)

    # ------------------------------------------------------------------
    def get_impact_values(self):
        """Affiche joliment les valeurs d’impact."""
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])

    # ------------------------------------------------------------------
    def plot_contour(self, alphaC, CdC, param_base, t_end):
        """
        Trace des contours de portée R(α, Cd).

        alphaC : liste d’angles (°)
        CdC    : liste de coefficients de traînée
        param_base : dict base contenant tous les autres paramètres
        """
        R = np.zeros((len(alphaC), len(CdC)))

        for i, alphai in enumerate(alphaC):
            for j, Cdj in enumerate(CdC):
                param_new = {
                    "v_0": param_base["v_0"],
                    "h": param_base["h"],
                    "npt": param_base["npt"],
                    "a": param_base["a"],
                    "area": param_base["area"],
                    "mass": param_base["mass"],
                    "Cl": param_base["Cl"],
                    "rho": param_base["rho"],
                    "alpha": alphai,
                    "Cd": Cdj,
                }
                self.update_param(param_new)  # mettre à jour les parametre du model pour avoir un nouveau model
                self.solve_trajectory(alphai, t_end)  # resoudre pour avoir les valeurs de position en x, z les vitesse et autre 
                impact = self.set_impact_values()  # recuperer les valeurs d'impact
                R[i, j] = impact["p"]   #  Sauvegarder les valeurs de l impact en fonction de l angle alpha et puis de Cdj

        CD, A = np.meshgrid(CdC, alphaC)
        fig, ax = plt.subplots()
        cont = ax.contourf(A, CD, R, levels=20, cmap="jet")

        cbar = fig.colorbar(cont, ax=ax)
        cbar.set_label("Portée R [m]")

        ax.set_ylabel(r"$c_d$")
        ax.set_xlabel(r"$\alpha$ [deg]")
        ax.set_title("Contours de la portée dans le plan ($\\alpha$, $c_d$)")
        plt.show()

# -*- coding: utf-8 -*-
"""
Projet Balistique - Model_1
===========================

Auteur : C. Airiau
Date : 30/10/2023
Étudiant : L3 Mécanique Énergétique

Description :
-------------
Ce module propose un "modèle analytique" de trajectoire balistique sans frottement,
en prenant en compte la hauteur initiale.
Il permet de :
- Déterminer les valeurs d’impact (temps, portée, vitesse, angle) ;
- Tracer la trajectoire et ses composants ;
- Étudier la portée maximale et l’altitude maximale selon l’angle de tir.
"""

# ======================================================================
# Importation des modules nécessaires
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g  # Accélération de la pesanteur en m/s²


# ======================================================================
# Définition de la classe Model_1
# ======================================================================

class Model_1:
    """ Classe représentant le modèle analytique de trajectoire balistique. """

    # ------------------------------------------------------------------
    # Constructeur de la classe
    # ------------------------------------------------------------------
    def __init__(self, h=10, v_0=10, alpha=30):
        """
        Initialise une instance du modèle.

        Paramètres
        ----------
        h : float
            Hauteur initiale en mètres.
        v_0 : float
            Vitesse initiale en m/s.
        alpha : float
            Angle de tir en degrés.
        """
        self.h = h
        self.v_0 = v_0
        self.alpha = np.deg2rad(alpha)  # Conversion en radians

        # Variables internes (initialisées à None)
        self.impact_values = None
        self.time, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.t_1, self.h_1 = None, None

    # ==================================================================
    # MÉTHODES "SETTERS"
    # ==================================================================

    def update_parameters(self, params_dict):
        """ Met à jour les paramètres à partir d’un dictionnaire. """
        if params_dict is None:
            params_dict = dict(v_0=10, h=20, alpha=30)
        else:
            for key, value in params_dict.items():
                if key == "v_0":
                    self.v_0 = value
                elif key == "h":
                    self.h = value
                elif key == "alpha":
                    self.alpha = np.deg2rad(value)

        self.get_parameters()

    # ------------------------------------------------------------------
    def set_velocity(self, t):
        """Retourne les composantes et la norme de la vitesse au temps t."""
        v_z = self.v_0 * np.sin(self.alpha) - g * t
        v_x = self.v_0 * np.cos(self.alpha)
        v = np.sqrt(v_x**2 + v_z**2)
        return v_x, v_z, v

    # ------------------------------------------------------------------
    def set_impact_values(self):
        """ Calcule les valeurs au moment de l’impact. """
        var = self.v_0 * np.sin(self.alpha)
        t_i = (var + np.sqrt(var**2 + 2 * g * self.h)) / g

        v_x, v_z, v = self.set_velocity(t_i)
        x_i = v_x * t_i
        theta_i = np.rad2deg(np.arctan(v_z / v_x))

        self.impact_values = {
            "t_i": t_i,
            "p": x_i,
            "angle": theta_i,
            "v": [v_x, v_z, v],
        }
        return self.impact_values

    # ------------------------------------------------------------------
    def set_trajectory(self, t_end, npt):
        """
        Calcule les positions, vitesses et temps d'une trajectoire complète.

        Paramètres


        
        ----------
        t_end : float
            Temps final de la trajectoire.
        npt : int
            Nombre de points de discrétisation.
        """
        self.time = np.linspace(0, t_end, npt)
        self.x = self.v_0 * np.cos(self.alpha) * self.time
        self.z = self.h + self.v_0 * np.sin(self.alpha) * self.time - 0.5 * g * self.time**2

        self.v_x = self.v_0 * np.cos(self.alpha)
        self.v_z = self.v_0 * np.sin(self.alpha) - g * self.time
        self.v = np.sqrt(self.v_x**2 + self.v_z**2)

    # ------------------------------------------------------------------
    def set_trajectories(self, alphas):
        """
        Calcule plusieurs trajectoires pour une liste d’angles en degrés.
        Retourne les listes des x, z et vitesses correspondantes.
        """
        list_x, list_z, list_v_x, list_v_z, list_v = [], [], [], [], []

        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            self.set_trajectory(impact_values["t_i"], npt=1001)

            list_x.append(self.x)
            list_z.append(self.z)
            list_v_x.append(self.v_x)
            list_v_z.append(self.v_z)
            list_v.append(self.v)

        return list_x, list_z, list_v_x, list_v_z, list_v

    # ==================================================================
    # MÉTHODES "GETTERS"
    # ==================================================================

    def get_impact_values(self):
        """ Affiche les valeurs d’impact de manière lisible. """
        print("==== Valeurs à l’impact ====")
        print(f"Temps        : {self.impact_values['t_i']:.2f} s")
        print(f"Portée       : {self.impact_values['p']:.2f} m")
        print(f"Angle        : {self.impact_values['angle']:.2f} °")
        print(f"|v|          : {self.impact_values['v'][2]:.2f} m/s")

    def get_parameters(self):
        """ Affiche les paramètres actuels du modèle. """
        print("==== Paramètres du Modèle ====")
        print(f"v₀           : {self.v_0:.2f} m/s")
        print(f"h            : {self.h:.2f} m")
        print(f"α            : {np.rad2deg(self.alpha):.2f} °")

    # ==================================================================
    # MÉTHODES DE TRAÇAGE / VISUALISATION
    # ==================================================================

    def plot_trajectory(self):
        """ Trace la trajectoire (Z en fonction de X). """
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X (m)", fontsize=12)
        plt.ylabel("Position Z (m)", fontsize=12)
        plt.title("Trajectoire balistique", fontsize=12)
        plt.legend(["Z en fonction de X"])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # ------------------------------------------------------------------
    def plot_component(self):
        """ Trace les composantes verticales et totales de la vitesse. """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("Composantes de la vitesse")

        ax1.plot(self.time, self.v_z, marker="+", color="red", linewidth=2)
        ax1.fill_between(self.time, self.v_z, alpha=0.3, color="red")
        ax1.set_ylabel("$v_z$ (m/s)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(self.time, self.v, marker="o", color="green", linewidth=2)
        ax2.fill_between(self.time, self.v, alpha=0.3, color="green")
        ax2.set_ylabel("$v$ (m/s)")
        ax2.set_xlabel("Temps (s)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # ------------------------------------------------------------------
    def plot_trajectories(self, x_list, z_list, alphas):
        """ Trace plusieurs trajectoires pour des angles différents. """
        plt.figure(figsize=(10, 6))
        for i, alpha in enumerate(alphas):
            plt.plot(x_list[i], z_list[i], label=f"α = {alpha}°", linewidth=3)

        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.title("Trajectoires pour différents angles de tir", fontsize=12)
        plt.legend(title="Angle (°)", loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    # ------------------------------------------------------------------
    def plot_maximum_distance(self, alphas):
        """ Trace la portée maximale en fonction de l’angle. """
        porter, porters = [], []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            porter.append(impact_values["p"])
          #  pf=(self.v_0**2 /g)*(  (np.sin(2*self.alpha)) / 2  + np.cos(self.alpha)*np.sqrt(    (np.sin(self.alpha))**2 + (2*g*self.h)/(self.v_0**2)))
           # porters.append(pf)
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, porter, color="blue", linewidth=3)
        #plt.plot(alphas, porters, color="red", linewidth=2)
        plt.xlabel("Angle (°)", fontsize=12)
        plt.ylabel("Portée maximale (m)", fontsize=12)
        plt.title("Portée maximale en fonction de l'angle", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    # ------------------------------------------------------------------
    def find_optimal_angle(self, alphas):
        """ Retourne l’angle et la portée correspondante maximisant la distance. """
        distances = []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            impact_values = self.set_impact_values()
            distances.append(impact_values["p"])

        optimal_index = np.argmax(distances)
        return alphas[optimal_index], distances[optimal_index]

    # ------------------------------------------------------------------
    def plot_maximum_height(self, alphas):
        """ Trace l’altitude maximale atteinte en fonction de l’angle. """
        heights = []
        for alpha in alphas:
            self.alpha = np.deg2rad(alpha)
            t_1 = (self.v_0 * np.sin(self.alpha)) / g
            h_1 = self.h + (self.v_0 * np.sin(self.alpha) * t_1) - (0.5 * g * t_1**2)
            heights.append(h_1)

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, heights, color="orange", linewidth=3)
        plt.xlabel("Angle (°)", fontsize=12)
        plt.ylabel("Altitude maximale (m)", fontsize=12)
        plt.title("Altitude maximale en fonction de l’angle", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

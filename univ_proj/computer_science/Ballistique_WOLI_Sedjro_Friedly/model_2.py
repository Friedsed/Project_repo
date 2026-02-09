# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

Partie 2 : Résolution ODE du problème balistique
Implémentation du modèle 1 sous forme ODE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g          # accélération de la pesanteur [m/s²]
from scipy.integrate import odeint     # solveur d’EDO

import colored_messages as cm          # gestion des messages colorés (module externe)
import constantes as cs                # constantes éventuelles (module externe)


class Model_2:
    """Modèle balistique résolu par EDO (odeint)."""

    def __init__(self, params):
        """
        Constructeur de la classe.

        Paramètres
        ----------
        params : dict
            Dictionnaire contenant :
            - "h"    : hauteur initiale [m]
            - "v_0"  : vitesse initiale [m/s]
            - "alpha": angle de tir [°]
            - "npt"  : nombre de points de discrétisation temporelle
        """
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.impact_values = None

    # ------------------------------------------------------------------
    @staticmethod
    def initial_message():
        """Affiche un message d’initialisation."""
        cm.set_title("Création d'une instance du modèle ODE (exemple d'apprentissage)")

    # ------------------------------------------------------------------
    def ode(self, y, t):
        """
        Système d’EDO pour la balistique sans frottement.

        y = [x, z, v_x, v_z]
        """
        dy = np.zeros(4)
        dy[0] = y[2]        # dx / dt = v_x
        dy[1] = y[3]        # dz / dt = v_z
        dy[2] = 0.0         # dv_x / dt = 0
        dy[3] = -g          # dv_z / dt = -g
        return dy

    # ------------------------------------------------------------------
    def solve_trajectory(self, alpha=30, t_end=1.0):
        """
        Résout la trajectoire dans un champ de pesanteur.

        Paramètres
        ----------
        alpha : float
            Angle de tir en degrés (paramétrique).
        t_end : float
            Temps final de calcul (approximation de la durée de vol).
        """
        # Discrétisation temporelle
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)

        # Conditions initiales : [x0, z0, v_x0, v_z0]
        y_init = [0,
                self.h,
                self.v_0 * np.cos(self.alpha),
                self.v_0 * np.sin(self.alpha)]

        # Résolution numérique de l’EDO
        y = odeint(self.ode, y_init, t=self.t)

        # Extraction des composantes
        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        self.v = np.sqrt(self.v_x**2 + self.v_z**2)

    # ------------------------------------------------------------------
    def plot_trajectory(self):
        """Trace la trajectoire (z en fonction de x)."""
        plt.figure()
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X (m)", fontsize=12)
        plt.ylabel("Position Z (m)", fontsize=12)
        plt.title("Trajectoire balistique (modèle ODE)", fontsize=12)
        plt.legend(["Z en fonction de X"], fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # ------------------------------------------------------------------
    def validation(self, t_end, npt):
        """
        Valide la solution numérique par comparaison avec la solution analytique.

        On compare ici :
        - la position (x, z) au dernier instant,
        - l’écart maximal sur toute la trajectoire.
        """
        print("Validation de la solution ODE")

        # Référence analytique au dernier temps calculé
        print(f"Solution analytique au temps t = {self.t[-1]:.6f} s")
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x_ref, z_ref               : %f  %f" % (x_ref, z_ref))

        print("Solution numérique au même temps :")
        print("x_num, z_num               : %f  %f" % (self.x[-1], self.z[-1]))

        # Recalcule une trajectoire analytique discrétisée pour comparer point à point
        time = np.linspace(0.0, t_end, npt)
        x_th = self.v_0 * np.cos(self.alpha) * time
        z_th = self.h + self.v_0 * np.sin(self.alpha) * time - 0.5 * g * time**2

        # Erreur maximale sur x et z
        # (on tronque éventuellement pour avoir la même longueur)
        n_min = min(len(time), len(self.t))
        ecart_x = np.max(np.abs(x_th[:n_min] - self.x[:n_min]))
        ecart_z = np.max(np.abs(z_th[:n_min] - self.z[:n_min]))
        ecart = [ecart_x, ecart_z]

        if np.max(ecart) < 1e-7:
            print("VALIDATION : OK (erreur max < 1e-7)")
        else:
            print("VALIDATION : ECHEC (erreur trop grande)")

        print("Erreur max (x, z)          : ", np.max(ecart))

        # Tracé comparatif
        plt.figure()
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=2, label="Numérique (ODE)")
        plt.plot(x_th, z_th, marker="x", color="green", linewidth=2, label="Analytique")
        plt.xlabel("Position X (m)", fontsize=12)
        plt.ylabel("Position Z (m)", fontsize=12)
        plt.title("Comparaison solution ODE / analytique", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # ------------------------------------------------------------------
    def set_reference_solution(self, t):
        """
        Solution analytique (modèle 1) au temps t.

        Retourne
        --------
        x, z : float
            Position analytique au temps t.
        """
        x = self.v_0 * np.cos(self.alpha) * t
        z = self.h + self.v_0 * np.sin(self.alpha) * t - 0.5 * g * t**2
        return x, z

    # ------------------------------------------------------------------
    def set_impact_values(self):
        """
        Détermine les paramètres à l’impact par interpolation linéaire.

        Méthode :
        ---------
        1) On repère le premier indice n tel que z[n] > 0 et z[n+1] <= 0.
        2) On interpole linéairement entre les instants t[n] et t[n+1]
           pour obtenir le temps exact d’impact t_i.
        3) On interpole également x, z, v_x, v_z à cet instant.
        4) On calcule l’angle d’impact à partir de v_x et v_z.
        """
        # 1) Trouver le premier indice où la trajectoire passe au sol
        n = None
        for i in range(len(self.z) - 1):
            if self.z[i] > 0.0 and self.z[i + 1] <= 0.0:
                n = i
                break

        if n is None:
            raise ValueError("Aucun impact détecté : z ne devient jamais négatif ou nul.")

        # 2) Coefficient d’interpolation linéaire (z = 0)
        a = -self.z[n] / (self.z[n + 1] - self.z[n])

        def interpo(a_loc, n_loc, u):
            """Interpolation linéaire entre u[n_loc] et u[n_loc+1]."""
            return u[n_loc] + a_loc * (u[n_loc + 1] - u[n_loc])

        # 3) Interpolation des grandeurs au temps d’impact
        t_i = interpo(a, n, self.t)
        x_i = interpo(a, n, self.x)
        z_i = interpo(a, n, self.z)
        vx_i = interpo(a, n, self.v_x)
        vz_i = interpo(a, n, self.v_z)
        v_i = np.sqrt(vx_i**2 + vz_i**2)

        # 4) Angle d’impact (en degrés)
        theta_i = np.rad2deg(np.arctan2(vz_i, vx_i))

        # Stockage des valeurs
        self.impact_values = {
            "t_i": t_i,
            "p": x_i,
            "z_i": z_i,
            "angle": theta_i,
            "v": [vx_i, vz_i, v_i],
        }

        return self.impact_values

    # ------------------------------------------------------------------
    def get_parameters(self):
        """Affiche les paramètres du modèle."""
        cm.set_info("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))

    # ------------------------------------------------------------------
    def get_impact_values(self):
        """Affiche les valeurs d’impact de manière lisible."""
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])

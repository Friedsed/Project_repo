
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 14 juin 2026

Code Développer tous seul en me bassant sur le code proposé par l'IA pour le programme : Comparaision.py 

"""

import numpy as np



def C_L_alpha_sect1(alpha, alpha_0):
    return 2 * np.pi * (alpha - alpha_0)


def C_L_alpha1(alpha, alpha_0, Ra):
    cl_alpha_sect = C_L_alpha_sect1(alpha, alpha_0)
    return cl_alpha_sect / (1 + cl_alpha_sect / (np.pi * Ra))


def C_L_function1(alpha, alpha_0, Ra):
    return C_L_alpha1(alpha, alpha_0, Ra) * (alpha - alpha_0)


def C_D_function1(Cdo, Cdol, Cl, hw, bw, e, Ra):
    return (
        Cdo
        + Cdol * Cl
        + ((16 * hw / bw) ** 2 * Cl ** 2)
        / ((1 + (16 * hw / bw) ** 2) * np.pi * e * Ra)
    )



def lift1(V, Sw, Cl):
    return 0.5 * V**2 * Sw * Cl


def drag1(V, Sw, Cd):
    return 0.5 * V**2 * Sw * Cd


def thrust1(V, T0, T1, T2):
    return T0 + T1 * V + T2 * V**2

def lift2(rho, Vr, Sw, Cl):
    return 0.5 * rho * (Vr / np.sqrt(2))**2 * Sw * Cl

def drag2(rho, Vr, Sw, Cd):
    return 0.5 * rho * (Vr / np.sqrt(2))**2 * Sw * Cd


def thrust_piston2(efficiency, P, Vr):
    return efficiency * 550 * P * np.sqrt(2) / Vr


def thrust_jet_powered2(T):
    return T


def friction(mu, W, lift_value):
    return mu * (W - lift_value)



        # -*- coding: utf-8 -*-
"""
 
@author: Christophe Airiau


Module about atmosphere
h <= 11 000

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
# import matplotlib.ticker as ticker
from flight.general import GAMMA, R_GAS, G_REF, T_REF, DELTA_LETTER, SIGMA_LETTER, THETA_LETTER
 

P_SL = constants.value('standard atmosphere')
AIR_MOLAR_MASS = 28.9645e-3    # kg/mol
R_AIR = R_GAS / AIR_MOLAR_MASS
K_TEMP = -6.5/1000   # temperature gradient for an altitude h <= 11 000 m
T_SL_CELCIUS = 15
p_0, temp_0, rho_0, a_0 = [0, 0, 0, 0]

def display_atmospheric_constants():
    """
    Physical constantes
    """
    print("r (air)       : %.4f" % R_AIR)
    print("p_ref /r      : %.4f" % (P_SL / R_AIR))


def set_exponents(r_ref, k_temp):
    """
    exponents in the atmosphere model
    """
    n = -G_REF / (r_ref * k_temp)
    m = G_REF / r_ref * 1000
    print("n      : %.5f" % n)
    print("m      : %.5f" % m, "\t Gudmundsson: ",  34.163195)
    return n, m


def h_from_temperature(ratio, k_temp=K_TEMP, t0=T_SL_CELCIUS):
    """
    return the altitude in meters
    """
    kappa = k_temp / (T_REF + t0)
    return (ratio - 1) / kappa


def atmos(h, k_temp=K_TEMP, t0=T_SL_CELCIUS, n=5.255894):
    """
    standard atmosphere ISA 1976, ratios
    h in meters
    """
    kappa = k_temp / (T_REF + t0)
    theta = 1 + kappa * h
    delta = theta ** n
    sigma = theta ** (n-1)
    return theta, delta, sigma


def set_sea_level_state(r_ref, t_0=T_SL_CELCIUS, p_ref=P_SL):
    """
    reference state at sea level
    """
    p_0, temp_0 = p_ref, T_REF + t_0
    rho_0 = p_0 / (r_ref * temp_0)
    a_0 = np.sqrt(GAMMA * R_AIR * temp_0)
    print("rho_0         : %.4f kg/m^3" % rho_0)
    return p_0, temp_0, rho_0, a_0


def humidity(x):
    """
    rho/rho_0 with humidity, correction factor
    """
    return (1 + x) / (1 + 1.609 * x)


def density2humidity(r):
    """
    humidity x = f(r),  r = rho/rho_0
    """
    return (r - 1) / (1 - r * 1.609)


def plot_density_humidity():
    """
    plot: rho/rho_0 with humidity, correction factor
    """
    x = np.linspace(0, 1)
    plt.figure()
    plt.plot(x, humidity(x), lw=3)
    plt.xlabel("humidity rate")
    plt.ylabel(r"$\dfrac{\rho}{\rho_{std}}$")
    plt.grid()
    plt.title("humidity correction factor")


def solve_humidity(r_target=1):
    """
    get humidity from a given correction
    """
    r_min = humidity(1)
    
    if r_target < r_min:
        raise ValueError("Value under the limit of rho / rho_sl = %.3f"
                         % r_min)
    else:
        return density2humidity(r_target)


def display_atmosphere(h, delta, sigma, theta):
    """ 
    p/p_0, sigma / sigma_0, theta / theta_0 
    
    h in feets
    """
    n = 50
    print()
    print("=" * n)
    print("Standard atmosphere")
    title = "  h [ft]    p / p_0    rho / rho_0    T / T_0 "
    subtitle = " " * 15 + DELTA_LETTER + " " * 12 + SIGMA_LETTER + " " * 12 + THETA_LETTER
    print("=" * n)
    print(title)
    print(subtitle)
    print("-" * n)
    form = " %6d      %6.4f     %6.4f         %6.4f "
    print(form % (h, delta, sigma, theta))
    print("=" * n)

# MAIN


print("Atmosphere initialization")
display_atmospheric_constants()
p_0, temp_0, rho_0, a_0 = set_sea_level_state(R_AIR, t_0=T_SL_CELCIUS, p_ref=P_SL)

